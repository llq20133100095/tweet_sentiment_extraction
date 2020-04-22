# coding: utf-8
import tensorflow as tf
from hparame import Hparame
from prepro import process_data, create_tokenizer_from_hub_module
from datetime import datetime
import run_classifier_custom
from bert import modeling
from bert import optimization
import logging
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from util import jaccard, eval_decoded_texts_in_position

logging.getLogger().setLevel(logging.INFO)
hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()
set_training = True


def create_model(bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids,
                 target_start_idx, target_end_idx, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    # "get_all_encoder_layers" for all encoder layer
    all_layer = model.get_all_encoder_layers()  # output_layer: 12 layer * [N, max_len, 768]

    hidden_size = all_layer[-1].shape[-1].value
    max_len = all_layer[-1].shape[1].value

    # Create our own layer to tune for politeness data. shape:[N, max_length, num_labels]
    with tf.variable_scope("first_softmax_llq", reuse=tf.AUTO_REUSE):
        output_weights = tf.get_variable("output_weights", [num_labels, 2 * hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.concat([all_layer[-1], all_layer[-2]], axis=-1)

        # Dropout helps prevent overfitting
        output_layer = tf.layers.dropout(output_layer, rate=0.1, training=is_training)

        # softmax operation
        logits = tf.einsum("nlh,hm->nlm", output_layer, tf.transpose(output_weights))
        logits = tf.nn.bias_add(logits, output_bias)
        logits_probs = tf.nn.log_softmax(logits, axis=-1)
        start_logits_probs, end_logits_probs = tf.split(logits_probs, 2, axis=-1)
        start_logits_probs = tf.squeeze(start_logits_probs)
        end_logits_probs = tf.squeeze(end_logits_probs)

        # Convert labels into one-hot encoding
        one_hot_start_idx = tf.one_hot(target_start_idx, depth=max_len, dtype=tf.float32)
        one_hot_end_idx = tf.one_hot(target_end_idx, depth=max_len, dtype=tf.float32)

        one_hot_start_labels = tf.one_hot(tf.argmax(start_logits_probs, axis=-1), depth=max_len, dtype=tf.int32, axis=-1)
        one_hot_end_labels = tf.one_hot(tf.argmax(end_logits_probs, axis=-1), depth=max_len, dtype=tf.int32, axis=-1)
        predicted_labels = one_hot_start_labels + one_hot_end_labels

        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, logits_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_start_loss = -tf.reduce_sum(one_hot_start_idx * start_logits_probs, axis=-1)
        per_example_end_loss = -tf.reduce_sum(one_hot_end_idx * end_logits_probs, axis=-1)
        loss = tf.reduce_mean(per_example_start_loss) + tf.reduce_mean(per_example_end_loss)
        return (loss, predicted_labels, logits_probs)


def train_eval_test_model(features, bert_config, num_labels, learning_rate, num_train_steps, num_warmup_steps,
                          use_one_hot_embeddings, is_training, is_predicting):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_id_list = features["label_id_list"]
    sentiment_ids = features["sentiment_id"]
    texts = features["texts"]
    selected_texts = features["selected_texts"]
    idx_start = features["idx_start"]
    idx_end = features["idx_end"]

    # TRAIN and EVAL
    if not is_predicting:
        (loss, predicted_labels, log_probs) = create_model(
            bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, idx_start, idx_end,
            num_labels, use_one_hot_embeddings)

        train_op = optimization.create_optimizer(
            loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

        global_step = tf.train.get_or_create_global_step()
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries, predicted_labels, sentiment_ids, texts, selected_texts, label_id_list, idx_start, idx_end

    else:
        (predicted_labels, log_probs) = create_model(
            bert_config, is_training, is_predicting, input_ids, input_mask, segment_ids, idx_start, idx_end, num_labels,
            use_one_hot_embeddings)
        return predicted_labels, sentiment_ids, texts, selected_texts


def eval_decoded_texts(texts, predicted_labels, sentiment_ids, tokenizer):
    decoded_texts = []
    for i, text in enumerate(texts):
        if type(text) == type(b""):
            text = text.decode("utf-8")
        # sentiment "neutral" or length < 2
        if sentiment_ids[i] == 0 or len(text.split()) < 2:
            decoded_texts.append(text)

        else:
            text_list = text.lower().split()
            text_token = tokenizer.tokenize(text)
            segment_id = []

            # record the segment id
            j_text = 0
            j_token = 0
            while j_text < len(text_list) and j_token < len(text_token):
                _j_token = j_token + 1
                text_a = "".join(tokenizer.tokenize(text_list[j_text])).replace("##", "")
                while True:
                    segment_id.append(j_text)
                    if "".join(text_token[j_token:_j_token]).replace("##", "") == text_a:
                        j_token = _j_token
                        break
                    _j_token += 1
                j_text += 1
            assert len(segment_id) == len(text_token)

            # get selected_text
            selected_text = []
            predicted_label_id = predicted_labels[i]
            predicted_label_id.pop(0)
            for _ in range(len(predicted_label_id) - len(text_token)):
                predicted_label_id.pop()

            max_len = len(predicted_label_id)
            assert len(text_token) == max_len
            j = 0
            while j < max_len:
                if predicted_label_id[j] == 1:
                    if j == max_len - 1:
                        j += 1
                    else:
                        a_selected_text = text_list[segment_id[j]]
                        selected_text.append(a_selected_text)
                        for new_j in range(j + 1, len(segment_id)):
                            if segment_id[j] != segment_id[new_j]:
                                j = new_j
                                break
                            elif new_j == len(segment_id) - 1:
                                j = new_j
                else:
                    j += 1

            decoded_texts.append(" ".join(selected_text))

    return decoded_texts


def eval_decoded_texts_no_id(texts, predicted_labels, sentiment_ids, tokenizer):
    """
    no use "record the segment id" in eval_decoded_texts
    :param texts: 
    :param predicted_labels: 
    :param sentiment_ids: 
    :param tokenizer: 
    :return: 
    """
    decoded_texts = []
    for i, text in enumerate(texts):
        if type(text) == type(b""):
            text = text.decode("utf-8")
        # sentiment "neutral" or length < 2
        if sentiment_ids[i] == 0 or len(text.split()) < 2:
            decoded_texts.append(text)

        else:
            text_token = tokenizer.tokenize(text)

            # get selected_text
            selected_text = []
            predicted_label_id = predicted_labels[i]
            predicted_label_id.pop(0)
            for _ in range(len(predicted_label_id) - len(text_token)):
                predicted_label_id.pop()

            max_len = len(predicted_label_id)
            assert len(text_token) == max_len
            j = 0
            while j < max_len:
                if predicted_label_id[j] == 1:
                    start_pos = j
                    end_pos = j + 1
                    while start_pos > 0 and "##" in text_token[j]:
                        start_pos -= 1
                    while end_pos < max_len and "##" in text_token[j]:
                        end_pos += 1

                    j = end_pos

                    selected_text.append("".join(text_token[start_pos:end_pos]).replace("##", ""))
                else:
                    j += 1

            decoded_texts.append(" ".join(selected_text))
    return decoded_texts


if __name__ == "__main__":
    train_features, eval_features = process_data(hp)
    tokenizer = create_tokenizer_from_hub_module(hp)

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / hp.BATCH_SIZE * hp.NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * hp.WARMUP_PROPORTION)
    num_train_batches = len(train_features) // hp.BATCH_SIZE + int(len(train_features) % hp.BATCH_SIZE != 0)
    num_eval_batches = len(eval_features) // hp.BATCH_SIZE + int(len(eval_features) % hp.BATCH_SIZE != 0)

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = run_classifier_custom.input_fn_builder(features=train_features, seq_length=hp.MAX_SEQ_LENGTH,
                                                            drop_remainder=False, is_predicting=False)

    eval_input_fn = run_classifier_custom.input_fn_builder(features=eval_features, seq_length=hp.MAX_SEQ_LENGTH,
                                                           drop_remainder=False, is_predicting=False)

    train_batches = train_input_fn(params={"batch_size": hp.BATCH_SIZE})
    eval_batches = eval_input_fn(params={"batch_size": hp.BATCH_SIZE})

    # create a iterator of the correct shape and type
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    features_input = iter.get_next()

    train_init_op = iter.make_initializer(train_batches)
    eval_init_op = iter.make_initializer(eval_batches)

    logging.info("# Load model")
    bert_config = modeling.BertConfig.from_json_file(hp.BERT_CONFIG)
    loss, train_op, global_step, train_summaries, eval_predicted_labels, eval_sentiment_ids, \
    eval_texts, eval_selected_texts, eval_label_id_list, eval_idx_start, eval_idx_end = train_eval_test_model(features=features_input, bert_config=bert_config,
                                                            num_labels=hp.num_label, learning_rate=hp.LEARNING_RATE,
                                                            num_train_steps=num_train_steps,
                                                            num_warmup_steps=num_warmup_steps,
                                                            use_one_hot_embeddings=False, is_training=set_training,
                                                            is_predicting=False)

    logging.info("# Session")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(hp.OUTPUT_DIR)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.OUTPUT_DIR, sess.graph)

        sess.run(train_init_op)
        _gs = sess.run(global_step)
        bleu_score = []

        test_len = len(train_features)
        for i in tqdm(range(_gs, num_train_steps + 1)):
            _, _gs, _summary, _loss = sess.run([train_op, global_step, train_summaries, loss])
            summary_writer.add_summary(_summary, _gs)

            if _gs and _gs == num_train_steps:
                ckpt_name = os.path.join(hp.OUTPUT_DIR, hp.model_output)
                saver.save(sess, ckpt_name, global_step=_gs)

            elif _gs and _gs % 500 == 0:
                logging.info("# Loss")
                logging.info(_loss)

                logging.info("# save models")
                ckpt_name = os.path.join(hp.OUTPUT_DIR, hp.model_output)
                saver.save(sess, ckpt_name, global_step=_gs)

                # logging.info("# train evaluation")
                # eval_texts_list = []
                # predicted_label_list = []
                # eval_sentiment_ids_list = []
                # eval_selected_texts_list = []
                # eval_label_id_list_list = []
                # all_idx_start = []
                # all_idx_end = []
                # for i in range(1):
                #     _eval_texts, _eval_predicted_labels, _eval_sentiment_ids, _eval_selected_texts, _eval_label_id_list, _eval_idx_start, _eval_idx_end \
                #         = sess.run([eval_texts, eval_predicted_labels, eval_sentiment_ids, eval_selected_texts, eval_label_id_list, eval_idx_start, eval_idx_end])
                #     eval_texts_list.extend(_eval_texts.tolist())
                #     predicted_label_list.extend(_eval_predicted_labels.tolist())
                #     eval_sentiment_ids_list.extend(_eval_sentiment_ids.tolist())
                #     eval_selected_texts_list.extend(_eval_selected_texts.tolist())
                #     eval_label_id_list_list.extend(_eval_label_id_list.tolist())
                #     all_idx_start.extend(_eval_idx_start.tolist())
                #     all_idx_end.extend(_eval_idx_end.tolist())
                #
                #
                # eval_predict, text_token_list = eval_decoded_texts_in_position(eval_texts_list, predicted_label_list, eval_sentiment_ids_list,
                #                                               tokenizer)
                #
                # jaccards = []
                # for i in range(len(eval_predict)):
                #     jaccards.append(jaccard(eval_selected_texts_list[i], eval_predict[i]))
                # score = np.mean(jaccards)
                # logging.info("jaccards: %f" % score)
                #
                # save_data = pd.DataFrame({"test_texts_list":eval_texts_list,
                #                           "text_token_list": text_token_list,
                #                           "selected_texts":eval_selected_texts_list,
                #                           "test_predict":eval_predict,
                #                           "predicted_label_list":predicted_label_list,
                #                           "eval_label_id_list_list":eval_label_id_list_list,
                #                           "all_idx_start":all_idx_start,
                #                           "all_idx_end":all_idx_end})
                # save_data.to_csv("./input_data/train_predict.csv", index=False)

                logging.info("# test evaluation")
                set_training = False
                sess.run([eval_init_op])

                eval_texts_list = []
                predicted_label_list = []
                eval_sentiment_ids_list = []
                eval_selected_texts_list = []
                for _ in range(num_eval_batches):
                    _eval_texts, _eval_predicted_labels, _eval_sentiment_ids, _eval_selected_texts \
                        = sess.run([eval_texts, eval_predicted_labels, eval_sentiment_ids, eval_selected_texts])
                    eval_texts_list.extend(_eval_texts.tolist())
                    predicted_label_list.extend(_eval_predicted_labels.tolist())
                    eval_sentiment_ids_list.extend(_eval_sentiment_ids.tolist())
                    eval_selected_texts_list.extend(_eval_selected_texts.tolist())

                logging.info("eval nums %d " % len(predicted_label_list))

                # calculate the jaccards
                eval_predict, text_token_list = eval_decoded_texts_in_position(eval_texts_list, predicted_label_list, eval_sentiment_ids_list,
                                                  tokenizer)
                jaccards = []
                for i in range(len(eval_predict)):
                    jaccards.append(jaccard(eval_selected_texts_list[i], eval_predict[i]))
                score = np.mean(jaccards)
                logging.info("jaccards: %f" % score)

                logging.info("# fall back to train mode")
                sess.run(train_init_op)
                set_training = True
