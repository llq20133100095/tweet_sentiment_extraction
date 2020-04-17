# coding: utf-8
'''
@create time: 2020.4.16
@author: llq
@function: test model
'''

import os
import tensorflow as tf
from hparame import Hparame
from prepro import process_test_data, create_tokenizer_from_hub_module, process_data
import run_classifier_custom
from train import train_eval_test_model, eval_decoded_texts
from bert import modeling
from util import jaccard
import pandas as pd
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)
hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()
set_training = True

logging.info("# Prepare test batches")
test_features = process_test_data(hp)

test_input_fn = run_classifier_custom.input_fn_builder(features=test_features, seq_length=hp.MAX_SEQ_LENGTH,
                                                       drop_remainder=False, is_predicting=True)

test_batches = test_input_fn(params={"batch_size": hp.BATCH_SIZE})

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
features_input = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
label_list = [int(i) for i in hp.label_list.split(",")]
tokenizer = create_tokenizer_from_hub_module(hp)
bert_config = modeling.BertConfig.from_json_file(hp.BERT_CONFIG)
test_predicted_labels, test_sentiment_ids, test_texts, test_selected_texts = \
    train_eval_test_model(features=features_input, bert_config=bert_config, num_labels=len(label_list),
                          learning_rate=None, num_train_steps=None,
                          num_warmup_steps=None,
                          use_one_hot_embeddings=False, is_training=False, is_predicting=True)

logging.info("# Session")
test_len = len(test_features)
num_test_batches = test_len // hp.BATCH_SIZE + int(test_len % hp.BATCH_SIZE != 0)

with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.OUTPUT_DIR)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_  # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# test evalution")
    test_texts_list = []
    predicted_label_list = []
    test_sentiment_ids_list = []
    test_selected_texts_list = []
    for _ in range(num_test_batches):
        _eval_texts, _eval_predicted_labels, _eval_sentiment_ids, _eval_selected_texts \
            = sess.run([test_texts, test_predicted_labels, test_sentiment_ids, test_selected_texts])
        test_texts_list.extend(_eval_texts.tolist())
        predicted_label_list.extend(_eval_predicted_labels.tolist())
        test_sentiment_ids_list.extend(_eval_sentiment_ids.tolist())
        test_selected_texts_list.extend(_eval_selected_texts.tolist())
        # print(test_texts_list)

    logging.info("test nums %d " % len(predicted_label_list))

    # calculate the jaccards
    test_predict = eval_decoded_texts(test_texts_list[:test_len], predicted_label_list[:test_len],
                                      test_sentiment_ids_list[:test_len], tokenizer)
    # jaccards = []
    # for i in range(len(test_predict)):
    #     jaccards.append(jaccard(test_selected_texts_list[i], test_predict[i]))
    # score = np.mean(jaccards)
    # logging.info("jaccards: %f" % score)

    # print(test_predict[:23])
    submission_df = pd.read_csv('./input_data/sample_submission.csv')
    submission_df.loc[:, 'selected_text'] = test_predict
    submission_df.to_csv("./input_data/submission.csv", index=False)
