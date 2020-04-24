"""
@create time: 2020.4.24
@author: llq
@function: custom in run_classifier of BERT code, link: https://github.com/google-research/bert/blob/master/run_classifier.py
"""
import tensorflow as tf
import pandas as pd
import tokenizers
from sklearn.model_selection import StratifiedKFold
import copy
import logging

logging.getLogger().setLevel(logging.INFO)


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """
  A single set of features of data.
  """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id_list,
                 sentiment_id,
                 texts,
                 idx_start=0,
                 idx_end=0,
                 selected_texts="",
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id_list = label_id_list
        self.sentiment_id = sentiment_id
        self.texts = texts
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.selected_texts = selected_texts
        self.is_real_example = is_real_example


class InputExample(object):
    """
  A single training/test example for simple sequence classification.
  """

    def __init__(self, guid, text_a, selected_text, text_b=None, sentiment=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      selected_text: string
      sentiment: int
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.selected_text = selected_text
        self.sentiment = sentiment


def _truncate_seq_pair(tokens_a, tokens_b, label_list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            label_list.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, max_seq_length, tokenizer, is_predicting):
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id_list=[0] * max_seq_length,
            sentiment_id=0,
            texts="",
            selected_texts="",
            is_real_example=False)

    tokens_a = tokenizer.encode(sequence=example.text_a).tokens[1:-1]
    tokens_b = [example.text_b]

    idx_start, idx_end = None, None
    if example.selected_text is not None:
        selected_texts_a = tokenizer.encode(sequence=example.selected_text).tokens[1:-1]
        # find the intersection between text and selected text
        for index in (i for i, c in enumerate(tokens_a) if c == selected_texts_a[0]):
            if tokens_a[index:index + len(selected_texts_a)] == selected_texts_a:
                idx_start = index
                idx_end = index + len(selected_texts_a)
                break

    label_id_list = [0] * len(tokens_a)
    if idx_start is not None and idx_end is not None:
        for char_idx in range(idx_start, idx_end):
            label_id_list[char_idx] = 1

        if idx_start >= max_seq_length - 3:
            idx_start = max_seq_length - 3 - 1
        if idx_end >= max_seq_length - 3:
            idx_end = max_seq_length - 3 - 1

        # add [CLS]
        idx_start += 1

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, label_id_list, max_seq_length - 3)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    label_id_list = [0] + label_id_list
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_id_list.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
            label_id_list.append(0)
    tokens.append("[SEP]")
    segment_ids.append(1)
    label_id_list.append(0)

    enc = tokenizer.encode(sequence=example.text_a, pair=example.text_b)
    input_ids = enc.ids

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_id_list.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_id_list) == max_seq_length

    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join([x for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if example.selected_text is not None:
            print("selected_text: %s" % (" ".join(selected_texts_a)))
            print("idx_start: %d" % idx_start)
            print("idx_end: %d" % idx_end)
        else:
            print("selected_text: None in test")
        print("label_id_list: %s" % " ".join([str(x) for x in label_id_list]))
        print("sentiment_id: %d" % example.sentiment)

    if not is_predicting:
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id_list=label_id_list,
            sentiment_id=example.sentiment,
            texts=example.text_a,
            idx_start=idx_start,
            idx_end=idx_end,
            selected_texts=example.selected_text,
            is_real_example=True)
    else:
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id_list=label_id_list,
            sentiment_id=example.sentiment,
            texts=example.text_a,
            is_real_example=True)
    return feature


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, max_seq_length, tokenizer, is_predicting):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, is_predicting)

        features.append(feature)
    return features


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_predicting):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_id_list = []
    all_sentiment_id = []
    all_texts = []
    all_idx_start = []
    all_idx_end = []
    all_selected_texts = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_id_list.append(feature.label_id_list)
        all_sentiment_id.append(feature.sentiment_id)
        all_texts.append(feature.texts)
        all_idx_start.append(feature.idx_start)
        all_idx_end.append(feature.idx_end)
        all_selected_texts.append(feature.selected_texts)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_id_list":
                tf.constant(all_label_id_list, shape=[num_examples, seq_length], dtype=tf.int32),
            "sentiment_id":
                tf.constant(all_sentiment_id, shape=[num_examples], dtype=tf.int32),
            "texts":
                tf.constant(all_texts, shape=[num_examples], dtype=tf.string),
            "idx_start":
                tf.constant(all_idx_start, shape=[num_examples], dtype=tf.int32),
            "idx_end":
                tf.constant(all_idx_end, shape=[num_examples], dtype=tf.int32),
            "selected_texts":
                tf.constant(all_selected_texts, shape=[num_examples], dtype=tf.string),
        })

        d = d.repeat()
        if not is_predicting:
            d = d.shuffle(100)

        d = d.batch(batch_size)
        return d

    return input_fn


# Load all files from a directory in a DataFrame.
def load_dataset(directory):
    data = pd.read_csv(directory)
    data.rename(columns={'text':'sentence'}, inplace=True)
    data['sentence'] = data['sentence'].astype(str)
    data['sentiment'] = data['sentiment'].astype(str)

    if "train" in directory:
        data['selected_text'] = data['selected_text'].astype(str)

    data['polarity'] = 0
    data.loc[data['sentiment'] == 'negative', 'polarity'] = 1
    data.loc[data['sentiment'] == 'positive', 'polarity'] = 2

    if 'train' in directory:
        data.drop(['textID'], axis=1, inplace=True)
    else:
        data.drop(['textID'], axis=1, inplace=True)
    return data


def process_data(hp):
    tokenizer = tokenizers.BertWordPieceTokenizer(hp.BERT_VOCAB, lowercase=True)

    train = load_dataset('/kaggle/input/my-data/train_process.csv')
    test = load_dataset('/kaggle/input/tweet-sentiment-extraction/test.csv')
    # train = train.sample(5000)
    # test = test.sample(5000)

    # train = train[train['polarity'].isin([1, 2])]
    # train.reset_index(inplace=True)
    sfolder = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for kfold_num, (train_idx, eval_idx) in enumerate(sfolder.split(train[hp.DATA_COLUMN], train[hp.polarity])):
        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        train_InputExamples = train.loc[train_idx].apply(lambda x:InputExample(guid=None,
                                                                                        # Globally unique ID for
                                                                                        # bookkeeping,
                                                                                        # unused in this example
                                                                                        text_a=x[hp.DATA_COLUMN],
                                                                                        selected_text=x[hp.selected_text],
                                                                                        text_b=x[hp.sentiment],
                                                                                        sentiment=x[hp.polarity]),
                                                     axis=1)

        eval_InputExamples = train.loc[eval_idx].apply(lambda x:InputExample(guid=None,
                                                                                      text_a=x[hp.DATA_COLUMN],
                                                                                      selected_text=x[hp.selected_text],
                                                                                      text_b=x[hp.sentiment],
                                                                                      sentiment=x[hp.polarity]), axis=1)
        break

    # print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = convert_examples_to_features(train_InputExamples, hp.MAX_SEQ_LENGTH, tokenizer, is_predicting=False)
    eavl_features = convert_examples_to_features(eval_InputExamples, hp.MAX_SEQ_LENGTH, tokenizer, is_predicting=False)

    return train_features, eavl_features


def process_test_data(hp):
    tokenizer = tokenizers.BertWordPieceTokenizer(hp.BERT_VOCAB, lowercase=True)
    test = load_dataset('/kaggle/input/tweet-sentiment-extraction/test.csv')
    test_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                                                 text_a=x[hp.DATA_COLUMN],
                                                                                 selected_text=None,
                                                                                 text_b=x[hp.sentiment],
                                                                                 sentiment=x[hp.polarity]), axis=1)
    test_features = convert_examples_to_features(test_InputExamples, hp.MAX_SEQ_LENGTH, tokenizer,
                                                                       is_predicting=True)
    return test_features


class BertQAModel(TFBertPreTrainedModel):
    """
    Bert Model
    """
    DROPOUT_RATE = 0.1
    NUM_HIDDEN_STATES = 2

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config, name="bert")
        self.concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(self.DROPOUT_RATE)
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            dtype='float32',
            name="qa_outputs")

    @tf.function
    def call(self, inputs, **kwargs):
        # outputs: Tuple[sequence, pooled, hidden_states]
        _, _, hidden_states = self.bert(inputs, **kwargs)

        hidden_states = self.concat([hidden_states[-i] for i in range(1, self.NUM_HIDDEN_STATES + 1)])

        hidden_states = self.dropout(hidden_states, training=kwargs.get("training", False))
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return start_logits, end_logits


def eval_decoded_texts_in_position(texts, predicted_labels_origin, sentiment_ids, tokenizer):
    """
    accroding to the preditction, decoded to the selected text
    :param texts:
    :param predicted_labels_origin:
    :param sentiment_ids:
    :param tokenizer:
    :return:
    """
    predicted_labels = copy.deepcopy(predicted_labels_origin)
    text_token_list = []
    for i, predicted_label in enumerate(predicted_labels):
        if 2 in predicted_label:
            start_idx = predicted_label.index(2)
            predicted_labels[i][start_idx] = 1
            continue
        else:
            first_1 = True
            for j, single_predicted_label in enumerate(predicted_label):
                if single_predicted_label == 1 and first_1:
                    first_1 = False
                elif single_predicted_label != 1 and not first_1:
                    predicted_labels[i][j] = 1
                elif single_predicted_label == 1 and not first_1:
                    first_1 = True
                    break

    decoded_texts = []
    for i, text in enumerate(texts):
        text_token_list.append(tokenizer.tokenize(text))
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
            start_pos = float("inf")
            end_pos = float("-inf")
            while j < max_len:
                if predicted_label_id[j] == 1:
                    if start_pos > j:
                        start_pos = j
                    if end_pos < j:
                        end_pos = j
                    selected_text.append(text_token[j])
                    j += 1

                else:
                    j += 1

            if start_pos != float("inf") or end_pos != float("-inf"):
                while start_pos > 0 and "##" in text_token[start_pos]:
                    start_pos -= 1
                    selected_text.insert(0, text_token[start_pos])

                end_pos += 1
                while end_pos < max_len and "##" in text_token[end_pos]:
                    selected_text.insert(len(selected_text), text_token[end_pos])
                    end_pos += 1

                decoded_texts.append(" ".join(selected_text).replace(" ##", ""))
            else:
                decoded_texts.append("")

    return decoded_texts, text_token_list


def train(model, dataset, loss_fn, optimizer):

    @tf.function
    def train_step(model, inputs, y_true, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
            loss = loss_fn(y_true[0], y_pred[0])
            loss += loss_fn(y_true[1], y_pred[1])
            scaled_loss = optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    epoch_loss = 0.
    for batch_num, sample in enumerate(dataset):
        loss, y_pred = train_step(model, sample[:3], sample[4:6], loss_fn, optimizer)

        epoch_loss += loss

        print("training ... batch {batch_num + 1:03d} : "
            f"train loss {epoch_loss / (batch_num + 1):.3f} ",
            end='\r')


def predict(model, dataset, loss_fn, optimizer):
    @tf.function
    def predict_step(model, inputs):
        return model(inputs)

    def to_numpy(*args):
        out = []
        for arg in args:
            if arg.dtype == tf.string:
                arg = [s.decode('utf-8') for s in arg.numpy()]
                out.append(arg)
            else:
                arg = arg.numpy()
                out.append(arg)
        return out

    # Initialize accumulators
    offset = tf.zeros([0, MAX_SEQUENCE_LENGTH, 2], dtype=tf.dtypes.int32)
    text = tf.zeros([0, ], dtype=tf.dtypes.string)
    selected_text = tf.zeros([0, ], dtype=tf.dtypes.string)
    sentiment = tf.zeros([0, ], dtype=tf.dtypes.string)
    pred_start = tf.zeros([0, MAX_SEQUENCE_LENGTH], dtype=tf.dtypes.float32)
    pred_end = tf.zeros([0, MAX_SEQUENCE_LENGTH], dtype=tf.dtypes.float32)

    for batch_num, sample in enumerate(dataset):
        print(f"predicting ... batch {batch_num + 1:03d}" + " " * 20, end='\r')

        y_pred = predict_step(model, sample[:3])

        # add batch to accumulators
        pred_start = tf.concat((pred_start, y_pred[0]), axis=0)
        pred_end = tf.concat((pred_end, y_pred[1]), axis=0)
        offset = tf.concat((offset, sample[3]), axis=0)
        text = tf.concat((text, sample[6]), axis=0)
        selected_text = tf.concat((selected_text, sample[7]), axis=0)
        sentiment = tf.concat((sentiment, sample[8]), axis=0)

    # pred_start = tf.nn.softmax(pred_start)
    # pred_end = tf.nn.softmax(pred_end)

    pred_start, pred_end, text, selected_text, sentiment, offset = \
        to_numpy(pred_start, pred_end, text, selected_text, sentiment, offset)

    return pred_start, pred_end, text, selected_text, sentiment, offset



hparame = Hparame()
parser = hparame.parser
hp = parser.parse_known_args()[0]

train_features, eavl_features = process_data(hp)

# get batch
train_input_fn = input_fn_builder(features=train_features, seq_length=hp.MAX_SEQ_LENGTH, is_predicting=False)
train_batches = train_input_fn(params={"batch_size": hp.BATCH_SIZE})
