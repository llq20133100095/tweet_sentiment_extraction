"""
@create time: 2020.4.14
@author: llq
@function: custom in run_classifier of BERT code, link: https://github.com/google-research/bert/blob/master/run_classifier.py
"""
import tensorflow as tf
from bert import tokenization


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
                 selected_texts="",
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id_list = label_id_list
        self.sentiment_id = sentiment_id
        self.texts = texts
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

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = [example.text_b]

    idx_start, idx_end = None, None
    if example.selected_text is not None:
        selected_texts_a = tokenizer.tokenize(example.selected_text)
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

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if example.selected_text is not None:
            tf.logging.info("selected_text: %s" % (" ".join(selected_texts_a)))
        else:
            tf.logging.info("selected_text: None in test")
        tf.logging.info("label_id_list: %s" % " ".join([str(x) for x in label_id_list]))
        tf.logging.info("sentiment_id: %d" % example.sentiment)

    if not is_predicting:
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id_list=label_id_list,
            sentiment_id=example.sentiment,
            texts=example.text_a,
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
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, is_predicting)

        features.append(feature)
    return features


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, drop_remainder, is_predicting):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_id_list = []
    all_sentiment_id = []
    all_texts = []
    all_selected_texts = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_id_list.append(feature.label_id_list)
        all_sentiment_id.append(feature.sentiment_id)
        all_texts.append(feature.texts)
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
            "selected_texts":
                tf.constant(all_selected_texts, shape=[num_examples], dtype=tf.string),
        })

        d = d.repeat()
        if not is_predicting:
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn
