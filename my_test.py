#from prepro import create_tokenizer_from_hub_module
from hparame import Hparame
import numpy as np

hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()
set_training = True


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


import tokenizers

TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"./input_data/vocab.txt",
    lowercase=True
)
max_len = 128

def process_data(tweet, selected_text, sentiment):
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    len_st = len(selected_text)
    idx0 = -1
    idx1 = -1
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind: ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != -1 and idx1 != -1:
        for j in range(idx0, idx1 + 1):
            if tweet[j] != " ":
                char_targets[j] = 1
    print(tweet)
    print(char_targets)
    tok_tweet = TOKENIZER.encode(sequence=sentiment, pair=tweet)
    tok_tweet_tokens = tok_tweet.tokens
    tok_tweet_ids = tok_tweet.ids
    tok_tweet_offsets = tok_tweet.offsets[3:-1]
    print(tok_tweet_tokens)
    print(tok_tweet.offsets)

    # print(tok_tweet_tokens)
    # print(tok_tweet.offsets)
    # ['[CLS]', 'spent', 'the', 'entire', 'morning', 'in', 'a', 'meeting', 'w', '/',
    # 'a', 'vendor', ',', 'and', 'my', 'boss', 'was', 'not', 'happy', 'w', '/', 'them',
    # '.', 'lots', 'of', 'fun', '.', 'i', 'had', 'other', 'plans', 'for', 'my', 'morning', '[SEP]']
    targets = [0] * (len(tok_tweet_tokens) - 4)
    if sentiment == "positive" or sentiment == "negative":
        sub_minus = 8
    else:
        sub_minus = 7

    for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
        if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
            targets[j] = 1

    targets = [0] + [0] + [0] + targets + [0]

    # print(tweet)
    # print(selected_text)
    # print([x for i, x in enumerate(tok_tweet_tokens) if targets[i] == 1])
    targets_start = [0] * len(targets)
    targets_end = [0] * len(targets)

    non_zero = np.nonzero(targets)[0]
    if len(non_zero) > 0:
        targets_start[non_zero[0]] = 1
        targets_end[non_zero[-1]] = 1

    # print(targets_start)
    # print(targets_end)

    mask = [1] * len(tok_tweet_ids)
    token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

    padding_length = max_len - len(tok_tweet_ids)
    ids = tok_tweet_ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    targets = targets + ([0] * padding_length)
    targets_start = targets_start + ([0] * padding_length)
    targets_end = targets_end + ([0] * padding_length)

    new_sentiment = [1, 0, 0]
    if sentiment == "positive":
        new_sentiment = [0, 0, 1]
    if sentiment == "negative":
        new_sentiment = [0, 1, 0]

    return {
        'mask': mask,
        'token_type_ids': token_type_ids,
        'tweet_tokens': " ".join(tok_tweet_tokens),
        'targets': targets,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'padding_len': padding_length,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': new_sentiment,
        'orig_sentiment': sentiment,
    }


def preprocess(tweet, selected_text, sentiment):
    """
    Will be used in tf.data.Dataset.from_generator(...)

    """

    # The original strings have been converted to
    # byte strings, so we need to decode it
    # tweet = tweet.decode('utf-8')
    # selected_text = selected_text.decode('utf-8')
    # sentiment = sentiment.decode('utf-8')

    # Clean up the strings a bit
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    # find the intersection between text and selected text
    idx_start, idx_end = None, None
    for index in (i for i, c in enumerate(tweet) if c == selected_text[0]):
        if tweet[index:index + len(selected_text)] == selected_text:
            idx_start = index
            idx_end = index + len(selected_text)
            break

    intersection = [0] * len(tweet)
    if idx_start != None and idx_end != None:
        for char_idx in range(idx_start, idx_end):
            intersection[char_idx] = 1

    # tokenize with offsets
    enc = TOKENIZER.encode(tweet)
    input_ids_orig, offsets = enc.ids, enc.offsets

    # compute targets
    target_idx = []
    for i, (o1, o2) in enumerate(offsets):
        if sum(intersection[o1: o2]) > 0:
            target_idx.append(i)

    print(enc.tokens)
    print(input_ids_orig)
    print(target_idx)
    target_start = target_idx[0]
    target_end = target_idx[-1]

    # add and pad data (hardcoded for BERT)
    # --> [CLS] sentiment [SEP] input_ids [SEP] [PAD]
    sentiment_map = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699,
    }

    input_ids = [101] + [sentiment_map[sentiment]] + [102] + input_ids_orig + [102]
    input_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    attention_mask = [1] * (len(input_ids_orig) + 4)
    offsets = [(0, 0), (0, 0), (0, 0)] + offsets + [(0, 0)]
    target_start += 3
    target_end += 3

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        input_type_ids = input_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)
    elif padding_length < 0:
        input_ids = input_ids[:padding_length - 1] + [102]
        attention_mask = attention_mask[:padding_length - 1] + [1]
        input_type_ids = input_type_ids[:padding_length - 1] + [1]
        offsets = offsets[:padding_length - 1] + [(0, 0)]
        if target_start >= max_len:
            target_start = max_len - 1
        if target_end >= max_len:
            target_end = max_len - 1

    print(input_ids)
    print(target_start)
    print(target_end)
    print(input_type_ids)
    # return (
    #     input_ids, attention_mask, input_type_ids, offsets,
    #     target_start, target_end, tweet, selected_text, sentiment,
    # )

if __name__ == "__main__":
    # texts = ['is back home now      gonna miss every one']
    # predicted_labels = [[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # sentiment_ids = [1, 0]
    # tokenizer = create_tokenizer_from_hub_module(hp)
    # output = eval_decoded_texts(texts, predicted_labels, sentiment_ids, tokenizer)
    # print(output)


    tweet = " Sooo SAD I will miss you here in San Diego!!!"
    selected_text = "Sooo SAD"
    sentiment = "negative"
    preprocess(tweet, selected_text, sentiment)

