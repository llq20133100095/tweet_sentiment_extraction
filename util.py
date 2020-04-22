"""
@create time: 2020.4.15
@author: llq
@function: util function
"""
import numpy as np
import copy


def jaccard(str1, str2):
    """
    calculate the distance of str1 and str2
    :param str1:
    :param str2:
    :return:
    """
    if type(str1) == type(b""):
        str1 = str1.decode("utf-8")
    str1, str2 = str(str1), str(str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def string_distance(str1, str2):
    """
    string_distance
    :param str1:
    :param str2:
    :return:
    """
    m = str1.__len__()
    n = str2.__len__()
    distance = np.zeros((m + 1, n + 1))

    for i in range(0, m + 1):
        distance[i, 0] = i
    for i in range(0, n + 1):
        distance[0, i] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            distance[i, j] = min(distance[i - 1, j] + 1, distance[i, j - 1] + 1,
                                 distance[i - 1, j - 1] + cost)  # 分别对应删除、插入和替换

    return distance[m, n]


def eval_decoded_texts_in_position(texts, predicted_labels_origin, sentiment_ids, tokenizer):
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