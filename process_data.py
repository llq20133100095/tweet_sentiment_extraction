"""
@create time: 2020.4.17
@author: llq
@function: process the train data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from util import string_distance
from prepro import create_tokenizer_from_hub_module
from hparame import Hparame

hparame = Hparame()
parser = hparame.parser
hp = parser.parse_args()


def align_texts(text_orign, selected_text_orign, tokenizer):
    """
    align the text_orign and selected_text_orign
    :param text_orign:
    :param selected_text_orign:
    :return:
    """
    # strip_pun = ["'", '"']
    # for pun in strip_pun:
    #     text_orign = text_orign.strip(pun).strip()
    #     selected_text_orign = selected_text_orign.strip(pun).strip()
    #     text_orign = text_orign.strip().strip(pun)
    #     selected_text_orign = selected_text_orign.strip().strip(pun)

    # text_list = text_orign.split()
    # selected_text_list = selected_text_orign.split()

    text_list = tokenizer.tokenize(text_orign)
    selected_text_list = tokenizer.tokenize(selected_text_orign)

    # for i in range(len(text_list)):
    #     text_list[i] = text_list[i].replace("##", "")
    # for i in range(len(selected_text_list)):
    #     selected_text_list[i] = selected_text_list[i].replace("##", "")

    align_selected_text = []

    len_selected = len(selected_text_list)

    if " ".join(text_list) == " ".join(selected_text_list):
        return text_orign, selected_text_orign
    elif len_selected == 1:
        min_distance = float("inf")
        min_pos = 0
        for i, text in enumerate(text_list):
            distance = string_distance(text.replace("##", ""), selected_text_list[0].replace("##", ""))
            if min_distance > distance:
                min_distance = distance
                min_pos = i

        # encode happy => happ ##y
        start_pos = min_pos
        end_pos = min_pos
        while start_pos > 0 and "##" in text_list[start_pos]:
            start_pos -= 1

        while end_pos < len(text_list) and "##" in text_list[end_pos]:
            end_pos += 1

        align_selected_text += text_list[start_pos:end_pos]

    elif len_selected == 2:
        for i, text in enumerate(text_list):
            if text == selected_text_list[0] and i < len(text_list) - 1:
                # the first char of the second word ==  the first char of text_list[i+1]
                if list(selected_text_list[1])[0] == list(text_list[i+1])[0]:
                    end_pos = i + 1
                    while end_pos < len(text_list) and "##" in text_list[end_pos]:
                        end_pos += 1
                    align_selected_text += text_list[i:end_pos+1]
                    break
            elif text == selected_text_list[1] and i > 0:
                if list(selected_text_list[0])[-1] == list(text_list[i-1])[-1]:
                    start_pos = i-1
                    while start_pos > 0 and "##" in text_list[start_pos]:
                        start_pos -= 1

                    if start_pos < 0:
                        start_pos = 0
                    align_selected_text += text_list[start_pos:i+1]
                    break

    else:
        for i, selected_text in enumerate(selected_text_list):
            loop_continue = True
            for j, text in enumerate(text_list):
                if text_list[j:j+2] == selected_text_list[i:i+2]:
                    start_pos = j - i
                    end_pos = j - i + len_selected

                    # encode happy => happ ##y
                    while start_pos > 0 and "##" in text_list[start_pos]:
                        start_pos -= 1

                    while end_pos < len(text_list) and "##" in text_list[end_pos]:
                        end_pos += 1

                    if start_pos < 0:
                        start_pos = 0
                    align_selected_text += text_list[start_pos:end_pos]
                    loop_continue = False
                    break

            if not loop_continue:
                break

    # nothing match
    if len(align_selected_text) == 0:
        len_selected_char = len(selected_text_orign)
        for i, text_char in enumerate(text_orign):
            if text_orign[i:i+len_selected_char] == selected_text_orign:
                start_pos = i
                while start_pos >= 0:
                    if text_orign[start_pos] == " ":
                        break
                    start_pos -= 1

                end_pos = i+len_selected_char
                while end_pos < len(text_orign):
                    if text_orign[end_pos] == " ":
                        break
                    end_pos += 1
                break

            else:
                continue
        align_selected_text.append(text_orign[start_pos+1:end_pos])

    # print(text_list)
    # print(selected_text_list)
    # print(align_selected_text)
    assert len(align_selected_text) != 0

    return text_orign, " ".join(align_selected_text).replace(" ##", "")


if __name__ == "__main__":
    start_time = datetime.now()
    train_data = pd.read_csv("./input_data/train.csv")
    train_data['text'] = train_data['text'].astype(str)
    train_data['selected_text'] = train_data['selected_text'].astype(str)
    tokenizer = create_tokenizer_from_hub_module(hp)
    train_data[['text', 'selected_text']] = train_data.apply(lambda x: align_texts(x["text"], x["selected_text"],
                                                                                   tokenizer),
                                                             axis=1, result_type="expand")
    train_data.to_csv("./input_data/train_process.csv", index=False)
    print("use time: %s" % (datetime.now() - start_time))

    # tokenizer = create_tokenizer_from_hub_module(hp)
    # print(align_texts("Jamie @ Sean Cody, up for some angry ****?: Jamie @ Sean Cody, I wouldn`t piss this one off  Hey there Guys, Do.. http://tinyurl.com/ddyyd6", "amie @ Sean Cody, up for some angry ****?: Jamie @ Sean Cody, I wouldn`t piss this one off  Hey there Guys, Do.. http://tinyurl.com/ddyyd6", tokenizer))
    # print(tokenizer.tokenize("well,only one week left of my holidays,sad sad"))
