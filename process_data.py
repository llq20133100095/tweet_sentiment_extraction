"""
@create time: 2020.4.17
@author: llq
@function: process the train data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from util import string_distance


def align_texts(text_orign, selected_text_orign):
    """
    align the text_orign and selected_text_orign
    :param text_orign:
    :param selected_text_orign:
    :return:
    """
    strip_pun = ["'", '"']
    for pun in strip_pun:
        text_orign = text_orign.strip(pun).strip()
        selected_text_orign = selected_text_orign.strip(pun).strip()
        text_orign = text_orign.strip().strip(pun)
        selected_text_orign = selected_text_orign.strip().strip(pun)

    text_list = text_orign.split()
    selected_text_list = selected_text_orign.split()

    align_selected_text = []

    len_selected = len(selected_text_list)

    if " ".join(text_list) == " ".join(selected_text_list):
        return text_orign, selected_text_orign
    elif len_selected == 1:
        process_flag = True
        min_distance = float("inf")
        min_pos = 0
        for i, text in enumerate(text_list):
            distance = string_distance(text, selected_text_list[0])
            if distance == 0:
                align_selected_text.append(text)
                process_flag = False
                break
            else:
                if min_distance > distance:
                    min_distance = distance
                    min_pos = i

        if process_flag:
            align_selected_text.append(text_list[min_pos])

    elif len_selected == 2:
        # example: "well,only one week left of my holidays,sad sad", "sad sad"
        if selected_text_list[0] == selected_text_list[1]:
            return text_orign, selected_text_orign
        else:
            for i, selected_text in enumerate(selected_text_list):
                try:
                    selected_pos = text_list.index(selected_text)
                    start_pos = selected_pos - i
                    end_pos = selected_pos - i + len_selected
                    align_selected_text += text_list[start_pos:end_pos]
                    break
                except:
                    continue

    else:
        for i, selected_text in enumerate(selected_text_list):
            loop_continue = True
            for j, text in enumerate(text_list):
                if text_list[j:j+2] == selected_text_list[i:i+2]:
                    start_pos = j - i
                    end_pos = j - i + len_selected
                    align_selected_text += text_list[start_pos:end_pos]
                    loop_continue = False
                    break

            if not loop_continue:
                break

    if len(align_selected_text) == 0:
        for selected_text in selected_text_list:
            process_flag = True
            min_distance = float("inf")
            min_pos = 0

            for i, text in enumerate(text_list):
                _distance = string_distance(text, selected_text)
                if _distance == 0:
                    process_flag = False
                    align_selected_text.append(text)
                    break
                else:
                    if min_distance > _distance:
                        min_distance = _distance
                        min_pos = i

            # distance no have 0
            if process_flag:
                align_selected_text.append(text_list[min_pos])

    # print(text_orign)
    # print(selected_text_orign)
    # print(align_selected_text)
    assert len(align_selected_text) == len_selected

    return text_orign, " ".join(align_selected_text)


if __name__ == "__main__":
    start_time = datetime.now()
    train_data = pd.read_csv("./input_data/train.csv")
    train_data['text'] = train_data['text'].astype(str)
    train_data['selected_text'] = train_data['selected_text'].astype(str)
    train_data[['text', 'selected_text']] = train_data.apply(lambda x: align_texts(x["text"], x["selected_text"]),
                                                             axis=1, result_type="expand")
    train_data.to_csv("./input_data/train_process.csv")
    print("use time: %s" % (datetime.now() - start_time))

    # align_texts("well,only one week left of my holidays,sad sad", "sad sad")
