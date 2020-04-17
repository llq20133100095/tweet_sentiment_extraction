"""
@create time: 2020.4.17
@author: llq
@function: process the train data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from util import string_distance


def align_texts(text, selected_text):
    text_list = text.split()
    selected_text_list = selected_text.split()

    algn_selected_text = []
    for selected_text in selected_text_list:
        process_flag = True
        min_distance = float("inf")
        min_pos = 0
        for i, text in enumerate(text_list):
            _distance = string_distance(text, selected_text)
            if _distance == 0:
                process_flag = False
                algn_selected_text.append(text)
                break
            else:
                if min_distance > _distance:
                    min_distance = _distance
                    min_pos = i

        # distance no have 0
        if process_flag:
            algn_selected_text.append(text_list[min_pos])

    return " ".join(algn_selected_text)


if __name__ == "__main__":
    start_time = datetime.now()
    train_data = pd.read_csv("./input_data/train.csv")
    train_data['text'] = train_data['text'].astype(str)
    train_data['selected_text'] = train_data['selected_text'].astype(str)
    train_data['selected_text'] = train_data.apply(lambda x: align_texts(x["text"], x["selected_text"]), axis=1)
    train_data.to_csv("./input_data/train_process.csv")
    print("use time: %s" % (datetime.now() - start_time))