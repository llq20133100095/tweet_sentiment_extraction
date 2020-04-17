# coding: utf-8
import pandas as pd
import tensorflow as tf
import os
import re


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


if __name__ == "__main__":
    train = load_dataset('./input_data/train.csv')
    test = load_dataset('./input_data/test.csv')
    print(train.head(10))
    print(test.head(10))
