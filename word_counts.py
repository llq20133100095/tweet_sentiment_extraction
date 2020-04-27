import pandas as pd
import numpy as np

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation
import string

# Import datasets
train = pd.read_csv('./input_data/train.csv')
test = pd.read_csv('./input_data/test.csv')
sample = pd.read_csv('./input_data/sample_submission.csv')

test["selected_text"] = test["text"]

# merge two data
data = pd.concat([train, test])

# The row with index 13133 has NaN text, so remove it from the dataset
# train = train.drop(13133)

# Make all the text lowercase - casing doesn't matter when
# we choose our selected text.
data['text'] = data['text'].apply(lambda x: str(x).lower())

# Make training/test split
# from sklearn.model_selection import train_test_split
#
# X_train, X_val = train_test_split(
#     train, train_size = 0.80, random_state = 0)

pos_data = data[data['sentiment'] == 'positive']
neutral_data = data[data['sentiment'] == 'neutral']
neg_data = data[data['sentiment'] == 'negative']

# Use CountVectorizer to get the word counts within each dataset

cv = CountVectorizer(max_df=0.95, min_df=2,
                     max_features=10000,
                     stop_words='english')

X_train_cv = cv.fit_transform(data['text'])

X_pos = cv.transform(pos_data['text'])
X_neutral = cv.transform(neutral_data['text'])
X_neg = cv.transform(neg_data['text'])

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

# Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that
# contain those words
pos_words = {}
neutral_words = {}
neg_words = {}

for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()

    pos_words[k] = pos / pos_data.shape[0]
    neutral_words[k] = neutral / neutral_data.shape[0]
    neg_words[k] = neg / neg_data.shape[0]

# We need to account for the fact that there will be a lot of words used in tweets of every sentiment.
# Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other
# sentiments that use that word.
neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])


def calculate_selected_text(df_row, tol=0):
    tweet = df_row['text']
    sentiment = df_row['sentiment']

    if sentiment == 'neutral':
        return tweet

    elif sentiment == 'positive':
        dict_to_use = pos_words_adj  # Calculate word weights using the pos_words dictionary
    elif sentiment == 'negative':
        dict_to_use = neg_words_adj  # Calculate word weights using the neg_words dictionary

    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j + 1] for i in range(words_len) for j in range(i, words_len)]

    score = 0
    selection_str = ''  # This will be our choice
    lst = sorted(subsets, key=len)  # Sort candidates by length

    for i in range(len(subsets)):

        new_sum = 0  # Sum for the current substring

        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            if lst[i][p].translate(str.maketrans('', '', string.punctuation)) in dict_to_use.keys():
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('', '', string.punctuation))]

        # If the sum is greater than the score, update our current selection
        if new_sum > score + tol:
            score = new_sum
            selection_str = lst[i]
            # tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if len(selection_str) == 0:
        selection_str = words

    return ' '.join(selection_str)


tol = 0.001

data['predicted_selection'] = ''

for index, row in data.iterrows():
    selected_text = calculate_selected_text(row, tol)

    data.loc[data['textID'] == row['textID'], ['predicted_selection']] = selected_text

data[["textID", "predicted_selection"]].to_csv("input_data/word_counts_prediction.csv", index=False)