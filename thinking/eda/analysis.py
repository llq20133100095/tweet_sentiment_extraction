import pandas as pd
import string
import numpy as np
import re

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def remove_punct(text):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in text if ch not in exclude)
    return s

def post_process(x):
    ss = x.strip().split()

    try:
        if len(ss) == 1:
            s = x.strip()
            a = re.findall('[^A-Za-z0-9]', s)
            b = re.sub('[^A-Za-z0-9]+', '', s)
            if a.count('*') == 2 and s.index('*') == 0:
                text = b + "*"
            elif a.count('.') == 3:
                text = b + '. ' + b + '.. ' + b + '... '
            elif a.count('!') == 4:
                text = b + '! ' + b + '!! ' + b + '!!!'
            else:
                text = s
            return text
        else:
            s = ss[-1]
            a = re.findall('[^A-Za-z0-9]', s)
            b = re.sub('[^A-Za-z0-9]+', '', s)
            if a.count('.') == 3:
                text = b + '. ' + b + '.. '
            else:
                text = s
            return " ".join(ss[:-1]) + " " + text
    except:
        return x

def find_num(text, selected_text):
    text_list = text.strip().split()

    for s in text_list:
        a = re.sub('[^0-9]+', '', s)
        if len(a) != 0 and a not in selected_text:
            selected_text += " " + s
    return selected_text

def find_word(text, selected_text):
    a = ["good", "happy", "love", "day", "thanks", "great", "fun", "nice", "hope", "thank",
     "miss", "sad", "sorry", "bad", "hate", "sucks", "sick", "like", "feel", "bored",
     "get", "go", "day", "work", "going", "quot", "lol", "got", "like", "today"]

    text_list = text.strip().split()
    for s in text_list:
        for j in a:
            if j in s and j not in selected_text:
                selected_text += " " + s
    return selected_text


valid_data = pd.read_csv("../../valid_submission.csv")

for i, d in valid_data.iterrows():
    pre_tokens = d["pre_selected_text"].split()
    # if d["sentiment"] != "neutral":
    valid_data.loc[i, "pre_selected_text"] = find_word(d["text"], d["pre_selected_text"])

valid_data["jaccard"] = valid_data.apply(lambda x: jaccard(x["selected_text"], x["pre_selected_text"]), axis=1)

print(np.mean(valid_data["jaccard"]))

valid_data.to_csv("./valid_submission_eda.csv")

print(valid_data.loc[2384])
