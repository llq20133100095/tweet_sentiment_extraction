"""
@create time: 2020.4.15
@author: llq
@function: util function
"""


def jaccard(str1, str2):
    """
    calculate the distance of str1 and str2
    :param str1:
    :param str2:
    :return:
    """
    str1, str2 = str(str1), str(str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))