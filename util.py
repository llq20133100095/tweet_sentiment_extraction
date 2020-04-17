"""
@create time: 2020.4.15
@author: llq
@function: util function
"""
import numpy as np


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
