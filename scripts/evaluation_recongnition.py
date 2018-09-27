#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import math

def get_score(a, b):
    sum = 0.0
    for i in range(0, len(a)):
       sum += a[i] * b[i]
    return sum

all_label = []
f = open("/var/darknet/lfw_small/labels_test.txt", 'rU')
for line in f.readlines():
   line = line.strip('\n')
   all_label.append(line)
f.close()

# print all_label

test_label = []
f = open("/var/darknet/lfw_small/test.txt", 'rU')
for line in f.readlines():
    found = False
    for label in all_label:
        if label in line:
            test_label.append(all_label.index(label))
            found = True
            break
    if not found:
        print("error: can not found label", line)
#print(test_label)
f.close()

positive_paire = []
for _label in range(0, len(all_label)):
    positive_index = []
    index = 0
    for _test in test_label:
        if _test == _label:
            positive_index.append(index)
        index += 1

    index = 0
    #print positive_index
    for _index in positive_index:
        for index_ in positive_index[index + 1:]:
            positive_paire.append([_index, index_])
        index += 1
    #print positive_paire, len(positive_paire)
    #exit()

negtive_paire = []
for _label in range(0, len(all_label)):
    positive_index = []
    negtive_index = []
    index = 0
    for _test in test_label:
        if _test == _label:
            positive_index.append(index)
        index += 1
    index = 0
    for _test in test_label:
        if _test != _label:
            negtive_index.append(index)
        index += 1

    index = 0
    #print positive_index, negtive_index, len(positive_index), len(negtive_index)
    for _index in positive_index:
        for index_ in negtive_index:
            negtive_paire.append([_index, index_])

negtive_num = 5000
negtive_paire_sample = []
strip_num = len(negtive_paire) / negtive_num
for i in range(0, negtive_num):
    negtive_paire_sample.append(negtive_paire[i * strip_num + random.randint(0, strip_num)])
negtive_paire = negtive_paire_sample
for i in range(0, 10): print positive_paire[i]
print('\n')
for i in range(0, 10): print negtive_paire[i]

features = []
f = open("features.txt", 'rU')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split(' ')
    line = line[:-1]
    for i in range(0, len(line)):
        line[i] = float(line[i])
    #print line, len(line)
    features.append(line)
f.close()

threshold = 0.6
while threshold < 0.75:
    print "threshold: ", threshold
    right_count = 0
    max_score = None
    min_score = None
    for item in positive_paire:
        score = get_score(features[item[0]], features[item[1]])
        if max_score is None or score > max_score:
            max_score = score
        if min_score is None or score < min_score:
            min_score = score
        #print score
        if score >= threshold:
            right_count += 1
    print("positve max_score, min_score", max_score, min_score)

    max_score = None
    min_score = None
    right_count_negtive = 0
    for item in negtive_paire:
        score = get_score(features[item[0]], features[item[1]])
        if max_score is None or score > max_score:
            max_score = score
        if min_score is None or score < min_score:
            min_score = score
        #print score
        if score < threshold:
            right_count_negtive += 1
    print("negtive max_score, min_score", max_score, min_score)
    print("positive", right_count, '/', len(positive_paire), float(right_count) / len(positive_paire))
    print("negtive", right_count_negtive, '/', len(negtive_paire), float(right_count_negtive) / len(negtive_paire))
    print("total", right_count + right_count_negtive, '/',  len(negtive_paire) + len(positive_paire),
          float(right_count + right_count_negtive) / (len(negtive_paire) + len(positive_paire)))
    threshold += 0.03
