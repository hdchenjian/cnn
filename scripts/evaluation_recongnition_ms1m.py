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

valid_set_path = '/media/iim/disk/darknet/faces_emore/'
all_label = []
negtive_paire = []
positive_paire = []
f = open(valid_set_path + "labels_test.txt", 'rU')
for line in f.readlines():
    line = int(line.strip('\n'))
    if line == 1:
        positive_paire.append(line)
    else:
        negtive_paire.append(line)
    all_label.append(line)
f.close()

test_files = []
f = open(valid_set_path + "test.txt", 'rU')
for line in f.readlines():
    line = line.strip('\n')
    test_files.append(line)


features = []
#f = open("features_mxnet.txt", 'rU')
f = open("features.txt", 'rU')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split(' ')
    line = line[:-1]
    for i in range(0, len(line)):
        line[i] = float(line[i])
    features.append(line)
f.close()
print('test image num: ', len(test_files), 'features num: ', len(features),
      'test_label num:', len(all_label), 'positive_paire num: ', len(positive_paire), 'negtive_paire num: ' ,len(negtive_paire))

threshold = 0.0
print_score = True
while threshold < 0.25:
    print "threshold: ", threshold
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    max_score_tp = None
    min_score_tp = None
    max_score_tn = None
    min_score_tn = None
    max_score_fp = None
    min_score_fp = None
    max_score_fn = None
    min_score_fn = None
    max_score = None
    min_score = None
    for index in range(0, len(all_label)):
        score = get_score(features[2*index], features[2*index +1])
        if print_score and (max_score is None or score > max_score):
            max_score = score
        if print_score and (min_score is None or score < min_score):
            min_score = score
        if score >= threshold and all_label[index] == 1:
            TP += 1.0
            if print_score and (max_score_tp is None or score > max_score_tp):
                max_score_tp = score
                os.system('cp ' + test_files[2*index] + ' max_score_tp1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' max_score_tp2.jpg')
            if print_score and (min_score_tp is None or score < min_score_tp):
                min_score_tp = score
                os.system('cp ' + test_files[2*index] + ' min_score_tp1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' min_score_tp2.jpg')
        elif score < threshold and all_label[index] == 0:
            TN += 1.0
            if print_score and (max_score_tn is None or score > max_score_tn):
                max_score_tn = score
                os.system('cp ' + test_files[2*index] + ' max_score_tn1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' max_score_tn2.jpg')
            if print_score and (min_score_tn is None or score < min_score_tn):
                min_score_tn = score
                os.system('cp ' + test_files[2*index] + ' min_score_tn1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' min_score_tn2.jpg')
        elif score >= threshold and all_label[index] == 0:
            FP += 1.0
            if print_score and (max_score_fp is None or score > max_score_fp):
                max_score_fp = score
                os.system('cp ' + test_files[2*index] + ' max_score_fp1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' max_score_fp2.jpg')
            if print_score and (min_score_fp is None or score < min_score_fp):
                min_score_fp = score
                os.system('cp ' + test_files[2*index] + ' min_score_fp1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' min_score_fp2.jpg')
        elif score < threshold and all_label[index] == 1:
            FN += 1.0
            if print_score and (max_score_fn is None or score > max_score_fn):
                max_score_fn = score
                os.system('cp ' + test_files[2*index] + ' max_score_fn1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' max_score_fn2.jpg')
            if print_score and (min_score_fn is None or score < min_score_fn):
                min_score_fn = score
                os.system('cp ' + test_files[2*index] + ' min_score_fn1.jpg')
                os.system('cp ' + test_files[2*index +1] + ' min_score_fn2.jpg')
        else:
            print('error\n')
            exit()

    print('TP, TN, FP, FN: ', TP, TN, FP, FN)
    if print_score: print("max_score, min_score", max_score, min_score)
    if print_score:
        print('tp', max_score_tp, min_score_tp)
        print('tn', max_score_tn, min_score_tn)
        print('fp', max_score_fp, min_score_fp)
        print('fn', max_score_fn, min_score_fn)

    print("precise", TP / (TP + FP + 0.0000000000001))
    print("precise positive", TN / (TN + FN + 0.0000000000001))
    print("recall", TP / (TP + FN + 0.0000000000001))
    threshold += 0.02
    print_score = False
