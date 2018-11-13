#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
cosface_new.cfg
agedb: 0.24
('max_score, min_score', 0.8335930451750014, -0.2832928619280002)
threshold:  0.2
('TP, TN, FP, FN: ', 5817.0, 5852.0, 148.0, 183.0)
('precise', 0.9751886001676446)
('precise positive', 0.9696768848384424)
('recall', 0.9695)
threshold:  0.22
('TP, TN, FP, FN: ', 5796.0, 5907.0, 93.0, 204.0)
('precise', 0.9842078451349975)
('precise positive', 0.9666175748649976)
('recall', 0.966)
threshold:  0.24
('TP, TN, FP, FN: ', 5769.0, 5951.0, 49.0, 231.0)
('precise', 0.9915778618081815)
('precise positive', 0.9626334519572953)
('recall', 0.9615)
threshold:  0.26
('TP, TN, FP, FN: ', 5733.0, 5979.0, 21.0, 267.0)
('precise', 0.9963503649635036)
('precise positive', 0.957252641690682)
('recall', 0.9555)
threshold:  0.28
('TP, TN, FP, FN: ', 5680.0, 5990.0, 10.0, 320.0)
('precise', 0.9982425307557118)
('precise positive', 0.9492868462757528)
('recall', 0.9466666666666667)



cosface_new_100.cfg
agedb: 0.2
threshold:  0.18
('TP, TN, FP, FN: ', 5844.0, 5878.0, 122.0, 156.0)
('precise', 0.9795507877975193)
('precise positive', 0.9741465031488233)
('recall', 0.974)
threshold:  0.2
('TP, TN, FP, FN: ', 5826.0, 5938.0, 62.0, 174.0)
('precise', 0.9894701086956522)
('precise positive', 0.9715314136125655)
('recall', 0.971)
threshold:  0.22
('TP, TN, FP, FN: ', 5809.0, 5964.0, 36.0, 191.0)
('precise', 0.9938408896492729)
('precise positive', 0.9689683184402924)
('recall', 0.9681666666666666)

lfw: 0.2
('max_score, min_score', 0.9706068935400003, -0.2566314181129999)
threshold:  0.18
('TP, TN, FP, FN: ', 5984.0, 5959.0, 41.0, 16.0)
('precise', 0.9931950207468879)
('precise positive', 0.9973221757322176)
('recall', 0.9973333333333333)
threshold:  0.2
('TP, TN, FP, FN: ', 5984.0, 5981.0, 19.0, 16.0)
('precise', 0.9968349158753956)
('precise positive', 0.9973319993329999)
('recall', 0.9973333333333333)
threshold:  0.22
('TP, TN, FP, FN: ', 5982.0, 5988.0, 12.0, 18.0)
('precise', 0.997997997997998)
('precise positive', 0.997002997002997)


cfp_fp: 0.18
('max_score, min_score', 0.8174587230269997, -0.22729761289799985)
threshold:  0.18
('TP, TN, FP, FN: ', 6799.0, 6944.0, 56.0, 201.0)
('precise', 0.9918307804522246)
('precise positive', 0.9718684394681596)
('recall', 0.9712857142857143)
threshold:  0.2
('TP, TN, FP, FN: ', 6748.0, 6977.0, 23.0, 252.0)
('precise', 0.9966031605375868)
('precise positive', 0.9651404066952552)
('recall', 0.964)

'''

import os
import random
import math

def get_score(a, b):
    sum = 0.0
    for i in range(0, len(a)):
       sum += a[i] * b[i]
    return sum

valid_set_path = '/var/darknet/face_data/'
#valid_set_path = '/var/darknet/insightface/src/lfw/'
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
while threshold < 0.45:
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
            print(score, threshold, all_label[index])
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
