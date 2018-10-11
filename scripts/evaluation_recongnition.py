#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import math

'''
f = open("features.txt", 'rU')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split(' ')
    line = line[:-1]
    _sum = 0.0
    for i in range(0, len(line)):
        line[i] = float(line[i])
        _sum += line[i] * line[i]
    print(_sum)
exit()
'''

def get_score(a, b):
    sum = 0.0
    for i in range(0, len(a)):
       sum += a[i] * b[i]
    return sum

valid_set_path = '/var/darknet/face_train_data_small/'
all_label = []
f = open(valid_set_path + "labels_test.txt", 'rU')
for line in f.readlines():
   line = line.strip('\n')
   all_label.append(line)
f.close()

# print all_label

test_label = []
test_files = []
f = open(valid_set_path + "test.txt", 'rU')
for line in f.readlines():
    test_files.append(line.strip('\n'))
    found = False
    for label in all_label:
        if label in line:
            test_label.append(all_label.index(label))
            found = True
            break
    if not found:
        print("error: can not found label", line)
        exit()
#print(test_label)
f.close()

positive_paire = []
test_paire_num = 10000.0
positive_paire_num = 1400000
negtive_paire_num = 99000000
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
            if random.random() < test_paire_num / positive_paire_num:
                positive_paire.append([_index, index_])
        index += 1
    #print positive_paire, len(positive_paire)
    #exit()

negtive_paire = []
negtive_paire_dict = {}
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
            if random.random() < test_paire_num / negtive_paire_num:
                if str(index_) + '|' + str(_index) not in negtive_paire_dict:
                    negtive_paire.append([_index, index_])
                    negtive_paire_dict[str(_index) + '|' + str(index_)] = 1

print('len(negtive_paire), len(positive_paire)', len(negtive_paire), len(positive_paire))
#print(negtive_paire)
#print(positive_paire)
negtive_num = 10000
negtive_paire_sample = []
strip_num = len(negtive_paire) / negtive_num
for i in range(0, negtive_num):
    negtive_paire_sample.append(negtive_paire[i * strip_num + random.randint(0, strip_num)])
negtive_paire = negtive_paire_sample
for i in range(0, 10): print positive_paire[i]
print('\n')
for i in range(0, 10): print negtive_paire[i]

print('len(negtive_paire), len(positive_paire)', len(negtive_paire), len(positive_paire))

features = []
f = open("features.txt", 'rU')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split(' ')
    line = line[:-1]
    for i in range(0, len(line)):
        line[i] = float(line[i])
    '''
    _sum = 0.0
    for j in range(0, len(line)):
        _sum += (line[j] * line[j])
    _sum = math.sqrt(_sum)
    for j in range(0, len(line)):
        line[j] /= _sum
    '''
    features.append(line)
f.close()

threshold = 0.2
print_score = True
while threshold < 0.85:
    print "threshold: ", threshold
    right_count = 0
    max_score = None
    min_score = None
    for item in positive_paire:
        score = get_score(features[item[0]], features[item[1]])
        if print_score and (max_score is None or score > max_score):
            max_score = score
            os.system('cp ' + test_files[item[0]] + ' positive_paire_max_score1.jpg')
            os.system('cp ' + test_files[item[1]] + ' positive_paire_max_score2.jpg')
            print('positive_paire_max_score', test_files[item[0]], test_files[item[1]])
        if print_score and (min_score is None or score < min_score):
            min_score = score
            os.system('cp ' + test_files[item[0]] + ' positive_paire_min_score1.jpg')
            os.system('cp ' + test_files[item[1]] + ' positive_paire_min_score2.jpg')
            print('positive_paire_min_score', test_files[item[0]], test_files[item[1]])
        #print score
        if score >= threshold:
            right_count += 1
    if print_score: print("positve max_score, min_score", max_score, min_score)

    max_score = None
    min_score = None
    right_count_negtive = 0
    for item in negtive_paire:
        score = get_score(features[item[0]], features[item[1]])
        if print_score and (max_score is None or score > max_score):
            max_score = score
            os.system('cp ' + test_files[item[0]] + ' negtive_paire_max_score1.jpg')
            os.system('cp ' + test_files[item[1]] + ' negtive_paire_max_score2.jpg')
            print('negtive_paire_max_score', test_files[item[0]], test_files[item[1]])
        if print_score and (min_score is None or score < min_score):
            min_score = score
            os.system('cp ' + test_files[item[0]] + ' negtive_paire_min_score1.jpg')
            os.system('cp ' + test_files[item[1]] + ' negtive_paire_min_score2.jpg')
            print('negtive_paire_min_score', test_files[item[0]], test_files[item[1]])
        #print score
        if score < threshold:
            right_count_negtive += 1
    if print_score: print("negtive max_score, min_score", max_score, min_score)

    print("positive", right_count, '/', len(positive_paire), float(right_count) / len(positive_paire))
    print("negtive", right_count_negtive, '/', len(negtive_paire), float(right_count_negtive) / len(negtive_paire))
    print("total", right_count + right_count_negtive, '/',  len(negtive_paire) + len(positive_paire),
          float(right_count + right_count_negtive) / (len(negtive_paire) + len(positive_paire)))
    threshold += 0.02
    print_score = False
