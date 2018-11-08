#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2

f = open('/var/darknet/celeba/Anno/list_bbox_celeba.txt', 'rU')
image_name_to_gt = {}
for line in f.readlines():
    if '.jpg' not in line: continue
    line = line.strip('\n')
    gt_list = line[10:].split(' ')
    gt = []
    for item in gt_list:
        if item != '':
            gt.append(int(item))
    image_name_to_gt[line[0:10]] = gt

f = open('/var/darknet/celeba/test.txt', 'rU')
count = 0
for line in f.readlines():
    line = line.strip('\n')
    gt = image_name_to_gt[line[26:]]
    gt = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
    label_path = '/var/darknet/celeba/labels/' + line[26:-3] + 'txt'
    ssd_label_file = open(label_path, 'rU')
    ssd_label = []
    img = cv2.imread(line)
    show_image = False
    for ssd_lable_l in ssd_label_file.readlines():
        ssd_lable_l = ssd_lable_l.strip('\n')
        ssd_lable_l = ssd_lable_l.split(' ')
        #print('tmp', ssd_lable_l)
        for index in range(1, len(ssd_lable_l)):
            #print(index, ssd_lable_l[index], type(ssd_lable_l[index]))
            ssd_lable_l[index] = float(ssd_lable_l[index])
        x = ssd_lable_l[1] - ssd_lable_l[3] / 2
        y = ssd_lable_l[2] - ssd_lable_l[4] / 2
        #print(img.shape)
        ssd_gt = [x * img.shape[1], y * img.shape[0], ssd_lable_l[3] * img.shape[1], ssd_lable_l[4] * img.shape[0]]
        ssd_gt = [ssd_gt[0], ssd_gt[1], ssd_gt[0] + ssd_gt[2], ssd_gt[1] + ssd_gt[3]]
    
        ixmin = max(ssd_gt[0], gt[0])
        iymin = max(ssd_gt[1], gt[1])
        ixmax = min(ssd_gt[2], gt[2])
        iymax = min(ssd_gt[3], gt[3])
        iw = max(ixmax - ixmin + 1., 0.)
        ih = max(iymax - iymin + 1., 0.)
        inters = iw * ih
        uni = ((ssd_gt[2] - ssd_gt[0] + 1.) * (ssd_gt[3] - ssd_gt[1] + 1.) +
               (gt[2] - gt[0] + 1.) * (gt[3] - gt[1] + 1.) - inters)
        overlaps = inters / uni
        #print(overlaps)
        if overlaps < 0.5:
            show_image = True
        cv2.rectangle(img, (int(ssd_gt[0]), int(ssd_gt[1])), (int(ssd_gt[2]), int(ssd_gt[3])), (255, 0, 0), 3)
    ssd_label_file.close()

    if show_image:
        cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 3)
        #cv2.imshow('origin', img)
        #cv2.waitKey(0)
        cv2.imwrite("false_image/" + line[26:], img)
        count += 1
    if count > 100: break
f.close()
