#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import commands

f = open('/var/darknet/cnn/results/comp4_det_test_face.txt', 'rU')
image_name_to_gt = {}
for line in f.readlines():
    if '.jpg' not in line: continue
    line = line.strip('\n')
    gt_list = line.split(' ')
    #print line
    #print(gt_list)
    gt = []
    for item in gt_list[2:]:
        gt.append(float(item))
    if gt_list[0] not in image_name_to_gt:
        image_name_to_gt[gt_list[0]] = []
    image_name_to_gt[gt_list[0]].append(gt)

commands.getstatusoutput('rm -r detect_failed/ detect_image/  face_failed/')
commands.getstatusoutput('mkdir detect_failed/ detect_image/  face_failed/')
f = open('/var/darknet/FDDB_reorder/test.txt', 'rU')
count = 0
failed_count = 0
face_failed_count = 0
face_count = 0
for line in f.readlines():
    line = line.strip('\n')
    img = cv2.imread(line)
    if line not in image_name_to_gt:
        failed_count += 1
        #print("detect failed", line)
        failed_count += 1
        cv2.imwrite("detect_failed/" + line[37:-3] + 'jpg', img)
        #cv2.imshow('origin', img)
        #cv2.waitKey(0)
        continue
    label_path = '/var/darknet/FDDB_reorder/labels/' + line[37:-3] + 'txt'
    #print(label_path)
    ssd_label_file = open(label_path, 'rU')
    ssd_label = []
    show_image = False
    for ssd_lable_l in ssd_label_file.readlines():
        face_count += 1
        ssd_lable_l = ssd_lable_l.strip('\n')
        ssd_lable_l = ssd_lable_l.split(' ')
        for index in range(1, len(ssd_lable_l)):
            ssd_lable_l[index] = float(ssd_lable_l[index])
        x = ssd_lable_l[1] - ssd_lable_l[3] / 2
        y = ssd_lable_l[2] - ssd_lable_l[4] / 2
        ssd_lable_l = [x, y, x + ssd_lable_l[3], y + ssd_lable_l[4]]

        max_overlaps = -1
        max_overlaps_index = None
        max_overlaps_index_ = 0
        gt_all = image_name_to_gt[line]
        #print gt_all
        cv2.rectangle(img, (int(ssd_lable_l[0]  * img.shape[1]), int(ssd_lable_l[1] * img.shape[0])),
                      (int(ssd_lable_l[2] * img.shape[1]), int(ssd_lable_l[3] * img.shape[0])), (255, 0, 0), 2)
        for ssd_gt in gt_all:
            ixmin = max(ssd_gt[0], ssd_lable_l[0])
            iymin = max(ssd_gt[1], ssd_lable_l[1])
            ixmax = min(ssd_gt[2], ssd_lable_l[2])
            iymax = min(ssd_gt[3], ssd_lable_l[3])
            iw = max(ixmax - ixmin, 0.)
            ih = max(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((ssd_gt[2] - ssd_gt[0]) * (ssd_gt[3] - ssd_gt[1]) +
                   (ssd_lable_l[2] - ssd_lable_l[0]) * (ssd_lable_l[3] - ssd_lable_l[1]) - inters)
            overlaps = inters / uni
            #print overlaps
            if overlaps > max_overlaps:
                max_overlaps = overlaps
                max_overlaps_index = max_overlaps_index_
            max_overlaps_index_ += 1
        if(max_overlaps_index is not None):
            cv2.rectangle(img, (int(gt_all[max_overlaps_index][0]  * img.shape[1]), int(gt_all[max_overlaps_index][1] * img.shape[0])),
                          (int(gt_all[max_overlaps_index][2] * img.shape[1]), int(gt_all[max_overlaps_index][3] * img.shape[0])), (0, 0, 255), 4)
            #print max_overlaps
            if max_overlaps < 0.5:
                face_failed_count += 1
                cv2.imwrite("face_failed/" + line[37:-3] + 'jpg', img)        
    ssd_label_file.close()
    #cv2.imshow('origin', img)
    #cv2.waitKey(0)
    #print("detect_image/" + line[37:])
    cv2.imwrite("detect_image/" + line[37:-3] + 'jpg', img)
    count += 1
    #if count > 100: break
    #break;
f.close()
print("detect failed count: ", failed_count, 'count', count, failed_count / float(count))
print("face_failed_count: ", face_failed_count, 'face_count', face_count, face_failed_count / (float(face_count) + 0.00000000000001))
