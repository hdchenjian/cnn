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

commands.getstatusoutput('rm -r fp_detect_failed/ fp_detect_image/  fp_face_failed/')
commands.getstatusoutput('mkdir fp_detect_failed/ fp_detect_image/  fp_face_failed/')
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
        cv2.imwrite("fp_detect_failed/" + line[37:-3] + 'jpg', img)
        #cv2.imshow('origin', img)
        #cv2.waitKey(0)
        continue
    label_path = '/var/darknet/FDDB_reorder/labels/' + line[37:-3] + 'txt'
    #print(label_path)
    ssd_label_file = open(label_path, 'rU')
    ssd_label = []
    show_image = False
    gt_label_all = []
    for ssd_lable_l in ssd_label_file.readlines():
        face_count += 1
        ssd_lable_l = ssd_lable_l.strip('\n')
        ssd_lable_l = ssd_lable_l.split(' ')
        for index in range(1, len(ssd_lable_l)):
            ssd_lable_l[index] = float(ssd_lable_l[index])
        x = ssd_lable_l[1] - ssd_lable_l[3] / 2
        y = ssd_lable_l[2] - ssd_lable_l[4] / 2
        ssd_lable_l = [x, y, x + ssd_lable_l[3], y + ssd_lable_l[4]]
        gt_label_all.append(ssd_lable_l)
        cv2.rectangle(img, (int(ssd_lable_l[0]  * img.shape[1]), int(ssd_lable_l[1] * img.shape[0])),
                      (int(ssd_lable_l[2] * img.shape[1]), int(ssd_lable_l[3] * img.shape[0])), (255, 0, 0), 2)

    detect_result_all = image_name_to_gt[line]
    for detect_result_box in detect_result_all:
        max_overlaps = -1
        max_overlaps_index = None
        max_overlaps_index_ = 0
        cv2.rectangle(img, (int(detect_result_box[0]  * img.shape[1]), int(detect_result_box[1] * img.shape[0])),
                      (int(detect_result_box[2] * img.shape[1]), int(detect_result_box[3] * img.shape[0])), (0, 0, 255), 2)
        for ssd_lable_l in gt_label_all:
            ixmin = max(detect_result_box[0], ssd_lable_l[0])
            iymin = max(detect_result_box[1], ssd_lable_l[1])
            ixmax = min(detect_result_box[2], ssd_lable_l[2])
            iymax = min(detect_result_box[3], ssd_lable_l[3])
            iw = max(ixmax - ixmin, 0.)
            ih = max(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((detect_result_box[2] - detect_result_box[0]) * (detect_result_box[3] - detect_result_box[1]) +
                   (ssd_lable_l[2] - ssd_lable_l[0]) * (ssd_lable_l[3] - ssd_lable_l[1]) - inters)
            overlaps = inters / uni
            if overlaps > max_overlaps:
                max_overlaps = overlaps
                max_overlaps_index = max_overlaps_index_
            max_overlaps_index_ += 1
        #print max_overlaps
        if max_overlaps < 0.5:
            face_failed_count += 1
            cv2.imwrite("fp_face_failed/" + line[37:-3] + 'jpg', img)        
    ssd_label_file.close()
    #cv2.imshow('origin', img)
    #cv2.waitKey(0)
    #print("detect_image/" + line[37:])
    cv2.imwrite("fp_detect_image/" + line[37:-3] + 'jpg', img)
    count += 1
    #if count > 100: break
    #break;
f.close()
print("detect failed count: ", failed_count, 'count', count, failed_count / float(count))
print("face_failed_count: ", face_failed_count, 'face_count', face_count, face_failed_count / (float(face_count) + 0.00000000000001))
