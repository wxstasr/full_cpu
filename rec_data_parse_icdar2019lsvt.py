import imp
import os
import sys
import cv2
import json
import numpy as np
from collections import defaultdict


def get_rotate_crop_image(img, points):
    """get_rotate_crop_image"""
    try:
        img_crop_width = int(max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        ))
        img_crop_height = int(max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        ))
        pts_std = np.float32([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)


def gen_crop_imgs(label_txt_paths, crop_imgs_path, rec_gt_txt_name,
                  need2del=[], convert_dict=None):
    """gen_crop_imgs"""
    rec_gt_txt = os.path.join(crop_imgs_path, rec_gt_txt_name)
    crop_img_dir = os.path.join(crop_imgs_path, 'crop_img')
    os.makedirs(crop_img_dir, exist_ok=True)

    ques_img = []
    crop_img_count = 0
    dedup_cnt = 0
    char_num_dict = defaultdict(int)

    fw = open(rec_gt_txt, 'w', encoding='utf-8')
    for label_txt_path, batch in label_txt_paths:
        print('processing %s, %s...' % (label_txt_path, batch))
        fr = open(label_txt_path, 'r')

        im_rel_path_list = []
        for line in fr.readlines():
            im_rel_path, labels = line.strip().split('\t')

            # for dedup
            if im_rel_path in need2del:
                dedup_cnt += 1
                continue

            labels = json.loads(labels)
            im_rel_path_list.append(im_rel_path)

            try:
                im_abs_path = os.path.join(
                    os.path.dirname(label_txt_path), '..',
                    im_rel_path)
                img = cv2.imread(im_abs_path)

                for i, label in enumerate(labels):
                    # if label['difficult']:
                    #     continue
                    img_crop = get_rotate_crop_image(
                        img, np.array(label['points'], np.float32))
                    img_crop_name = im_rel_path.split('/')[-1] \
                        + '_crop_' + str(i) + '_' + batch + '.jpg'
                    cv2.imwrite(os.path.join(
                        crop_img_dir, img_crop_name), img_crop)
                    fw.write(
                        '../../rec_data_20211110/crop_imgs_train/crop_img/' + img_crop_name + '\t')

                    # convert_dict
                    if convert_dict:
                        transcription = ""
                        for ch in label['transcription']:
                            if ch in convert_dict.keys():
                                transcription += convert_dict[ch]
                                char_num_dict[convert_dict[ch]] += 1
                            else:
                                transcription += ch
                                char_num_dict[ch] += 1

                        fw.write(transcription + '\n')
                    else:
                        fw.write(label['transcription'] + '\n')
                    crop_img_count += 1

            except Exception as e:
                ques_img.append(im_abs_path)
                print("Can not read image ", e)

        # for search duplication lines:
        # '/Users/yiche/repos/ocr_data/data_yc/20210406/img_train/train_label.txt' L136: img_train/143_49295203.jpg
        # print('im_rel_path_list, %s' % len(im_rel_path_list))
        # print('im_rel_path_set, %s' % len(set(im_rel_path_list)))
        # for elem in set(im_rel_path_list):
        #     im_rel_path_list.remove(elem)
        # print('im_rel_path_list, %s' % im_rel_path_list)

    assert dedup_cnt == len(need2del)

    if ques_img:
        print("The following images can not be saved, "
              + "please check the image path and labels.\n"
              + "".join(str(i) + '\n' for i in ques_img))

    print("Cropped images %s have been saved in %s"
          % (crop_img_count, str(crop_img_dir)))


if __name__ == '__main__':
    # baidu: https://ai.baidu.com/broad/introduction?dataset=lsvt  for tianchi

    # --20211110(done)----------------------------------------------------------------------------------------------------------------------------------------
    # train
    convert_dict = {
        "　": " ",
        "Ｊ": "J",
        "Ｘ": "X",
        "ｙ": "y",
        "ｗ": "w",
        "ｖ": "v",
        "Ｑ": "Q",
        "ｘ": "x",
        "ｑ": "q",
        "＼": "\\",
        "＿": "_",
        "＠": "@",
    }
    label_txt_paths = [
        ['/ocr_data/det_data_20211110/train_full_labels.json.train.txt', '20211110'],
    ]
    crop_imgs_path = '/ocr_data/rec_data_20211110/crop_imgs_train/'
    rec_gt_txt_name = 'rec_gt_train_20211110.txt'
    gen_crop_imgs(label_txt_paths, crop_imgs_path,
                  rec_gt_txt_name, convert_dict=convert_dict)
