import imp
import os
import cv2
import json
import numpy as np


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


def gen_crop_imgs(label_txt_paths, crop_imgs_path, rec_gt_txt_name, need2del=[]):
    """gen_crop_imgs"""
    rec_gt_txt = os.path.join(crop_imgs_path, rec_gt_txt_name)
    crop_img_dir = os.path.join(crop_imgs_path, 'crop_img')
    os.makedirs(crop_img_dir, exist_ok=True)

    ques_img = []
    crop_img_count = 0
    dedup_cnt = 0
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
                im_abs_path = os.path.join('/tcdata/', im_rel_path)
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
                    fw.write('crop_img/' + img_crop_name + '\t')
                    fw.write(label['transcription'] + '\n')
                    crop_img_count += 1

            except Exception as e:
                ques_img.append(im_abs_path)
                print("Can not read image ", e)

    assert dedup_cnt == len(need2del)

    if ques_img:
        print("The following images can not be saved, "
              + "please check the image path and labels.\n"
              + "".join(str(i) + '\n' for i in ques_img))

    print("Cropped images %s have been saved in %s"
          % (crop_img_count, str(crop_img_dir)))


if __name__ == '__main__':

    """
    |--tcdata
    |--Xeon1OCR_round2_train1_20210816
        |--1525765576900-image.jpg
        |--***.jpg
        ...
    |--Xeon1OCR_round2_train2_20210816
        |--T1e3CEFDBdXXXXXXXX_%21%210-item_pic.jpg
        |--***.jpg
        ...
    |--Xeon1OCR_round2_train1_20210816.csv
    |--Xeon1OCR_round2_train2_20210816.csv

    """

    # tianchi dataset

    # --20211027(done)----------------------------------------------------------------------------------------------------------------------------------------
    # train
    label_txt_paths = [
        ['/ocr_data/det_data_20211027/Xeon1OCR_round2_train2_20210816.csv.train.txt', '20211027'],
        ['/ocr_data/det_data_20211027/Xeon1OCR_round2_train1_20210816.csv.train.txt', '20211027'],
    ]
    crop_imgs_path = '/ocr_data/rec_data_20211027/crop_imgs_train/'
    rec_gt_txt_name = 'rec_gt_train_20211027.txt'
    gen_crop_imgs(label_txt_paths, crop_imgs_path, rec_gt_txt_name)

    # val
    label_txt_paths = [
        ['/ocr_data/det_data_20211027/Xeon1OCR_round2_train2_20210816.csv.val.txt', '20211027'],
        ['/ocr_data/det_data_20211027/Xeon1OCR_round2_train1_20210816.csv.val.txt', '20211027'],
    ]
    crop_imgs_path = '/ocr_data/rec_data_20211027/crop_imgs_val/'
    rec_gt_txt_name = 'rec_gt_val_20211027.txt'
    gen_crop_imgs(label_txt_paths, crop_imgs_path, rec_gt_txt_name)
