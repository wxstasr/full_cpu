import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def det_data_parse_dbpytorch(csv_file_path, images_dir,
                             splits, res_dir):
    """
    det_data_parse_dbpytorch

    """
    trainset = pd.read_csv(csv_file_path)
    trainset_dict = dict()

    print('processing %s' % csv_file_path)

    for index in range(len(trainset)):
        img_path = eval(trainset.loc[index, "原始数据"])["tfspath"]
        img_name = os.path.basename(img_path)
        img_labels = eval(trainset.loc[index, "融合答案"])[0]
        coords = ""
        for l in img_labels:
            for coord in l["coord"]:
                coords += "{},".format(coord)
            coords += "0\n"
        trainset_dict[img_name] = coords

    print('trainset_dict done: %s' % len(trainset))

    for split, split_txt in splits.items():
        sub_res_dir = os.path.join(res_dir, split + "_gts")
        sub_res_dir_merge = os.path.join(res_dir, "train" + "_gts")
        os.makedirs(sub_res_dir)
        os.symlink(images_dir, os.path.join(res_dir, split + "_images"))
        count = 0
        with open(split_txt, "r") as fr:
            img_names = fr.readlines()
            for img_name in img_names:
                img_name = img_name.strip()
                if split == 'train':
                    gt_file_name = os.path.join(sub_res_dir, img_name + '.txt')
                if split == 'test':
                    gt_file_name = os.path.join(
                        sub_res_dir, 'gt_' + img_name.split('.')[0] + '.txt')
                    gt_file_name = os.path.join(
                        sub_res_dir_merge, img_name + '.txt')
                with open(gt_file_name, "w") as fw:
                    fw.write(trainset_dict[img_name])
                count += 1
        print('data parse done: %s, %s' % (split, count))

    print('data parse done.\n')


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

    # Xeon1OCR_round2_train1_20210816
    train1_csv_file_path = "/tcdata/Xeon1OCR_round2_train1_20210816.csv"
    train1_images_dir = "/tcdata/Xeon1OCR_round2_train1_20210816/"
    train1_splits = {
        "train": "/ocr_tianchi/det_splits/bill_train_list.txt",
        "test": "/ocr_tianchi/det_splits/bill_val_list.txt"
    }
    train1_res_dir = "/ocr_data/det_data_dbpytorch_20211101/train1/"
    det_data_parse_dbpytorch(csv_file_path=train1_csv_file_path, images_dir=train1_images_dir,
                             splits=train1_splits, res_dir=train1_res_dir)

    # Xeon1OCR_round2_train2_20210816
    train2_csv_file_path = "/tcdata/Xeon1OCR_round2_train2_20210816.csv"
    train2_images_dir = "/tcdata/Xeon1OCR_round2_train2_20210816/"
    train2_splits = {
        "train": "/ocr_tianchi/det_splits/poster_train_list.txt",
        "test": "/ocr_tianchi/det_splits/poster_val_list.txt"
    }
    train2_res_dir = "/ocr_data/det_data_dbpytorch_20211101/train2/"
    det_data_parse_dbpytorch(csv_file_path=train2_csv_file_path, images_dir=train2_images_dir,
                             splits=train2_splits, res_dir=train2_res_dir)
