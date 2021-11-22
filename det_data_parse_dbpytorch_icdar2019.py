from genericpath import exists
import pandas as pd
import json
import os
import numpy as np
import cv2
from tqdm import tqdm


def det_data_parse_dbpytorch_icdar2019(json_file_path, images_dirs, split, res_dirs):

    with open(json_file_path, "r") as f:
        data = json.loads(f.read())

    part0_res_dir = res_dirs[0]
    gt_dir_res0 = os.path.join(part0_res_dir, split + "_gts")
    os.makedirs(gt_dir_res0, exist_ok=True)
    os.symlink(images_dirs[0], os.path.join(part0_res_dir, split + "_images"))
    list_writer0 = open(os.path.join(part0_res_dir, split + "_list.txt"), "w")

    part1_res_dir = res_dirs[1]
    gt_dir_res1 = os.path.join(part1_res_dir, split + "_gts")
    os.makedirs(gt_dir_res1, exist_ok=True)
    os.symlink(images_dirs[1], os.path.join(part1_res_dir, split + "_images"))
    list_writer1 = open(os.path.join(part1_res_dir, split + "_list.txt"), "w")

    count = 0
    for sample_name, sample_labels in data.items():
        sample_name += ".jpg"

        dir0_bool = os.path.exists(
            os.path.join(images_dirs[0], sample_name))
        dir1_bool = os.path.exists(
            os.path.join(images_dirs[1], sample_name))

        if dir0_bool and (not dir1_bool):
            gt_dir_res = gt_dir_res0
            list_writer = list_writer0
        elif dir1_bool and (not dir0_bool):
            gt_dir_res = gt_dir_res1
            list_writer = list_writer1
        else:
            raise BaseException

        list_writer.write(sample_name)
        list_writer.write("\n")

        save_name_txt = sample_name + ".txt"
        txt_writer = open(os.path.join(gt_dir_res, save_name_txt), "w")
        for single_label in sample_labels:
            for point in single_label["points"]:
                write_s = "{},{},".format(point[0], point[1])
                txt_writer.write(write_s)
            txt_writer.write("0\n")
        txt_writer.close()
        count += 1

    list_writer0.close()
    list_writer1.close()

    print("data parse done: %s, %s" % (split, count))
    print("data parse done.\n")


if __name__ == '__main__':

    json_file_path = "/ocr_data/ICDAR2019_LSVT/train_full_labels.json"
    images_dirs = [
        "/ocr_data/ICDAR2019_LSVT/train_full_images_0/",
        "/ocr_data/ICDAR2019_LSVT/train_full_images_1/"
    ]
    res_dirs = [
        "/ocr_data/icdar2019/part0/",
        "/ocr_data/icdar2019/part1/"
    ]

    det_data_parse_dbpytorch_icdar2019(json_file_path=json_file_path,
                                       images_dirs=images_dirs,
                                       split="train",
                                       res_dirs=res_dirs)
