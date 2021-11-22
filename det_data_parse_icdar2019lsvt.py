# coding: utf-8
import os
import pandas as pd
import json
import numpy as np


def gen_det_label(dataset_dir, json_file_name, img_dir_name, res_dir):
    """
    gen_det_label for ICDAR2019_LSVT

    """

    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, json_file_name + '.train.txt'), 'w') as train_out_file:
        trainset = json.load(
            open(os.path.join(dataset_dir, json_file_name), 'r'))

        img_count = 0
        all_line_count = 0
        valid_line_count = 0
        count_gt4_points = 0
        count_illegibility = 0
        for key, value in trainset.items():

            img_name = key + '.jpg'
            dir0 = img_dir_name + '_0'
            dir1 = img_dir_name + '_1'

            dir0_bool = os.path.exists(
                os.path.join(dataset_dir, dir0, img_name))
            dir1_bool = os.path.exists(
                os.path.join(dataset_dir, dir1, img_name))

            # paddleocr format
            label_res = []
            if dir0_bool and (not dir1_bool):
                img_dir_res = dir0
            elif dir1_bool and (not dir0_bool):
                img_dir_res = dir1
            else:
                raise BaseException
            img_path_res = '/ocr_data/ICDAR2019_LSVT/' + img_dir_res + '/' + img_name
            # print('img_path_res: ', img_path_res)

            # bbox
            for line in value:

                all_line_count += 1

                # text
                text = line['transcription']
                illegibility = line['illegibility']
                if text == "###" or illegibility:
                    count_illegibility += 1
                    continue

                # coord
                points = line['points']

                # paddleocr format
                points = list(map(lambda x: [str(x[0]), str(x[1])], points))
                # assert len(points) == 4
                if len(points) > 4:
                    count_gt4_points += 1
                    continue

                result = {"transcription": text, "points": points}
                label_res.append(result)
                valid_line_count += 1

            train_out_file.write(img_path_res + '\t' + json.dumps(
                label_res, ensure_ascii=False) + '\n')
            img_count += 1

        print('img_count: ', img_count)
        print('all_line_count: ', all_line_count)
        print('valid_line_count: ', valid_line_count)
        print('count_gt4_points: ', count_gt4_points)
        print('count_illegibility: ', count_illegibility)

    print('done.')


if __name__ == '__main__':
    # baidu: https://ai.baidu.com/broad/introduction?dataset=lsvt  for tianchi

    # semifinal(20211110)(done).
    dataset_dir = '/ocr_data/ICDAR2019_LSVT/'
    json_file_name = 'train_full_labels.json'
    img_dir_name = 'train_full_images'
    res_dir = '/ocr_data/det_data_20211110/'
    gen_det_label(dataset_dir, json_file_name, img_dir_name, res_dir)
