# coding: utf-8
import os
from re import T
import pandas as pd
import json
import numpy as np


def toPaddleStyle(direction):
    if direction == "底部朝下":
        ord = [0, 1, 2, 3]
    elif direction == "底部朝右":
        ord = [1, 2, 3, 0]
    elif direction == "底部朝上":
        ord = [2, 3, 0, 1]
    elif direction == "底部朝左":
        ord = [3, 0, 1, 2]
    return ord


def gen_det_label(csv_dir, csv_train_files, data_dir, res_dir,
                  use_random_seqs=False, use_det_splits=False):
    """
    gen_det_label
    params: 
        use_random_seqs, if true random select 'val_size' imgs from 'trainset'; 
                         if false, select first 'val_size' directly.
        use_det_splits, use det splits or not, for keeping same trainset and valset.
                        higest priority.
    """

    os.makedirs(res_dir, exist_ok=True)

    for csv_file in csv_train_files:

        csv_file, split, val_size = csv_file
        val_count = 0
        with open(os.path.join(res_dir, csv_file + '.train.txt'), 'w') as train_out_file, \
                open(os.path.join(res_dir, csv_file + '.val.txt'), 'w') as val_out_file:
            trainset = pd.read_csv(os.path.join(csv_dir, csv_file))

            if use_det_splits is False:
                if use_random_seqs:
                    np.random.seed(909)
                    random_seqs = [True] * val_size + \
                        [False] * (trainset.shape[0] - val_size)
                    random_seqs = np.array(random_seqs, dtype=np.bool)
                    np.random.shuffle(random_seqs)
                    row_idx = 0

            if use_det_splits is True:
                assert os.path.isfile(os.path.join(
                    csv_dir, val_size)), "please set det split file"
                val_set_det = []
                with open(os.path.join(csv_dir, val_size), 'r') as det_split_file:
                    for ll in det_split_file.readlines():
                        if ll.strip():
                            val_set_det.append(ll.strip())

            for row in trainset.iterrows():
                # parse
                ident = (row[1]['数据ID'])
                path = json.loads(row[1]['原始数据'])['tfspath']
                labels = json.loads(row[1]['融合答案'])

                # labels
                label = labels[0]  # label dict
                img_direction = labels[1]['option']  # option dict

                # paddleocr format
                label_res = []
                img_name = path.split('/')[-1]
                img_path = data_dir + split + '/' + img_name
                # order = toPaddleStyle(img_direction)
                img_order = toPaddleStyle(img_direction)

                # bbox
                for l in label:
                    # text
                    text_dict = json.loads(l['text'])
                    text = text_dict['text']
                    text_direction = text_dict.get('direction', 'no')
                    # coord
                    coord = l['coord']

                    # paddleocr format
                    points = [[coord[ii], coord[ii+1]]
                              for ii in range(0, len(coord), 2)]
                    assert len(points) == 4
                    # result = {"transcription": text, "points": points}

                    txt_order = None
                    if text_direction != 'no':
                        txt_order = toPaddleStyle(text_direction)
                    order = txt_order if txt_order else img_order
                    result = {"transcription": text,
                              "points": [points[ii] for ii in order]}

                    label_res.append(result)

                if use_det_splits is False:
                    if use_random_seqs:
                        if bool(random_seqs[row_idx]) is True:
                            val_out_file.write(img_path + '\t' + json.dumps(
                                label_res, ensure_ascii=False) + '\n')
                        else:
                            train_out_file.write(img_path + '\t' + json.dumps(
                                label_res, ensure_ascii=False) + '\n')
                        row_idx += 1
                    else:
                        if val_count < val_size:
                            val_out_file.write(img_path + '\t' + json.dumps(
                                label_res, ensure_ascii=False) + '\n')
                            val_count += 1
                        else:
                            train_out_file.write(img_path + '\t' + json.dumps(
                                label_res, ensure_ascii=False) + '\n')

                if use_det_splits is True:
                    if img_name in val_set_det:
                        val_out_file.write(img_path + '\t' + json.dumps(
                            label_res, ensure_ascii=False) + '\n')
                    else:
                        train_out_file.write(img_path + '\t' + json.dumps(
                            label_res, ensure_ascii=False) + '\n')


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

    # semifinal(20211027)(done). use 'toPaddleStyle' related codes. use_random_seqs=True, use_det_splits=True
    csv_dir = '/tcdata/'
    csv_train_files = [
        ('Xeon1OCR_round2_train2_20210816.csv', 'Xeon1OCR_round2_train2_20210816',
         "/ocr_tianchi/det_splits/poster_val_list.txt"),  # poster
        ('Xeon1OCR_round2_train1_20210816.csv', 'Xeon1OCR_round2_train1_20210816',
         "/ocr_tianchi/det_splits/bill_val_list.txt"),  # receipt
    ]
    data_dir = ''
    res_dir = '/ocr_data/det_data_20211027/'
    gen_det_label(csv_dir, csv_train_files, data_dir,
                  res_dir, use_det_splits=True)

    pass
