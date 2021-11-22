#!/bin/bash

set -x
set -e


# # ICDAR2019_LSVT
# echo "ICDAR2019_LSVT data parse running"
# cd /ICDAR2019_LSVT
# tar xvf train_full_images_0.tar.gz  >> /ocr_tianchi/tar.out
# tar xvf train_full_images_1.tar.gz  >> /ocr_tianchi/tar.out
# rm train_full_images_0.tar.gz train_full_images_1.tar.gz /ocr_tianchi/tar.out  -f




# # det data parse
# echo "/ocr_tianchi/det_data_parse_dbpytorch.py running"
# cd /DB
# python3.7 /ocr_tianchi/det_data_parse_dbpytorch.py
# python3.7 /ocr_tianchi/det_data_parse_dbpytorch_icdar2019.py

# # det train
# echo "/DB/train.py running"
# python3.7 /DB/train.py \
#   /ocr_tianchi/cfgs_20211110/det_db_resnet50_deform_thre.yaml \
#   --resume /models/pre-trained-model-synthtext-resnet50




# # rec data parse
# echo "/ocr_tianchi/det_data_parse.py /ocr_tianchi/rec_data_parse.py running"
# cd /PaddleOCR
# python3.7 /ocr_tianchi/det_data_parse.py
# python3.7 /ocr_tianchi/rec_data_parse.py
# python3.7 /ocr_tianchi/det_data_parse_icdar2019lsvt.py
# python3.7 /ocr_tianchi/rec_data_parse_icdar2019lsvt.py

# # rec train
# echo "tools/train.py running"
# rm -rf /root/.visualdl/conf
# unset GREP_OPTIONS
# python3.7 -m paddle.distributed.launch --nproc_per_node=32 --backend=gloo \
#   tools/train.py \
#   -c /ocr_tianchi/cfgs_20211110/rec_chinese_common_train_v2.0_20210927.yml

# # rec export
# echo "tools/export_model.py running"
# cd /PaddleOCR
# python3.7 tools/export_model.py \
#   -c /ocr_tianchi/cfgs_20211110/rec_chinese_common_train_v2.0_20210927.yml \
#   -o Global.pretrained_model=./output/rec_chinese_common_train_v2.0_20211027/best_accuracy \
#     Global.save_inference_dir=./inference_tianchi/rec_chinese_common_train_v2.0_20211027/




# # service
# echo "/ocr_tianchi/server.py running"
# cd /ocr_tianchi/
# CUDA_VISIBLE_DEVICES=0 python3.7 /ocr_tianchi/server.py \
#   --rec_model_dir="/PaddleOCR/inference_tianchi/rec_chinese_common_train_v2.0_20211027/" \
#   --cls_model_dir="/models/ch_ppocr_mobile_v2.0_cls_infer/" \
#   --rec_char_dict_path="/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt" \
#   --use_angle_cls=ture

  # --rec_model_dir="/models/ch_ppocr_server_v2.0_rec_infer/" \
