# ocr_tianchi

on cpu machine `8.130.179.137`, test full ocr.


## code

get code that of full ocr.

```
git clone --recursive https://github.com/wxstasr/full_cpu.git ~/full_cpu/
```


## pretrain models

on cpu machine `8.130.179.137` of `root` user, pretrain models in `/root/ocr_tianchi_data/pretrain/`


## data

train data `tcdata` and `ICDAR2019_LSVT` in `/root/ocr_tianchi_data/`


## cfgs

cfgs in `/root/ocr_tianchi/docker_finals/cfgs_20211110/`


## env

run the commonds below to build image `full_cpu:v20211119_3` for full ocr.

```
cd ~/full_cpu/
sudo docker build -f dockerfile_submit_20211119_3_full \
  -t full_cpu:v20211119_3 \
  ./docker_finals/
```

then run the new image to get a container `full_cpu`:

```
# data
TCDATA_DIR=/root/ocr_tianchi_data/tcdata/
DATA_DIR=/root/ocr_tianchi_data/
# det
RESNET50=/root/ocr_tianchi_data/pretrain/resnet50-19c8e357.pth
# cfgs
OCR_TIANCHI_DIR=/root/ocr_tianchi/docker_finals/
# run
sudo docker run -it --rm --network=host \
  --shm-size 92g  --name full_cpu \
  -v /etc/localtime:/etc/localtime \
  -v ${TCDATA_DIR}:/tcdata \
  -v ${DATA_DIR}:/ocr_data \
  -v ${RESNET50}:/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
  -v ${OCR_TIANCHI_DIR}:/ocr_tianchi \
  full_cpu:v20211119_3 \
  /bin/bash
```


## data parse

in container `full_cpu`, run commonds below for det data parse.

```
cd /DB
python3.7 /ocr_tianchi/det_data_parse_dbpytorch.py
python3.7 /ocr_tianchi/det_data_parse_dbpytorch_icdar2019.py
```

run commonds below for rec data parse.

```
cd /PaddleOCR
python3.7 /ocr_tianchi/det_data_parse.py
python3.7 /ocr_tianchi/rec_data_parse.py
python3.7 /ocr_tianchi/det_data_parse_icdar2019lsvt.py
python3.7 /ocr_tianchi/rec_data_parse_icdar2019lsvt.py

```


## train

in container `full_cpu`; firstly, train det model:

```
cd /DB
python3.7 /DB/train.py \
  /ocr_tianchi/cfgs_20211110/det_db_resnet50_deform_thre.yaml \
  --resume /models/pre-trained-model-synthtext-resnet50
```

when det train done, get model in `/DB/workspace/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/model/final`.

then, train rec model:

```
rm -rf /root/.visualdl/conf
unset GREP_OPTIONS
python3.7 -m paddle.distributed.launch --nproc_per_node=32 --backend=gloo \
  tools/train.py \
  -c /ocr_tianchi/cfgs_20211110/rec_chinese_common_train_v2.0_20210927.yml
```

export rec infer model:

```
python3.7 tools/export_model.py \
  -c /ocr_tianchi/cfgs_20211110/rec_chinese_common_train_v2.0_20210927.yml \
  -o Global.pretrained_model=./output/rec_chinese_common_train_v2.0_20211027/best_accuracy \
    Global.save_inference_dir=./inference_tianchi/rec_chinese_common_train_v2.0_20211027/
```

when rec train and export done, get model in `/PaddleOCR/inference_tianchi/rec_chinese_common_train_v2.0_20211027/`.


## service

run service in container `full_cpu` used commands below

```
cd /ocr_tianchi/
CUDA_VISIBLE_DEVICES=0 python3.7 /ocr_tianchi/server.py \
  --rec_model_dir="/PaddleOCR/inference_tianchi/rec_chinese_common_train_v2.0_20211027/" \
  --cls_model_dir="/models/ch_ppocr_mobile_v2.0_cls_infer/" \
  --rec_char_dict_path="/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt" \
  --use_angle_cls=ture
```

test used `client.py`:

```
python client.py
```
