# -*- coding: utf-8 -*-
import json
import base64
import requests
# import cv2


def load_data(image_path, data_set, image_name):
    # 20210817
    # # data_json示例：XXXX
    # data_json = {}
    # data_json["img"] = cv2.imread(image_path)
    # data_json["data_set"] = data_set
    # data_json["index"] = image_name
    # # requests data示例
    # data_json['img'] = data_json['img'].tolist()
    # data = json.dumps(data_json)

    # 20210823
    # data_json示例：
    data_json = {}
    with open(image_path, 'rb') as jpg_file:
        byte_content = jpg_file.read()
    base64_bytes = base64.b64encode(byte_content)
    data_json["img"] = base64_bytes.decode('utf-8')
    data_json["data_set"] = data_set
    data_json["index"] = image_name
    # requests data示例
    # data_json['img'] = data_json['img'].tolist()
    data_json['img'] = [data_json['img']]
    data = json.dumps(data_json)

    return data


def post_request(url, data):
    headers = {'content-type': 'application/json'}
    response = requests.post(
        url=url, data=data, headers=headers)
    return response


if __name__ == '__main__':

    url = "http://127.0.0.1:8080/tccapi"
    # url = "http://10.168.47.7:8080/tccapi"
    url = "http://8.130.179.137:8080/tccapi"
    image_path = "PaddleOCR/doc/ppocr_framework.png"
    # image_path = "PaddleOCR/doc/pgnet_framework.png"

    # image_path = '/Users/yiche/repos/ocr_tianchi/data_semifinal/train_semifinal_1/T1e3CEFDBdXXXXXXXX_%21%210-item_pic.jpg'
    # image_path = '/Users/yiche/repos/ocr_tianchi/data_semifinal/train_semifinal_2/1525765576900-image.jpg'
    # image_path = '/Users/yiche/Desktop/tianchi/tianchi_test_20210924/1.jpg'
    # image_path = '/Users/yiche/Desktop/tianchi/tianchi_test_20210924/2.jpg'
    # image_path = '/Users/yiche/Desktop/tianchi/tianchi_test_20210924/3.jpg'
    # image_path = '/Users/yiche/Desktop/tianchi/tianchi_test_20210924/4.jpg'

    data = load_data(image_path, "data_set", "image_name")
    response = post_request(url, data)

    try:
        result_data = response.json()
        print("result_data: %s" % result_data)
    except BaseException as e:
        print('%s' % e)
