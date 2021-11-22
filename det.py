#!python3
import argparse
import os
import sys
import math
import time
import cv2
import numpy as np

import torch

# # v1.8.0
# import intel_pytorch_extension as ipex
# # Automatically mix precision
# ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath('/DB'))

if True:
    from experiment import Structure, Experiment
    from concern.config import Configurable, Config


class ARGS:

    def __init__(self):
        # self.exp = "/DB/experiments/seg_detector/totaltext_resnet18_deform_thre.yaml"
        # self.resume = "/models/totaltext_resnet18"  # Resume from checkpoint
        self.exp = "/DB/experiments/seg_detector/totaltext_resnet50_thre.yaml"
        # # cmds/dk_finals_submit_20211110.txt
        self.resume = "/DB/workspace/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/model/final"
        # cmds/dk_finals_submit_20211115.txt
        # self.resume = "/DB/workspace/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/model/model_epoch_4_minibatch_12790"
        # self.resume = "/test_ipex190_20211115/workspace/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/model/model_epoch_0_minibatch_0"
        # self.image_path = ""  # image path
        # self.result_dir = "./demo_results/"  # path to save results
        # self.data = ""  # The name of dataloader which will be evaluated on.
        self.image_short_side = 1280  # The threshold to replace it in the representers
        # self.thresh = 0.3  # The threshold to replace it in the representers
        self.box_thresh = 0.45  # The threshold to replace it in the representers
        self.visualize = False
        self.resize = False
        self.polygon = False
        self.eager = False

        #
        self.img_type_cls_model_path = ''


class DB:

    def __init__(self):
        self.args = ARGS()
        self.args = vars(self.args)
        self.args = {k: v for k, v in self.args.items() if v is not None}
        pwd = os.getcwd()
        os.chdir("/DB")
        conf = Config()
        self.experiment_args = conf.compile(
            conf.load(self.args['exp']))['Experiment']
        self.experiment_args.update(cmd=self.args)
        self.experiment = Configurable.construct_class_from_config(
            self.experiment_args)
        self.demo = Demo_new(
            self.experiment, self.experiment_args, cmd=self.args)
        self.model = self.demo.init_all()
        os.chdir(pwd)

    def __call__(self, img, img_type="common"):
        starttime = time.time()
        dt_boxes = self.demo.inference(img, self.model, img_type)
        elapse = time.time() - starttime
        return dt_boxes, elapse


class Demo_new:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        if path.split('.')[-1] == "pth":
            model.load_state_dict(states.state_dict(), strict=False)
        else:
            model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, img):
        img = img.astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def init_all(self):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)

        # # v1.8.0
        # model = model.to(ipex.DEVICE)
        
        model.eval()
        return model

    def inference(self, img, model, visualize=False, img_type="common"):
        batch = dict()
        batch['filename'] = ["image_path"]
        img, original_shape = self.load_image(img)
        batch['shape'] = [original_shape]
        dt_boxes = []
        with torch.no_grad():
            batch['image'] = img

            # # v1.8.0
            # batch['image'] = batch['image'].to(ipex.DEVICE)
            
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(
                batch, pred, is_output_polygon=self.args['polygon'])
            # print("output: ", output)
            # print("box_thresh: %s" % self.args['box_thresh'])

            batch_boxes, batch_scores = output
            for index in range(batch['image'].size(0)):
                boxes = batch_boxes[index]
                scores = batch_scores[index]
                for i in range(boxes.shape[0]):
                    score = scores[i]
                    if score < self.args['box_thresh']:
                        continue
                    dt_boxes.append(boxes[i])

        return np.array(dt_boxes, dtype=np.float32)
