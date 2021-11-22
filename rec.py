# -*- coding: utf-8 -*-
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath('/PaddleOCR'))

if True:
    import tools.infer.utility as utility
    from ppocr.utils.logging import get_logger
    logger = get_logger()
    from tools.infer.predict_system import TextSystem, sorted_boxes
    from tools.infer.predict_rec import TextRecognizer
    from tools.infer.predict_cls import TextClassifier
