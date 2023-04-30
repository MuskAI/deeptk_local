"""
@Created by deeptk
description:
1. using this script to evaluate the model inference speed
"""

import os
import warnings

import numpy
import numpy as np
import onnxruntime as ort
import cv2
import random
import time
# import pdb
# pdb.set_trace()
import MNN
#import traceback




################
def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh
def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


# 计算时间函数
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.5f' % (func.__name__ ,time.time() - local_time))
    return wrapper

# 深想模型速度评测类
class SpeedEvalDeepTk:
    """
    SpeedEvalDeepTk class is for model speed analysis
    you could add the
    """
    def __init__(self,model_path,img_path,input_tensor_size=(1, 3, 448, 448)):
        self.img_path = img_path
        self.model_path = model_path
        self.input_tensor_size = input_tensor_size


        if '.mnn' in model_path:
            interpreter = MNN.Interpreter("./deeptkhand.mnn")
            session = interpreter.createSession({'numThread': 4})
            input_tensor = interpreter.getSessionInput(session)
        else:
            warnings.warn('You need add more model at init function')

        self.interpreter = interpreter
        self.session = session
        self.input_tensor = input_tensor
    def __data_process(self):
        im0 = _image = cv2.imread(self.img_path)
        image = np.array(_image)
        image = process_data(image, self.input_tensor_size[-1])
        image = np.array([image])
        image = image.astype(np.float32)
        return im0,image
    @print_run_time
    def inference_total_time(self):

        data_pre_process_start = time.time()
        im0,image = self.__data_process()
        print('The [%s] run time is %.5f' % ('data_pre_process', time.time() - data_pre_process_start))

        inference_start = time.time()
        tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                               image, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        output_tensor = self.interpreter.getSessionOutputAll(self.session)['output']

        print('The [%s] run time is %.5f' % ('inference/ AI识别时间', time.time() - inference_start))








if __name__ == '__main__':
    img_path = 'test_data/1.jpg'
    model_path = '../tmp/epoch200_2.mnn'
    input_tensor_size = (1,3,448,448)
    eval_model = SpeedEvalDeepTk(model_path=model_path,img_path=img_path,input_tensor_size=input_tensor_size)
    eval_model.inference_total_time()