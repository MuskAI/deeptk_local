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
import MNN
#import traceback



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
    def __init__(self,model_path,img_path,img_dir,input_tensor_size=(1, 3, 448, 448)):
        self.img_path = img_path
        self.model_path = model_path
        self.input_tensor_size = input_tensor_size
        self.model_type = ''
        self.img_dir = img_dir
        img_list = os.listdir(img_dir)
        for idx, item in enumerate(img_list):
            img_list[idx] = os.path.join(img_dir, item)

        self.img_list = img_list
        if '.mnn' in model_path:
            interpreter = MNN.Interpreter(model_path)
            session = interpreter.createSession({'numThread': 4})
            input_tensor = interpreter.getSessionInput(session)

            self.interpreter = interpreter

            self.input_tensor = input_tensor
            self.model_type = 'mnn'
        elif '.onnx' in model_path:
            session = ort.InferenceSession(model_path)
            self.model_type = 'onnx'
        else:
            warnings.warn('You need add more model at init function')

        self.session = session
    def __data_process(self,img_path):
        im0 = _image = cv2.imread(img_path)
        image = np.array(_image)
        image = process_data(image, self.input_tensor_size[-1])
        image = np.array([image])
        image = image.astype(np.float32)
        return im0,image
    @print_run_time
    def inference_total_time(self):
        data_pre_process_time = 0
        inference_time = 0
        for idx, item in enumerate(self.img_list):
            data_pre_process_start = time.time()
            im0,image = self.__data_process(item)
            data_pre_process_time += time.time() - data_pre_process_start
            inference_start = time.time()

            if self.model_type =='mnn':
                tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                                       image, MNN.Tensor_DimensionType_Caffe)
                self.input_tensor.copyFrom(tmp_input)
                self.interpreter.runSession(self.session)

                output_tensor = self.interpreter.getSessionOutputAll(self.session)
            elif self.model_type == 'onnx':
                input_name = self.session.get_inputs()[0].name
                pred = self.session.run(None, {input_name: image})

            inference_time+=time.time() - inference_start

        print('The AVG [%s] run time is %.5f' % ('data_pre_process', data_pre_process_time/len(self.img_list)))

        print('The AVG [%s] run time is %.5f' % ('inference/ AI识别时间',inference_time/len(self.img_list)))


if __name__ == '__main__':
    img_path = 'test_data/1.jpg'
    img_dir = 'test_data/speed_test_data'

    model_path = 'mobv3-factor2-dw-hand.onnx'
    print('Testing model :',model_path)
    # model_path_onnx = 'hand-tiny_512-2021-02-19.onnx'
    input_tensor_size = (1,3,320,320)
    eval_model = SpeedEvalDeepTk(model_path=model_path,img_path=img_path,img_dir =img_dir, input_tensor_size=input_tensor_size)
    eval_model.inference_total_time()