#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeptk_pipelines 
@File    ：onnx_mnn_matching.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/23 22:50

测试onnx 转 mnn的匹配情况
'''

import os
import numpy as np
import onnxruntime as ort
import cv2
import MNN.expr as F
import MNN


class Matcher:
    def __init__(self, onnx_model_path, mnn_model_path,img=None):
        """

        :param onnx_model_path:
        :param mnn_model_path:
        :param img:
        """
        assert os.path.exists(onnx_model_path), 'model not exists'
        assert os.path.exists(mnn_model_path), 'model not exists'

        self.onnx_model_path = onnx_model_path
        self.mnn_model_path = mnn_model_path
        self.mean_std = {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],  # [1, 1, 1][58.395, 57.12, 57.375]
        }
        self.input_tensor_size = (1, 3, 320, 320)
        im0, image = self.__pre_process(img)
        self.image_shape = (im0.shape[0], im0.shape[1])
        self.image = image

    def __pre_process(self, img, process_type='resize'):
        assert process_type in ('resize'),'暂时不支持{}类型的操作'.format(process_type)
        if isinstance(img,str):
            im0 = cv2.imread(img)
        else:
            # TODO 这里最好加一个判断，默认为opencv的数据类型
            im0 = img
        image = cv2.resize(im0,
                           dsize=(self.input_tensor_size[-2], self.input_tensor_size[-1]))
        image = np.array(image)
        image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
        # Normalize RGB
        image = (image - self.mean_std['mean']) / self.mean_std['std']
        image = np.array([image])
        image = np.transpose(image, [0, 3, 1, 2])
        # image = image[:, :, :, :-1].transpose(2, 0, 1)  # BGR to RGB


        image = image.astype(np.float32)
        return im0, image

    def __get_mnn_output(self,image):
        # interpreter = MNN.Interpreter(self.mnn_model_path)
        # session = interpreter.createSession({'numThread': 4})
        # input_tensor = interpreter.getSessionInput(session)

        vars = F.load_as_dict(self.mnn_model_path)
        inputVar = vars['input']
        if inputVar.data_format == F.NC4HW4:
            inputVar.reorder(F.NCHW)

        inputVar.write(image.tolist())

        keys = [1026,1029,1134,1137,810,813,918,921]
        # tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
        #                        image, MNN.Tensor_DimensionType_Caffe)
        # input_tensor.copyFrom(tmp_input)
        # interpreter.runSession(session)
        # output = interpreter.getSessionOutputAll(session)
        output = [vars[str(key)] for key in keys]
        return output

    def __get_onnx_output(self,image):
        session = ort.InferenceSession(self.onnx_model_path)
        input_name = session.get_inputs()[0].name
        # 模型的直接输出结果
        output = session.run(None, {input_name: image})
        return output


    def matching(self):
        mnn_output = self.__get_mnn_output(self.image)
        onnx_output = self.__get_onnx_output(self.image)

        #首先根据支持匹配输出
        new_onnx_output = []
        for idx,item in enumerate(mnn_output):
            item = item.read()
            _size = item.shape
            item = item.reshape(1,-1)
            size = item.shape
            for idx2,item2 in enumerate(onnx_output):
                print(onnx_output[0][0:5])
                exit(0)
                item2 = item2.reshape(1,-1)
                size2 = item2.shape
                # print('mnn size{} / onnx size{}'.format(size,size2))
                if size2 == size:
                    dis = np.sum(np.absolute(item - item2))
                    # print('第{}个特征图的Size是{},L1距离为{}'.format(idx,_size,dis))


        # 开始进行匹配检验
        # assert len(mnn_output) == len(onnx_output)
        # length = len(mnn_output)
        # for _mnn_i , _onnx_i in zip(mnn_output,onnx_output):
        #     print('debug 看看先')

    def __repr__(self):
        return 'onnx 转 mnn的匹配测试工具'

if __name__ == '__main__':
    mnn_model = '/Users/musk/Desktop/实习生工作/生鲜-第二次交付-加入目标检测/pipeline-small-fresh-nopp.mnn'
    onnx_model = '/Users/musk/Desktop/实习生工作/生鲜-第二次交付-加入目标检测/pipeline-small-秤.onnx'
    img_path = './images/fresh.jpg'
    matcher = Matcher(onnx_model_path=onnx_model,mnn_model_path=mnn_model, img=img_path)
    matcher.matching()