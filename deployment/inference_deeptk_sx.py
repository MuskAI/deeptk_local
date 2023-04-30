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
import matplotlib.pyplot as plt
import random
import time
# import pdb
import MNN


# import traceback


# 深想模型推理类
class InferenceDeepTk:
    """
    InferenceDeepTk class is for model inference
    1. pytorch onnx mnn model is available in this class
    2. only support cpu inference at this time, but we will add auto-choice function in the future
    3. support both single image inference and batch image processing

    Notice:
    1. Before new InferenceDeepTk , you need setting (model_path, mean & std, input_size)
    2. Before using InferenceDeepTk().inference() , you need setting (image_path)

    """

    def __init__(self, model_path, mean_std=None, input_tensor_size=(1, 3, 448, 448), classes=None):
        assert os.path.isfile(model_path), '{}'.format(model_path)
        self.model_path = model_path

        self.score_thr = 0.1

        self.classes = classes
        # image_mean = [123.675, 116.28, 103.53]  # fixed ,coco mean is [123.675, 116.28, 103.53] ,but we use 0 instead
        # image_std = [58.395, 57.12, 57.375]  # fixed, coco std is [1, 1, 1],

        if mean_std is None:
            self.mean_std = {
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],  # [1, 1, 1][58.395, 57.12, 57.375]
            }
        else:
            assert mean_std['mean'] is not None and mean_std['std'] is not None
            self.mean_std = mean_std

        self.input_tensor_size = input_tensor_size
        self.model_type = ''

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
            warnings.warn('You need support more model at init function')

        self.session = session

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
        # image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB


        image = image.astype(np.float32)
        return im0, image

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None, isShow=False):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        if isShow:
            print('正在显示!')
            cv2.namedWindow('show')
            cv2.imshow('show', img)
            cv2.waitKey(0)

    @staticmethod
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

    @staticmethod
    def process_data(img, img_size=416):  # 图像预处理
        img, _, _, _ = InferenceDeepTk.letterbox(img, height=img_size)
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    # 计算时间函数
    @staticmethod
    def print_run_time(func):
        def wrapper(*args, **kw):
            local_time = time.time()
            func(*args, **kw)
            print('current Function [%s] run time is %.5f' % (func.__name__, time.time() - local_time))

        return wrapper

    def inference_batch(self, img_dir):
        data_pre_process_time = 0
        inference_time = 0

        img_list = os.listdir(img_dir)
        for idx, item in enumerate(img_list):
            img_list[idx] = os.path.join(img_dir, item)

        for idx, item in enumerate(img_list):
            if '.jpg' not in item:
                continue
            data_pre_process_start = time.time()
            im0, image = self.__pre_process(item)
            self.image_shape = (im0.shape[0], im0.shape[1])
            data_pre_process_time += time.time() - data_pre_process_start
            inference_start = time.time()

            if self.model_type == 'mnn':
                tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                                       image, MNN.Tensor_DimensionType_Caffe)
                self.input_tensor.copyFrom(tmp_input)
                self.interpreter.runSession(self.session)
                output = self.interpreter.getSessionOutputAll(self.session)

            elif self.model_type == 'onnx':
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: image})

                filtered_output = self.__parse_model_pred(output=output, model_type=self.model_type, bbox_type='xyxy',
                                                          nms=True)

                if filtered_output is None:
                    print('未检测出目标')
                else:
                    print(filtered_output)
                    # 开始画图
                    for points in filtered_output:
                        xyxy = (points['lx'], points['ly'], points['rx'], points['ry'])
                        label = '%s|%.2f' % (points['name'], points['score'])
                        self.plot_one_box(xyxy, img=im0, label=label, color=(15, 155, 255), line_thickness=3,
                                          isShow=True)

            inference_time += time.time() - inference_start

        # print('The AVG [%s] run time is %.5f' % ('data_pre_process', data_pre_process_time / len(self.img_list)))
        #
        # print('The AVG [%s] run time is %.5f' % ('inference/ AI识别时间', inference_time / len(self.img_list)))
    def get_output(self,output,need_type):
        """
        @param output: 经过nms 解析为统一输出格式的结果，只需要根据业务修改这个函数
        @param need_type: （sx-cheng）

        @return selected ：返回不同业务需要输出。 {'lx': 718,
                                                'ly': 1,
                                                'rx': 1358,
                                                'ry': 587,
                                                'score': 0.7429307103157043,
                                                'name': 'd'}
        """
        assert need_type in ('sx-cheng'), '业务类型不在列'
        selected = None
        if need_type == 'sx-cheng':

            for i in output:
                if i['name'] == 'c':
                    selected = i
                else:
                    print('分数最高的不是秤盘')

        return selected
    def inference(self, img=None):
        """
        单张图片推理
        @param img: 一张图片的路径,或者opencv读取后的数据类型
        @return:
        """
        assert img is not None, '输入图片无效'
        data_pre_process_time = 0
        inference_time = 0
        data_pre_process_start = time.time()
        # 进行预处理操作
        im0, image = self.__pre_process(img)
        self.image_shape = (im0.shape[0], im0.shape[1])
        data_pre_process_time += time.time() - data_pre_process_start
        inference_start = time.time()

        # 如果使用的是mnn模型
        if self.model_type == 'mnn':
            tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                                   image, MNN.Tensor_DimensionType_Caffe)
            self.input_tensor.copyFrom(tmp_input)
            self.interpreter.runSession(self.session)
            output = self.interpreter.getSessionOutputAll(self.session)

        # 如果使用的是onnx模型
        elif self.model_type == 'onnx':
            input_name = self.session.get_inputs()[0].name
            # 模型的直接输出结果
            output = self.session.run(None, {input_name: image})

            # 解析模型输出结果
            filtered_output = self.__parse_model_pred(output=output, model_type=self.model_type, bbox_type='xyxy',
                                                      nms=False)



            # if len(filtered_output) == 0:
            #     print('未检测出目标')
            # else:
            #     print(filtered_output)

        inference_time += time.time() - inference_start
        print('The AVG [%s] run time is %.5f' % ('data_pre_process', data_pre_process_time / len(self.img_list)))
        print('The AVG [%s] run time is %.5f' % ('inference/ AI识别时间', inference_time / len(self.img_list)))
        return self.get_output(filtered_output,need_type='sx-cheng')

    def __parse_model_pred(self, output=None, model_type=None, bbox_type='xyxy', nms=True):
        """
        using this method to uniform different model inference output
        retrun a list ,{'lx':item[0],'ly':item[1],'rx':item[2],'ry':item[3],'score':item[4],'name':self.classes[labels[idx]],}

        """

        # 未检出目标判断
        assert output is not None ,'没有检出目标'
        if output[0].shape == (1, 1, 5) and sum(output[0][:,0][0]) == 0:
            return None

        if model_type is 'onnx':
            pred = np.squeeze(output[0])  # [x,y,x,y,bbox_score]
            labels = np.squeeze(output[1])  # [1,2,0] 这里的数字分别对应类别中的第几类

            # pred要求 (lx,ly,rx,ry,bbox_score,cls)
            # nms
            if nms:
                nms_pred = InferenceDeepTk.non_max_suppression(
                    prediction=np.hstack(([pred, labels.reshape(labels.shape[0], 1)])),
                    conf_thres=0.2,
                    nms_thres=0.2)
            else:
                nms_pred = pred
            # filter bbox according to score
            filtered_pred = []
            if nms_pred[0] is None or len(nms_pred) == 0:
                return None

            # 过滤掉置信度低的
            for idx, item in enumerate(nms_pred):
                if item[4] > self.score_thr:  # if score > score thr
                    _ = {
                        'lx': item[0],
                        'ly': item[1],
                        'rx': item[2],
                        'ry': item[3],
                        'score': item[4],
                        'name': self.classes[labels[idx]]}

                    _ = InferenceDeepTk.__post_process(bbox=_, image_shape=self.image_shape,
                                                       input_shape=self.input_tensor_size)

                    # 处理界外的点
                    _['lx'] = self.image_shape[1] if _['lx'] > self.image_shape[1] else _['lx']
                    _['ly'] = self.image_shape[0] if _['ly'] > self.image_shape[0] else _['ly']
                    _['rx'] = self.image_shape[1] if _['rx'] > self.image_shape[1] else _['rx']
                    _['ry'] = self.image_shape[0] if _['ry'] > self.image_shape[0] else _['ry']

                    _['lx'] = 0 if _['lx'] < 0 else _['lx']
                    _['ly'] = 0 if _['ly'] < 0 else _['ly']
                    _['rx'] = 0 if _['rx'] < 0 else _['rx']
                    _['ry'] = 0 if _['ry'] < 0 else _['ry']

                    filtered_pred.append(_)

                else:
                    continue  # if nms_pred is sorted ,else continue

            if len(nms_pred) == 0:
                return None
            else:
                return filtered_pred


        elif model_type is 'mnn':
            pass
        else:
            raise TypeError()

    @staticmethod
    def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, min_wh=10):
        """
        Removes detections with lower object confidence score than 'conf_thres'
        Non-Maximum Suppression to further filter detections.
        min_wh = 2  # (pixels) minimum box width and height
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_conf, class)


        """
        # 如果只是一张图 则组成batch
        prediction = np.array([prediction])
        output = [None] * len(prediction)
        for image_i, pred in enumerate(prediction):
            # class_conf = np.max(pred[:, 4:], axis=1)
            # class_pred = np.argmax(pred[:, 4:], axis=1)
            # step 0 : 对所有框的conf进行从大到小的排序
            index = np.lexsort((pred[:, -2],))
            pred = pred[index]

            #  step 1: 去掉太小的框和置信度低的框
            i = (pred[:, 4] > conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
            pred2 = pred[i]

            # If none are remaining => process next image
            if len(pred2) == 0:
                continue

            # Select predicted classes
            # class_conf = class_conf[i]
            # class_pred = np.expand_dims(class_pred[i], 1)
            # class_conf = np.expand_dims(class_conf, 1)
            # numpy.concatenate((pred2[:, :5], class_conf, class_pred), 1)
            # Get detections sorted by decreasing confidence scores

            det_max = []
            nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)

            # 开始无类别的nms
            # Non-maximum suppression
            if nms_style == 'OR':  # default
                while pred2.shape[0]:
                    det_max.append(pred2[-1])  # save highest conf detection

                    pred2 = pred2[:-1]
                    if pred2.shape[0] == 0:  # Stop if we're at the last detection
                        break
                    iou = InferenceDeepTk.bbox_iou(det_max[-1], pred2)  # iou with other boxes
                    print(iou)
                    pred2 = pred2[iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    i = InferenceDeepTk.bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            # for c in np.unique(pred2[:, -1]):
            #     dc = pred2[pred2[:, -1] == c]  # select class c
            #     dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes
            #
            #     # Non-maximum suppression
            #     if nms_style == 'OR':  # default
            #         while dc.shape[0]:
            #             det_max.append(dc[:1])  # save highest conf detection
            #             if len(dc) == 1:  # Stop if we're at the last detection
            #                 break
            #             iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
            #             dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            #
            #     elif nms_style == 'AND':  # requires overlap, single boxes erased
            #         while len(dc) > 1:
            #             iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
            #             if iou.max() > 0.5:
            #                 det_max.append(dc[:1])
            #             dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            #
            #     elif nms_style == 'MERGE':  # weighted mixture box
            #         while len(dc):
            #             i = InferenceDeepTk.bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
            #             weights = dc[i, 4:5]
            #             dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
            #             det_max.append(dc[:1])
            #             dc = dc[i == 0]

            if len(det_max):
                # det_max = torch.cat(det_max)  # concatenate
                output = np.array([i.reshape(-1) for i in det_max])
                index = np.lexsort((output[:, -1],))
                output = output[index]
                output = np.flipud(output)

        return output

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        # box2 = box2.t()
        box2 = np.transpose(box2)

        # Get the coordinates of bounding boxes
        if x1y1x2y2:
            # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            # x, y, w, h = box1
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter_area = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0, 9999999) * \
                     (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0, 9999999)

        # Union Area
        union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        return inter_area / union_area  # iou

    @staticmethod
    def __post_process(bbox, image_shape, input_shape, bbox_type='xyxy', process_type='resize'):
        """
        对结果进行后处理

        """

        assert process_type in ('resize')  # we may add more pre process method
        assert bbox_type in ('xywh', 'xyxy')

        w_factor = image_shape[1] / input_shape[2]
        h_factor = image_shape[0] / input_shape[3]

        bbox['lx'], bbox['ly'] = int(bbox['lx'] * w_factor), int(bbox['ly'] * h_factor)
        bbox['rx'], bbox['ry'] = int(bbox['rx'] * w_factor), int(bbox['ry'] * h_factor)

        return bbox


if __name__ == '__main__':

    img_dir = {'hand': '../tmp/yolo2coco/images',
               'cargoboat': '/Users/musk/Desktop/实习生工作/dataset/cargoboat_split/train_img',
               'garbage': '/Users/musk/Desktop/实习生工作/dataset/FloW_IMG/training/images',
               'sx-hand': '/Users/musk/Movies/hand检测测试数据',
               'sx-hand2': '/Users/musk/Movies/sx-hand-video2',
               'coco-hand':'/Users/musk/Desktop/实习生工作/COCO-Hand/COCO-Hand-S/COCO-Hand-S_Images',
               'sx-client-data':'/Users/musk/Desktop/实习生工作/杂类',
               }

    model_path = {
        'hand-mobv3': './model_zoo/mobv3-hand.onnx',
        'cargoboat-ssd': './model_zoo/ssd300_cargoboat.onnx',
        'garbage-ssdlite': './model_zoo/ssdlite_mobilenetv2_garbage.onnx',
        'model-s-hand':'./model_zoo/model-s-hand.onnx',
        'pipeline-small-秤':'./model_zoo/pipeline-small-秤.onnx'
    }

    input_tensor_size = {
        'hand-mobv3': (1, 3, 320, 320),
        'cargoboat-ssd': (1, 3, 300, 300),
        'garbage-ssdlite': (1, 3, 320, 320),
        'pipeline-small':(1, 3, 320, 320),
    }

    classes = {
        'hand': ['hand'],
        'cargoboat': ['cargoboart', 'otherboat'],
        'garbage': ['bottle'],
        'sx-cheng':['d','w','c']
    }

    # step1 : 选择模型
    model_path = model_path['pipeline-small-秤']

    # step2：: 设置输入tensor 的shape
    input_tensor_size = input_tensor_size['pipeline-small']
    classes = classes['sx-cheng']

    print('Testing model :', model_path)

    ##
    eval_model = InferenceDeepTk(model_path=model_path, input_tensor_size=input_tensor_size, classes=classes)
    eval_model.inference(img='/Users/musk/PycharmProjects/yolo_v3-master/deployment/images/hand.png')
    # eval_model.inference_batch(img_dir['sx-client-data'])
