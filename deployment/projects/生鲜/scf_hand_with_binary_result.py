import os

import numpy
import numpy as np
import onnxruntime as ort
import cv2
import random
import time
import traceback
#### global variable area
classes = LABELMAP = ['hand']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(LABELMAP))]

INPUT_SIZE = (416,416)
IMG_PATH = 'test_data/my_hand.png'
ONNX_MODEL_PATH = 'hand-tiny_512-2021-02-19.onnx'
conf_thres, nms_thres =0.4, 0.4
IMG_DIR = '/Users/musk/Downloads/hand_img'

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
def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = numpy.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
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
    inter_area = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0,9999999) * \
                 (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0,9999999)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Filter out confidence scores below threshold
        # class_conf, class_pred = pred[:, 5:].max(1)  # max class_conf, index
        class_conf = np.max(pred[:, 5:],axis=1)
        class_pred = np.argmax( pred[:, 5:],axis=1)
        pred[:, 4] *= class_conf  # finall conf = obj_conf * class_conf

        i = (pred[:, 4] > conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
        # s2=time.time()
        pred2 = pred[i]
        # print("++++++pred2 = pred[i]",time.time()-s2, pred2)

        # If none are remaining => process next image
        if len(pred2) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        # class_pred = class_pred[i].unsqueeze(1).float()
        class_pred = np.expand_dims(class_pred[i],1)

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred2[:, :4] = xywh2xyxy(pred2[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        # pred2 = torch.cat((pred2[:, :5], class_conf.unsqueeze(1), class_pred), 1)
        class_conf = np.expand_dims(class_conf,1)
        numpy.concatenate((pred2[:, :5], class_conf, class_pred), 1)
        # Get detections sorted by decreasing confidence scores
        pred2 = pred2[(-pred2[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        # for c in pred2[:, -1].unique():
        for c in np.unique(pred2[:, -1]):
            dc = pred2[pred2[:, -1] == c]  # select class c
            dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            # det_max = torch.cat(det_max)  # concatenate
            det_max = det_max[0]
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    return output
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
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
def detect(img_path,input_size):

    im0 = _image = cv2.imread(img_path)
    image = np.array(_image)
    image = process_data(image, INPUT_SIZE[0])
    image = np.array([image])
    image = image.astype(np.float32)
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    s1 = time.time()
    pred  = session.run(None, {input_name: image})
    s2 = time.time()
    pred = pred[0]
    print('AI识别时间:',s2-s1)
    detections = non_max_suppression(pred, conf_thres)[0]  # nms

    judge_ret = inference_result2binary(detections,score_thr=0.3)

    ## print the judge result
    if judge_ret[0] == True:
        print('Hand, score:%.5f'%judge_ret[1])
        # save_name =img_path.split('/')[-1].split('.')[0]+'score_%.2f'%judge_ret[1]+'.jpg'
        # cv2.imwrite(os.path.join('./tmp/hand_in',save_name),im0)
    else:
        save_name = img_path.split('/')[-1]
        # cv2.imwrite(os.path.join('./tmp/no_hand',save_name),im0)
        print('NO Hand',img_path)


    return judge_ret

def inference_result2binary(result_after_nms,score_thr):
    """
    result_after_nms (x1, y1, x2, y2, object_conf, class_conf, class)
    return: [bool,float] True: hand in image ;False:no hand in image
    """

    judge_result = [None, None]
    if result_after_nms is None:
        return [False, 0]

    for idx, item in enumerate(result_after_nms):
        # get cls confidence
        conf_cls = item[5]
        # get bbox confidence
        conf_bbox = item[4]

        # a formula , input cls confidence and bbox confidence, getting a comprehensive score which can answer
        # yes or no problem.

        x = lambda score: np.sin((score[0]*1.2 + score[1]*0.8)/2)
        score = x([conf_cls,conf_bbox])

        if score > score_thr:
            if judge_result == [None,None]:

                judge_result[0] = True
                judge_result[1] = score
            elif score > judge_result[1]:
                judge_result[1] = score
            else:
                pass


        else:
            pass

    if judge_result == [None, None]:
        judge_result = [False, 0]


    return judge_result




if __name__ == '__main__':
    img_list = os.listdir(IMG_DIR)
    for idx, item in enumerate(img_list):
        # img_path = os.path.join(IMG_DIR,item)

        try:
            img_path = 'test_data/1.jpg'
            det_result = detect(img_path,INPUT_SIZE)
            # cv2.imwrite('./tmp/%s'%(item), det_result)
        except Exception as e:
            print(e)
            traceback.print_exc(e)
            # traceback.print_exception(e)
            continue