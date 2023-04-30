import numpy as np
import numpy
import onnxruntime as ort
import cv2
import random
import time
import warnings
warnings.filterwarnings("ignore")
#### global variable area
LABELMAP = ['cargoboat','otherboat']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(LABELMAP))]

INPUT_SIZE = (300,300)
IMG_PATH = 'ship1.jpg'
ONNX_MODEL_PATH = './onnx_model/ssd300_cargoboat.onnx'
VIDEO_PATH = './test_data/szg_58139922_50001_7e8d43721db44e929911c066148ff695.f622.mp4'
VIDEO_FRAME_INTERVAL = 3
SCORE_THR = 0.51
image_mean = [123.675, 116.28, 103.53]
image_std = [1, 1, 1]

################

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

# def xywh2xyxy(x):
#     # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
#     y = numpy.zeros_like(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2
#     y[:, 1] = x[:, 1] - x[:, 3] / 2
#     y[:, 2] = x[:, 0] + x[:, 2] / 2
#     y[:, 3] = x[:, 1] + x[:, 3] / 2
#     return y
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
        class_conf = np.max(pred[:, 4:],axis=1)
        class_pred = np.argmax( pred[:, 4:],axis=1)
        # pred[:, 4] *= class_conf  # finall conf = obj_conf * class_conf

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
        # pred2[:, :4] = xywh2xyxy(pred2[:, :4])
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

def detect(img_path,input_size,score_thr,video_path):
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    video_capture = cv2.VideoCapture(video_path)
    video_writer = None
    loc_time = time.localtime()
    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    save_video_path = "./test_result/cargoboat_{}.mp4".format(str_time)
    frame_num = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)  # ==> 总帧数
    count_num = 0
    while True:
        ret, im0 = video_capture.read()
        count_num +=1

        print('\nThe video processing : (%.4f)'%(count_num/frame_num))
        if count_num % VIDEO_FRAME_INTERVAL !=0:
            continue
        if ret:
            t = time.time()
            image = cv2.resize(im0, INPUT_SIZE)
            image = (image - image_mean) / image_std
            image = np.array([image])
            image = np.transpose(image, [0, 3, 1, 2])
            image = image.astype(np.float32)
            t1 = time.time()
            print("process time:", t1 - t)
            # session.set_providers(['CUDAExecutionProvider'], [{'device_id': 1}])
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: image})
            t2 = time.time()
            print("inference time:", t2 - t1)
            width_factor = im0.shape[0] / INPUT_SIZE[0]
            length_factor = im0.shape[1] / INPUT_SIZE[1]
            det = np.squeeze(output[0])
            labels = np.squeeze(output[1])


            # nms
            nms_det = non_max_suppression(np.array([det]),conf_thres=0.5,nms_thres=0.5)[0]
            if nms_det is None or len(nms_det) == 0:
                # cv2.namedWindow('image', 0)
                # cv2.imshow("image", im0)
                # key = cv2.waitKey(1)
                # if key == 27:
                #     break
                continue

            for idx, item in enumerate(nms_det):
                # if labels[idx] == 1:
                #     pass
                # else:
                #     continue

                if item[4] > SCORE_THR:  # if score > score thr
                    score = item[4]
                    label_name = LABELMAP[labels[idx]]
                    label_conf = '%s|%.2f' % (label_name, score)
                    pt = item
                    coords = (pt[0] * length_factor, pt[1] * width_factor, pt[2] * length_factor, pt[3] * width_factor)
                    plot_one_box(coords, im0, label=label_conf, color=colors[labels[idx]])
                else:
                    continue


            t3 = time.time()
            print("get res time:", t3 - t2)




            s2 = time.time()
            print("detect time: {} \n".format(s2 - t))

            str_fps = ("{:.2f} Fps".format(1. / (s2 - t + 0.00001)))
            cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255), 4)
            cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 1)

            # cv2.namedWindow('image', 0)
            # cv2.imshow("image", im0)
            # key = cv2.waitKey(1)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps=25, frameSize=(im0.shape[1], im0.shape[0]))
            video_writer.write(im0)
            # if key == 27:
            #     break
        else:
            break

    # cv2.destroyAllWindows()
    video_writer.release()





if __name__ == '__main__':

    det_result = detect(IMG_PATH,INPUT_SIZE,SCORE_THR,VIDEO_PATH)