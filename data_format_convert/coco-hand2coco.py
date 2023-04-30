"""
处理coco-hand数据集
由于拿到手的是这样的标注格式：000000000036.jpg,216,257,208,257,230,257,257,255,243,208,216,211,hand

Annotation format:

The annotations are provided in the format below:

[image_name, xmin, xmax, ymin, ymax, x1, y1, x2, y2, x3, y3, x4, y4]

where (x1, y1), (x2, y2), (x3, y3) and (x4, y4) are the coordinates of the quadrilateral bounding box in the anti-clockwise order.
Note that the wrist is given by the line connecting (x1, y1) and (x2, y2). We also let xmin to be the minimum of (x1, x2, x3, x4)
and xmax to be the maximum of (x1, x2, x3, x4). Similarly for ymin and ymax.

换言之：txt文件中每一行只是一个bbox，给出的是手掌四个点的坐标，我们要使用的只是xmin xmax ymin ymax


需要进行如下转换：
1. 将coco-hand txt转化为一张图一个txt标注文件，这种形式是与我们的标注软件是一致：0 0.428125 0.619444 0.147917 0.316667（cls xywh）
2. 将1的结果转化为deeptk coco 格式
"""
import cv2
import os, traceback
from os.path import join
import numpy as np
from tqdm import tqdm


class CocoHand:
    def __init__(self,):

        pass

    def read_and_distribute(self, txt_file, img_dir):
        """
        读取coco-hand的txt文件，然后对每一张图生成一个txt文件，并与图片放在一块
        """
        assert os.path.isfile(txt_file)
        assert os.path.isdir(img_dir)

        with open(txt_file, 'r') as f:
            anno_list = f.readlines()

        base_idx = 0

        for idx, item in enumerate(tqdm(anno_list)):
            if idx<base_idx:
                continue
            base_idx = 0
            splited_item = item.split(',')
            img_name = splited_item[0]
            cls_name = splited_item[-1]
            bboxes = splited_item[1:5] #  xmin, xmax, ymin, ymax
            bboxes = np.array(bboxes,dtype='float')

            # 打开图片获取shape信息
            if os.path.exists(join(img_dir,img_name)):
                img = cv2.imread(join(img_dir,img_name))
                H,W,_ = img.shape

            # 向前看一看还有没有bbox
            while True:
                base_idx +=1
                if idx + base_idx > len(anno_list)-1:
                    break

                _item = anno_list[idx+base_idx]
                _item = _item.split(',')
                _img_name = _item[0]
                _cls_name = _item[-1]
                _bboxes = _item[1:5]  # xmin, xmax, ymin, ymax
                _bboxes = np.array(_bboxes, dtype='float')
                if img_name != _img_name:
                    break
                else:
                    bboxes = np.vstack((bboxes,_bboxes))

            base_idx = idx + base_idx

            if bboxes.size == 4:
                bboxes = np.array([bboxes])

            bboxes = CocoHand.coord2xywh(bboxes,(H,W))

            with open(join(img_dir, '{}.txt'.format(img_name.split('.')[0])), 'w') as f:
                for l in bboxes:
                    line = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(0,l[0],l[1],l[2],l[3])
                    f.writelines(line)

    @staticmethod
    def coord2xywh(anno_array,img_shape):
        """
        将正常坐标系下的坐标转换成xywh，需要进行归一化
        """
        H = img_shape[0]
        W = img_shape[1]

        anno_array[:,:2] = anno_array[:,:2] / W
        anno_array[:, 2:] = anno_array[:, 2:] / H

        # 这里有个异常的情况：标注点超出图片，我们当作最大值处理，也就是说小于0的当作0，大于1的当作1


        anno_array = np.where(anno_array>1,1,anno_array)
        anno_array = np.where(anno_array<0,0,anno_array)

        xywh_array = []
        for line in anno_array:
            _xywh = CocoHand.xyxy2xywh(
                {
                    'lx':line[0],
                    'ly':line[2],
                    'rx':line[1],
                    'ry':line[3]
                }

            )
            xywh_array.append([_xywh['x'],_xywh['y'],_xywh['w'],_xywh['h']])

        anno_array = np.array(xywh_array)
        return anno_array

    @staticmethod
    def xyxy2xywh(xyxy):
        """
         Convert bounding box format from  [x1, y1, x2, y2] to [x, y, w, h]
        """
        x = xyxy['lx'] + (xyxy['rx'] - xyxy['lx']) / 2
        y = xyxy['ly'] + (xyxy['ry'] - xyxy['ly']) / 2

        w = xyxy['rx'] - xyxy['lx']
        h = xyxy['ry'] - xyxy['ly']

        return {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
        }


if __name__ == '__main__':
    anno_file = '/Users/musk/Desktop/实习生工作/COCO-Hand/COCO-Hand-Big/COCO-Hand-Big_annotations.txt'
    img_file = '/Users/musk/Desktop/实习生工作/COCO-Hand/COCO-Hand-Big/COCO-Hand-Big_Images'
    converter = CocoHand()
    converter.read_and_distribute(txt_file=anno_file,img_dir=img_file)



