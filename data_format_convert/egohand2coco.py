"""
@created by haoran in deeptk
In order to use http://vision.soic.indiana.edu/projects/egohands/ datasets
1. open mat format annotation file,and convert it to image mask
2. convert image mask to bbox coco format file

"""
import os,shutil
import re

import cv2
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm


class Mat2BBox():
    def __init__(self, img_dir, img_shape=(720, 1280), classes=0):
        self.img_dir = img_dir
        self.img_shape = img_shape
        self.classes = classes

    def get_bbox(self, polygons):
        """
        get bbox from an array of objects boundary points
        """
        rx, ry = polygons.max(0)
        lx, ly = polygons.min(0)
        return {
            'lx': int(lx),
            'ly': int(ly),
            'rx': int(rx),
            'ry': int(ry)
        }

    def draw_bboxes(self, img, bboxes):
        # img = np.array(img,dtype='uint8')
        H, W, _ = img.shape
        if isinstance(bboxes[0], list):
            for idx, box in enumerate(bboxes):
                bboxes[idx] = {
                    'cls': box[0],
                    'lx': box[1],
                    'ly': box[2],
                    'rx': box[3],
                    'ry': box[4]
                }
        elif isinstance(bboxes[0],dict):
            pass
        else:
            raise TypeError('Type error!')

        for bbox in bboxes:
            cv2.rectangle(img=img, pt1=(int(bbox['lx']*W),int(bbox['ly']*H)),
                          pt2=(int(bbox['rx']*W), int(bbox['ry']*H)),
                          color=(255,255,0),thickness=2)

        plt.imshow(img)
        plt.show()
        return img

    def xyxy2xywh(self, xyxy):
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

    def xywh2xyxy(self, xywh):
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]

        xyxy = {
            'lx': xywh['x'] - xywh['w']/2,
            'ly': xywh['y'] - xywh['h']/2,
            'rx': xywh['x'] + xywh['w']/2,
            'ry': xywh['y'] + xywh['h']/2
        }

        return xyxy

    def draw_points(self, points, img):
        """drawing points in img"""
        np.array(img)
        for idx, xy in enumerate(points):
            x, y = int(xy[0]), int(xy[1])
            img[y, x, :] = 255
        return cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2]])

    def draw_bbox(self, data):
        """
        I just using it in debug process
        """
        # per image in same video
        for idx, item in enumerate(data):
            # per hand in same image
            gt = np.ones((720, 1280, 3))
            for idx2, item2 in enumerate(item):
                # draw
                # per (x,y) point in hand
                if item2.size is not 0:
                    bbox = mater.get_bbox(item2)
                    bbox = mater.xyxy2xywh(bbox)
                    ax = plt.gca()
                    ax.add_patch(plt.Rectangle((bbox['x'], bbox['y']), bbox['w'], bbox['h'], color="red", fill=False,
                                               linewidth=2))

                # for idx3, item3 in enumerate(item2):
                #     # print('The pos is {}-{}-{}'.format(idx,idx2,idx3))
                #     # print(item3)
                #     pass

            plt.imshow(gt)
            plt.show()

        print('The shape of data is {}'.format(data.shape))

    def save_yolo_format(self, data, img_dir=None, img_list=None, img_shape=None):
        # per image in same video
        for idx, item in tqdm(enumerate(data)):
            bbox_in_single_image = []
            img = cv2.imread(os.path.join(img_dir, img_list[idx]))
            # per hand in same image
            for idx2, item2 in enumerate(item):
                # per (x,y) point in hand
                if item2.size is not 0:
                    bbox = self.get_bbox(item2)
                    img = self.draw_points(item2, img)
                    bbox = self.xyxy2xywh(bbox)
                    bbox = {
                                 'x':bbox['x'] / img_shape[1],
                                 'y':bbox['y'] / img.shape[0],
                                 'w':bbox['w'] / img.shape[1],
                                 'h':bbox['h'] / img.shape[0]
                    }
                    # bbox = self.xywh2xyxy(bbox)
                    # uniformize the bbox
                    bbox_in_single_image.append(bbox)

            # self.draw_bboxes(img=img,
            #                  bboxes=bbox_in_single_image)

            with open(os.path.join(img_dir,'{}.txt'.format(img_list[idx].split('.')[0])),'w') as f:
                for _ in bbox_in_single_image:
                    f.write('{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(0,_['x'],_['y'],_['w'],_['h']))
                f.close()

    def set_list_order(self, img_dir):
        img_list = []
        for _ in os.listdir(img_dir):
            if 'txt' in _:
                continue
            else:
                img_list.append(_)

        if '.DS_Store' in img_list:
            img_list.remove('.DS_Store')
        if 'polygons.mat' in img_list:
            img_list.remove('polygons.mat')
        ordered_list = sorted(img_list,
                              key=lambda name: re.search('\d+', name).group())
        return ordered_list

    def deal_batch_data(self):
        # per video in dir root
        dir_list = [os.path.join(self.img_dir, name) for name in os.listdir(self.img_dir)]
        for idx, vDir in tqdm(enumerate(dir_list)):
            # per frame in video
            frame_dir = os.path.join(self.img_dir, vDir)
            if '.DS_Store' in frame_dir:
                continue
            if 'polygons' in frame_dir:
                continue
            data = sio.loadmat(os.path.join(frame_dir, 'polygons.mat'))
            data = np.array(data['polygons'])[0]
            self.save_yolo_format(data=data,
                                  img_dir=frame_dir,
                                  img_list=self.set_list_order(frame_dir),
                                  img_shape=(720, 1280))

class Mask2TXT():
    def __init__(self):
        pass

class TXT2COCO():
    def __init__(self):
        pass

    def __get_anno(self, txt_path):
        """read txt file which is yolo format annotations file"""
        assert os.path.isfile(txt_path)
        anno_list = []
        with open(txt_path, 'r') as f:
            label_list = f.readlines()
            for label in label_list:
                label = label.strip().split()
                cls = int(label[0])
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])
                anno_list.append({
                    'cls': cls,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                })

        return anno_list

    def xywh2xyxy(self, xywh, img_shape):
        # convert x,y,w,h to x1,y1,x2,y2
        H, W, _ = img_shape
        bboxes_list = []
        for idx, item in enumerate(xywh):
            if isinstance(item, dict):
                cls, x, y, w, h = item['cls'], item['x'], item['y'], item['w'], item['h']
            else:
                raise TypeError('list type is needed!')
            bboxes_list.append({
                'cls': cls,
                'lx': (x - w/2) * W,
                'ly': (y - h/2) * H,
                'rx': (x + w/2) * W,
                'ry': (y + h/2) * H
            })

        return bboxes_list

    def xyxy2xywh(self, xyxy):
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

    def draw_bboxes(self, img, bboxes):

        for bbox in bboxes:
            cv2.rectangle(img=img, pt1=(int(bbox['lx']), int(bbox['ly'])),
                          pt2=(int(bbox['rx'] ), int(bbox['ry'] )),
                          color=(255, 255, 0), thickness=2)
        plt.imshow(img)
        plt.show()

    def check_anno(self, img_dir):
        assert os.path.isdir(img_dir)
        img_list = os.listdir(img_dir)

        for idx, item in enumerate(img_list):
            if '.png' not in item and '.jpg' not in item:
                continue
            img_path = os.path.join(img_dir, item)
            anno_path = os.path.join(img_dir, '{}.txt'.format(item.split('.')[0]))
            if os.path.isfile(anno_path):
                img = cv2.imread(img_path)
                img_shape = img.shape
                # bboxes is list
                bboxes = self.__get_anno(anno_path)
                bboxes = self.xywh2xyxy(bboxes, img_shape)

                self.draw_bboxes(img=img, bboxes=bboxes)


            else:
                print('The anno File {} is not existing'.format(item))
                continue

    @staticmethod
    def clean_listdir(listdir,needed_type):
        """
        cleaning the listdir
        needed_type is what you want to keep, others will be del from listdir
        """
        assert isinstance(listdir,list)
        assert needed_type in ('dir','image','image_and_txt'), 'Your input type is not supported  at this time'
        no_need_type_keys = {
            'dir':['.DS_Store','.mat'],
            'image':['.DS_Store','.mat','.txt'],
            'image_and_txt': ['.DS_Store', '.mat']
        }
        new_listdir = []
        for idx,item in enumerate(listdir):
            if True in [no_need_type_keys[needed_type][i] in item for i in range(len(no_need_type_keys[needed_type]))]:
                continue
            else:
                new_listdir.append(item)


        return new_listdir

    @staticmethod
    def file_style_converter(img_root,style='yolo2coco',save_root=None):
        """
        在处理手掌数据集的时候，文件结构是图片和txt的标注文件是放在同一个目录下的，但是我们需要将他们分开放
        也就是将yolo style 的文件结构转化成coco style的文件结构

        """
        assert os.path.isdir(img_root)
        assert style in ('yolo2coco'),'not support this format'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
            os.makedirs(os.path.join(save_root,'images'))
        else:
            assert os.path.isdir(save_root)

        # traverse root
        img_dir = TXT2COCO.clean_listdir(os.listdir(img_root),needed_type='dir')
        img_dir_path = [os.path.join(img_root,_) for _ in img_dir]

        # traverse dir to get image,item is image name
        for idx,item in enumerate(tqdm(img_dir_path)):
            img_list = TXT2COCO.clean_listdir(os.listdir(item),needed_type='image_and_txt')

            # move data
            for idx2, item2 in enumerate(img_list):
                shutil.copyfile(src=os.path.join(item,item2),
                                dst=os.path.join(os.path.join(save_root,'images'),
                                                 '{}_{}'.format(item.split('/')[-1],item2)))

class VOC2COCO():
    def __init__(self):
        pass

class ConvertVerify():
    def __init__(self):
        pass

class VOC2DeepTkCOCO():
    def __init__(self):
        pass


if __name__ == '__main__':
    # mater = Mat2BBox(img_dir='/Users/musk/Desktop/实习生工作/egohands_data/_LABELLED_SAMPLES')
    # mater.deal_batch_data()
    txter = TXT2COCO()
    txter.file_style_converter(img_root='/Users/musk/Desktop/实习生工作/egohands_data/_LABELLED_SAMPLES',
                               save_root='/Users/musk/PycharmProjects/yolo_v3-master/tmp/yolo2coco')
    # txter.check_anno('/Users/musk/Desktop/实习生工作/egohands_data/_LABELLED_SAMPLES/CARDS_COURTYARD_B_T/')
