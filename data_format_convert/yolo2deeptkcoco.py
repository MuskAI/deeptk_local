import traceback
from pathlib import Path
import cv2
import shutil, os, sys, json
from tqdm import tqdm


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class YOLO2DeepTkCOCO:
    def __init__(self, dir_path):
        self.src_data = Path(dir_path)
        self.src = self.src_data.parent
        self.classes = ['fire']

        # self.train_txt_path = self.src_data / 'train.txt'
        # self.val_txt_path = self.src_data / 'val.txt'

        # 构建COCO格式目录
        self.dst = Path(self.src) / f"{Path(self.src_data).name}_COCO_format"
        self.coco_train = "train2017"
        self.coco_val = "val2017"
        self.coco_annotation = "annotations"
        self.coco_train_json = self.dst / self.coco_annotation \
                               / f'instances_{self.coco_train}.json'
        self.coco_val_json = self.dst / self.coco_annotation \
                             / f'instances_{self.coco_val}.json'

        mkdir(self.dst)
        mkdir(self.dst / self.coco_train)
        mkdir(self.dst / self.coco_val)
        mkdir(self.dst / self.coco_annotation)

        # 构建json内容结构
        self.type = 'instances'
        self.categories = []
        self.annotation_id = 1

        # 读取类别数
        self._get_category()

        self.info = {
            'year': 2022,
            'version': '1.0',
            'description': 'For object detection',
            'date_created': '2022',
        }

        self.licenses = [{
            'id': 1,
            'name': 'DeepTk',
            'url': 'deeptk.com',
        }]

    @staticmethod
    def clean_listdir(listdir, needed_type):
        """
        cleaning the listdir
        needed_type is what you want to keep, others will be del from listdir
        """
        assert isinstance(listdir, list)
        assert needed_type in ('dir', 'image', 'image_and_txt', 'txt'), 'Your input type is not supported  at this time'
        no_need_type_keys = {
            'dir': ['.DS_Store', '.mat'],
            'image': ['.DS_Store', '.mat', '.txt'],
            'image_and_txt': ['.DS_Store', '.mat'],
            'txt': ['.DS_Store', '.mat', '.png', '.jpg', '.PNG', '.JPG'],

        }
        new_listdir = []
        for idx, item in enumerate(listdir):
            if True in [no_need_type_keys[needed_type][i] in item for i in range(len(no_need_type_keys[needed_type]))]:
                continue
            else:
                new_listdir.append(item)

        return new_listdir

    @staticmethod
    def read_txt(txt_path):
        with open(str(txt_path), 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = list(map(lambda x: x.rstrip('\n'), data))
        return data

    def _get_category(self):
        # src_num = len(os.listdir(self.src_data))//2
        # class_list = ['hand' for _ in range(src_num)]
        class_list = self.classes
        for i, category in enumerate(class_list, 1):
            self.categories.append({
                'supercategory': category,
                'id': i,
                'name': category,
            })

    def generate(self):
        # self.train_files = YOLO2DeepTkCOCO.read_txt(self.src_data)
        txt_list = YOLO2DeepTkCOCO.clean_listdir(os.listdir(str(self.src_data)), needed_type='txt')
        img_list = YOLO2DeepTkCOCO.clean_listdir(os.listdir(str(self.src_data)), needed_type='image')

        # if Path(self.val_txt_path).exists():
        #     self.valid_files = YOLO2DeepTkCOCO.read_txt(self.val_txt_path)
        self.train_files = [os.path.join(str(self.src_data), img_name) for img_name in img_list]
        train_dest_dir = Path(self.dst) / self.coco_train
        self.gen_dataset(self.train_files, train_dest_dir,
                         self.coco_train_json)
        # self.gen_dataset(self.train_files, train_dest_dir,
        #                  self.coco_train_json)
        #
        # val_dest_dir = Path(self.dst) / self.coco_val
        # if Path(self.val_txt_path).exists():
        #     self.gen_dataset(self.valid_files, val_dest_dir,
        #                      self.coco_val_json)

        print(f"The output directory is: {str(self.dst)}")

    def gen_dataset(self, img_paths, target_img_path, target_json):
        """
        https://cocodataset.org/#format-data

        """
        images = []
        annotations = []

        for img_id, img_path in enumerate(tqdm(img_paths), 1):
            img_path = Path(img_path)

            if not img_path.exists():
                continue
            label_path = str(img_path.parent.parent
                             / 'image' / f'{img_path.stem}.txt')

            imgsrc = cv2.imread(str(img_path))
            height, width = imgsrc.shape[:2]
            height, width = int(height),int(width)
            dest_file_name = f'{img_id:012d}.jpg'
            save_img_path = target_img_path / dest_file_name

            if Path(label_path).exists():
                new_anno = self.read_annotation(label_path, img_id,
                                                height, width)
                if len(new_anno) > 0:
                    annotations.extend(new_anno)
                else:
                    print('\nNOTICE: when you see this notice , you need run twice')
                    print('\n{} not contain bbox, it will be delete'.format(label_path))
                    os.remove(label_path)
                    img_path = '{}.jpg'.format(label_path.split('.')[0])
                    print('\n{} not contain bbox, it will be delete'.format(img_path))
                    os.remove(img_path)
                    continue

                    # raise ValueError(f'{label_path} is empty')

            else:
                raise FileExistsError(f'{label_path} not exists')

            if img_path.suffix.lower() == ".jpg":
                shutil.copyfile(img_path, save_img_path)
            else:
                cv2.imwrite(str(save_img_path), imgsrc)

            images.append({
                'date_captured': '2021',
                'file_name': dest_file_name,
                'id': img_id,
                'height': height,
                'width': width,
            })

        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        with open(target_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)

    def read_annotation(self, txtfile, img_id,
                        height, width):
        annotation = []
        allinfo = YOLO2DeepTkCOCO.read_txt(txtfile)
        for label_info in allinfo:
            # 遍历一张图中不同标注对象
            label_info = label_info.split(" ")
            if len(label_info) < 5:
                continue

            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = self._get_annotation(vertex_info,
                                                            height, width)
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': int(category_id) + 1,
                'id': self.annotation_id,
            })
            self.annotation_id += 1
        return annotation

    @staticmethod
    def check(data, check_target='coco-bbox', img_shape=None):
        """
        由于在处理coco-hand数据的时候 会出现标注点超出图片坐标的情况，换言之，会出现在归一化后超出[0,1]范围的点，
        通过check方法可以找出这些点，并作出修正

        """
        assert check_target in ('coco-bbox')
        H, W = img_shape[0], img_shape[1]

        # 如果要check的是coco格式的bbox
        if check_target == 'coco-bbox':
            x0, y0, w, h = data[0], data[1], data[2], data[3]
            check_result = w + x0 > W or h + y0 > H

        # 对check_result进行判断
        if check_result is True:
            print(data)
            print('该图片的w:{},最大W{}'.format(w + x0,W))
            print('该图片的h:{},最大H{}'.format(h + y0,H))
            print('开始纠正……')
        w = W-x0 if w + x0 > W else w
        h = H-y0 if h + y0 > H else h
        return [int(x0), int(y0), int(w), int(h) ]
    @staticmethod
    def _get_annotation(vertex_info, height, width):
        cx, cy, w, h = [float(i) for i in vertex_info]

        cx = cx * width
        cy = cy * height
        box_w = int(w * width)
        box_h = int(h * height)

        # left top
        x0 = int(max(cx - box_w / 2, 0))
        y0 = int(max(cy - box_h / 2, 0))

        # right bottomt
        x1 = int(min(x0 + box_w, width))
        y1 = int(min(y0 + box_h, height))

        segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
        bbox = [x0, y0, box_w, box_h]

        bbox = YOLO2DeepTkCOCO.check(bbox,img_shape=(height,width))

        area = box_w * box_h
        return segmentation, bbox, area


if __name__ == '__main__':
    dir_path = '/Users/musk/Desktop/实习生工作/fire/train/fire_datasets_3526_after_cleaning/image'
    converter = YOLO2DeepTkCOCO(dir_path=dir_path)
    converter.generate()
