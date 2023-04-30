import os
import random
import argparse
import shutil
from sklearn.model_selection import train_test_split
"""
created by haoran in deeptk
time :2022-1-13
基本上所有类型的标注数据都是一个folder装有图片，其中图片名称可能含有类别和编号如：ship123.jpg
采用相同的随机种子进行处理是可以确保得到一样的结果来实现图片和标签能够匹配上，但是这样不稳定，
所以下面是对图片数据进行随机划分，而后根据划分好的图片去找标签，这样也可以对数据集进行二次检查

self.train_src_list, self.val_src_list, self.train_gt_list, self.val_gt_list = \
                train_test_split(train_val_src_list, train_val_gt_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1000)
"""
args = argparse.ArgumentParser()
args.add_argument('--trainval_percent',type=float,default=0.8,help='the percent of trainval dataset')
args.add_argument('--ImagePath',type=str,help='')
args.add_argument('--AnnoPath',type=str,help='')
args.add_argument('--AnnoType',type=str,help='voc or coco',default='voc')
args.add_argument('--save_path',type=str,help='',default=None)
args.parse_args()


# create folder
# if args.save_path == None:
#     args.save_path = (args.ImagePath).replace('\\','/').split('/')[-1][0]
#     args.save_path = os.path.join(args.save_path,'after_split_data')
#
# if os.path.exists(args.save_path):
#     pass
# else:
#     os.mkdir(args.save_path)
#     os.mkdir(os.path.join(args.save_path,'trainval'))
#     os.mkdir(os.path.join(args.save_path,'test'))


class SplitDataset():
    def __init__(self):
        self.trainval_percent = None
        self.ImagePath = None
        self.AnnoPath = None
        self.save_path = None
    def setting_config(self,trainval_percent,ImagePath,AnnoPath,save_path):
        self.trainval_percent = trainval_percent
        self.ImagePath = ImagePath
        self.AnnoPath = AnnoPath
        self.save_path = save_path

    def split(self):
        trainval_percent = self.trainval_percent
        # image_list = os.listdir(self.ImagePath)
        xmlfilepath = self.AnnoPath
        total_xml = os.listdir(xmlfilepath)
        num = int(len(total_xml)*trainval_percent)
        trainval_list = random.sample(total_xml,num)

        self.save(total_xml,trainval_list)
    def save(self,total_xml_list,trainval_list):
        """
        搬运工
        :return:
        """
        if os.path.exists(self.save_path):
            pass
        else:
            os.mkdir(self.save_path)
            os.mkdir(os.path.join(self.save_path,'train_img'))
            os.mkdir(os.path.join(self.save_path,'test_img'))
            os.mkdir(os.path.join(self.save_path,'train_xml'))
            os.mkdir(os.path.join(self.save_path,'test_xml'))
            self.save_train_img = os.path.join(self.save_path,'train_img')
            self.save_test_img = os.path.join(self.save_path,'test_img')
            self.save_train_xml = os.path.join(self.save_path,'train_xml')
            self.save_test_xml = os.path.join(self.save_path,'test_xml')



        for idx ,item in enumerate(total_xml_list):
            if item in trainval_list:
                img_name = self.__check_match(item)
                if img_name is not None:
                    shutil.copy(src=os.path.join(self.ImagePath,img_name),
                                dst=os.path.join(self.save_train_img,img_name))
                    shutil.copy(src=os.path.join(self.AnnoPath,item),
                                dst=os.path.join(self.save_train_xml))


                else:
                    pass

            else:
                img_name = self.__check_match(item)
                if img_name is not None:
                    shutil.copy(src=os.path.join(self.ImagePath, img_name),
                                dst=os.path.join(self.save_test_img, img_name))
                    shutil.copy(src=os.path.join(self.AnnoPath, item),
                                dst=os.path.join(self.save_test_xml))
                else:
                    pass

    def __check_match(self,xml_name):
        img_name = xml_name.split('.')[0] + '.jpg'
        if os.path.exists(os.path.join(self.ImagePath,img_name)):
            return img_name
        else:
            print(img_name,'not match!!')
            return None


if __name__ == '__main__':
    AnnoPath = '/Users/musk/Desktop/实习生工作/dataset/sx_mask_dataset_2/Annotations'
    Image_Path = '/Users/musk/Desktop/实习生工作/dataset/sx_mask_dataset_2/JPEGImages'
    save_path = '/Users/musk/Desktop/实习生工作/dataset/cargoboat_split'
    spliter = SplitDataset()
    spliter.setting_config(trainval_percent=0.8,ImagePath=Image_Path,AnnoPath=AnnoPath,save_path=save_path)
    spliter.split()