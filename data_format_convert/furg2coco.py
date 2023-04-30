#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeptk_pipelines 
@File    ：furg2coco.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/7 11:37 AM

在明火检测项目中需要使用https://github.com/steffensbola/furg-fire-dataset数据集，将此数据集转化为coco格式

'''

import os,sys
import numpy as np
import cv2
from tqdm import tqdm
import xml.dom.minidom as xmldom


class Furg:
    def __init__(self,src,gt):
        assert os.path.exists(src)

        self.src = src
        self.gt = gt
        self.uni_data = {}

    def read(self,img_dir,anno_dir):
        img_list = os.listdir(img_dir)
        anno_list = os.listdir(anno_dir)
        # 开始遍历图片
        for idx,item in enumerate(tqdm(img_list)):
            img_path = os.path.join(img_dir,item)
            gt_file_name = self.__match(item)
            gt_file_path = os.path.join(anno_dir,gt_file_name)

            # 开始解析xml文件
            self.__parse_xml(gt_file_path)


            pass


            # 开始转化为通用格式


    def __parse_xml(self,xml_file):
        """
        输入xml file，并解析它
        :param xml_file:
        :return:
        """
        # 得到文件对象
        dom_obj = xmldom.parse(xml_file)

        # 得到元素对象
        element_obj = dom_obj.documentElement

        # 获得子标签
        subElementObj = element_obj.getElementsByTagName("")

    def __match(self,src_name):
        """
        通过图片名去找到对应的gt
        :param src_name:
        :return: gt_file name
        """
        return

    def write(self):
        pass
    def show(self):

        pass

    def convert(self):
        self.read(self.src,self.gt)

        pass

    def __repr__(self):
        return '明火检测数据集格式转换'


if __name__ == '__main__':

    img_dir = '/Users/musk/Desktop/实习生工作/火焰检测项目/furg/fire'
    anno_dir = '/Users/musk/Desktop/实习生工作/火焰检测项目/Annotations'
    furg = Furg(src=img_dir, gt=anno_dir)
    furg.convert()



