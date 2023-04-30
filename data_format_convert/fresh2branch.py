#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeptk_local 
@File    ：fresh2branch.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/20 13:45 
'''
import os,sys
import shutil
save_dir = '/Users/musk/Desktop/fresh_label_data/labeled_data'
def filter(img,label):
    img_list = os.listdir(img)
    label_list = os.listdir(label)

    for idx, l in enumerate(label_list):
        print(idx)
        key_words = l.split('.')[0] + '.jpg'
        if key_words in img_list:
            shutil.copy(src=os.path.join(img,key_words),dst=os.path.join(save_dir,key_words))



if __name__ == '__main__':
    img ='/Users/musk/Desktop/fresh_label_data/label_data'
    label = '/Users/musk/Desktop/fresh_label_data/label_xml_20220420'
    filter(img,label)