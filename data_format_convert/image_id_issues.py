#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeptk_pipelines 
@File    ：image_id_issues.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/20 22:13
Deal with image id issues ,  rename image name as number
'''
import os, sys, shutil


def rename(img_dir, label_dir, dst_dir=None):
    assert dst_dir is not None
    if not os.path.exists(dst_dir):
        os.makedirs(os.path.join(dst_dir, 'image'))
        os.makedirs(os.path.join(dst_dir, 'label'))
    else:
        pass

    img_list = os.listdir(img_dir)
    label_list = os.listdir(label_dir)
    # 从1开始
    for idx, item in enumerate(img_list):
        print(idx)
        img_path = os.path.join(img_dir, item)
        # special for fire datasets
        label_name = item.replace('.jpg','').replace('.png','')
        label_path = os.path.join(label_dir, label_name+'.txt')
        img_dst = os.path.join(dst_dir, 'image', '{}.jpg'.format(idx + 1))
        label_dst = os.path.join(dst_dir, 'label', '{}.txt'.format(idx + 1))
        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.copy(src=img_path, dst=img_dst)
            shutil.copy(src=label_path, dst=label_dst)
        else:
            continue


if __name__ == '__main__':
    img = '/Users/musk/Desktop/实习生工作/fire/train/images'
    label = '/Users/musk/Desktop/实习生工作/fire/train/labels'
    dst = '/Users/musk/Desktop/实习生工作/fire/train/fire_datasets_3526_after_cleaning'
    rename(img_dir=img, label_dir=label, dst_dir=dst)
