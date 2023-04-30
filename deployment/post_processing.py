#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mmdetection 
@File    ：post_processing.py
@IDE     ：PyCharm 
@Author  ：haoran in deeptk
@Date    ：2022/3/31 2:46 PM
转换模型的时候使用skip-processing参数，在这里自己实现自己的后处理
'''


class PostProcess:
    """
    对一个转化后的模型进行后处理操作
    """
    def __init__(self,model_name,outputs):
        assert model_name in ('mobv3-2-gfl')
        if model_name == 'mobv3-2-gfl':
            pass

        # 开始解析输出结果
        self.output_parse(outputs)
    def output_parse(self, outputs):
        """
        Args:
            ouputs:onnx的输出结果
        """
        # 获取outputs的信息以判断是何种类型
        level_shape = []
        if isinstance(outputs,list):
            for i in outputs:
                level_shape.append(i)




        pass
    def get_outputs(self):
        pass

    def __repr__(self):
        return '深想后处理类 beta v1'
