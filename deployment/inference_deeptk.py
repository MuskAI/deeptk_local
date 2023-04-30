"""
@Created by deeptk
description:
1. using this script to evaluate the model inference speed
"""

import os
import warnings

import numpy
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
import random
import time
# import pdb
import MNN


# import traceback

class AnchorGenerator:
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_priors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_priors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (:obj:`torch.dtype`): Dtype of priors.
                Default: torch.float32.
            device (str): The device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse anchors according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """

        height, width = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs //
             num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width //
             num_base_anchors) % height * self.strides[level_idx][1]
        priors = torch.stack([x, y, x, y], 1).to(dtype).to(device) + \
            self.base_anchors[level_idx][base_anchor_id, :].to(device)

        return priors

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        warnings.warn('``grid_anchors`` would be deprecated soon. '
                      'Please use ``grid_priors`` ')

        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        warnings.warn(
            '``single_level_grid_anchors`` would be deprecated soon. '
            'Please use ``single_level_grid_priors`` ')

        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str

# 深想模型推理类
class InferenceDeepTk:
    """
    InferenceDeepTk class is for model inference
    1. pytorch onnx mnn model is available in this class
    2. only support cpu inference at this time, but we will add auto-choice function in the future
    3. support both single image inference and batch image processing

    Notice:
    1. Before new InferenceDeepTk , you need setting (model_path, mean & std, input_size)
    2. Before using InferenceDeepTk().inference() , you need setting (image_path)

    """

    def __init__(self, model_path, mean_std=None, input_tensor_size=(1, 3, 448, 448), classes=None):
        assert os.path.isfile(model_path), '{}'.format(model_path)
        self.model_path = model_path

        self.score_thr = 0.1

        self.classes = classes
        # image_mean = [123.675, 116.28, 103.53]  # fixed ,coco mean is [123.675, 116.28, 103.53] ,but we use 0 instead
        # image_std = [58.395, 57.12, 57.375]  # fixed, coco std is [1, 1, 1],

        if mean_std is None:
            self.mean_std = {
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],  # [1, 1, 1][58.395, 57.12, 57.375]
            }
        else:
            assert mean_std['mean'] is not None and mean_std['std'] is not None
            self.mean_std = mean_std

        self.input_tensor_size = input_tensor_size
        self.model_type = ''

        if '.mnn' in model_path:
            interpreter = MNN.Interpreter(model_path)
            session = interpreter.createSession({'numThread': 4})
            input_tensor = interpreter.getSessionInput(session)

            self.interpreter = interpreter

            self.input_tensor = input_tensor
            self.model_type = 'mnn'
        elif '.onnx' in model_path:
            session = ort.InferenceSession(model_path)
            self.model_type = 'onnx'
        else:
            warnings.warn('You need support more model at init function')

        self.session = session

    def __pre_process(self, img_path, process_type='resize'):
        assert process_type in ('resize')
        im0 = cv2.imread(img_path)
        image = cv2.resize(im0,
                           dsize=(self.input_tensor_size[-2], self.input_tensor_size[-1]))
        image = np.array(image)
        image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
        # Normalize RGB
        image = (image - self.mean_std['mean']) / self.mean_std['std']
        image = np.array([image])
        image = np.transpose(image, [0, 3, 1, 2])
        # image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB

        # norm
        # img[0,:, :] = (img[0,:, :] - self.mean_std['mean'][0]) / self.mean_std['std'][0]
        # img[1,:, :] = (img[1,:, :] - self.mean_std['mean'][1]) / self.mean_std['std'][1]
        # img[2,:, :] = (img[2,:, :] - self.mean_std['mean'][2]) / self.mean_std['std'][2]

        image = image.astype(np.float32)
        return im0, image

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None, isShow=False):
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

        if isShow:
            print('正在显示!')
            cv2.namedWindow('show')
            cv2.imshow('show', img)
            cv2.waitKey(0)

    @staticmethod
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

    @staticmethod
    def process_data(img, img_size=416):  # 图像预处理
        img, _, _, _ = InferenceDeepTk.letterbox(img, height=img_size)
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    # 计算时间函数
    @staticmethod
    def print_run_time(func):
        def wrapper(*args, **kw):
            local_time = time.time()
            func(*args, **kw)
            print('current Function [%s] run time is %.5f' % (func.__name__, time.time() - local_time))

        return wrapper

    def inference_batch(self, img_dir):
        data_pre_process_time = 0
        inference_time = 0

        img_list = os.listdir(img_dir)
        for idx, item in enumerate(img_list):
            img_list[idx] = os.path.join(img_dir, item)

        for idx, item in enumerate(img_list):
            if '.jpg' not in item:
                continue
            data_pre_process_start = time.time()
            im0, image = self.__pre_process(item)
            self.image_shape = (im0.shape[0], im0.shape[1])
            data_pre_process_time += time.time() - data_pre_process_start
            inference_start = time.time()

            if self.model_type == 'mnn':
                tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                                       image, MNN.Tensor_DimensionType_Caffe)
                self.input_tensor.copyFrom(tmp_input)
                self.interpreter.runSession(self.session)
                output = self.interpreter.getSessionOutputAll(self.session)

            elif self.model_type == 'onnx':
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: image})

                filtered_output = self.__parse_model_pred(output=output, model_type=self.model_type, bbox_type='xyxy',
                                                          nms=True)

                if filtered_output is None:
                    print('未检测出目标')
                else:
                    print(filtered_output)
                    # 开始画图
                    for points in filtered_output:
                        xyxy = (points['lx'], points['ly'], points['rx'], points['ry'])
                        label = '%s|%.2f' % (points['name'], points['score'])
                        self.plot_one_box(xyxy, img=im0, label=label, color=(15, 155, 255), line_thickness=3,
                                          isShow=True)

            inference_time += time.time() - inference_start

        # print('The AVG [%s] run time is %.5f' % ('data_pre_process', data_pre_process_time / len(self.img_list)))
        #
        # print('The AVG [%s] run time is %.5f' % ('inference/ AI识别时间', inference_time / len(self.img_list)))
    def get_output(self,output,need_type):
        """
        @param output: 经过nms 解析为统一输出格式的结果，只需要根据业务修改这个函数
        @param need_type:

        """
        assert need_type in ('sx-cheng'), '业务类型不在列'
        if need_type == 'sx-cheng':
            seleted = None
            for i in output:
                if i['name'] == 'c':
                    seleted = i
                else:
                    print()

        return seleted
    def inference(self, img):
        data_pre_process_time = 0
        inference_time = 0
        data_pre_process_start = time.time()
        # 进行预处理操作
        im0, image = self.__pre_process(img)
        self.image_shape = (im0.shape[0], im0.shape[1])
        data_pre_process_time += time.time() - data_pre_process_start
        inference_start = time.time()

        # 如果使用的是mnn模型
        if self.model_type == 'mnn':
            tmp_input = MNN.Tensor(self.input_tensor_size, MNN.Halide_Type_Float, \
                                   image, MNN.Tensor_DimensionType_Caffe)
            self.input_tensor.copyFrom(tmp_input)
            self.interpreter.runSession(self.session)
            output = self.interpreter.getSessionOutputAll(self.session)

        # 如果使用的是onnx模型
        elif self.model_type == 'onnx':
            input_name = self.session.get_inputs()[0].name
            # 模型的直接输出结果
            output = self.session.run(None, {input_name: image})

            # 解析模型输出结果
            filtered_output = self.__parse_model_pred(output=output, model_type=self.model_type, bbox_type='xyxy',
                                                      nms=False)
            # 开始画图
            for points in filtered_output:
                xyxy = (points['lx'], points['ly'], points['rx'], points['ry'])
                self.plot_one_box(xyxy, img=im0, label=points['name'], color=(15, 155, 255), line_thickness=3,
                                  isShow=True)
            if len(filtered_output) == 0:
                print('未检测出目标')
            else:
                print(filtered_output)

        inference_time += time.time() - inference_start

        print('The AVG [%s] run time is %.5f' % ('data_pre_process', data_pre_process_time / len(self.img_list)))

        print('The AVG [%s] run time is %.5f' % ('inference/ AI识别时间', inference_time / len(self.img_list)))

    def __parse_model_pred(self, output=None, model_type=None, bbox_type='xyxy', nms=True):
        """
        using this method to uniform different model inference output
        retrun a list ,{'lx':item[0],'ly':item[1],'rx':item[2],'ry':item[3],'score':item[4],'name':self.classes[labels[idx]],}

        """

        # 未检出目标判断
        assert output is not None ,'没有检出目标'
        if output[0].shape == (1, 1, 5) and sum(output[0][:,0][0]) == 0:
            return None

        if model_type is 'onnx':
            pred = np.squeeze(output[0])  # [x,y,x,y,bbox_score]
            labels = np.squeeze(output[1])  # [1,2,0] 这里的数字分别对应类别中的第几类

            # pred要求 (lx,ly,rx,ry,bbox_score,cls)
            # nms
            if nms:
                nms_pred = InferenceDeepTk.non_max_suppression(
                    prediction=np.hstack(([pred, labels.reshape(labels.shape[0], 1)])),
                    conf_thres=0.2,
                    nms_thres=0.2)
            else:
                nms_pred = pred
            # filter bbox according to score
            filtered_pred = []
            if nms_pred[0] is None or len(nms_pred) == 0:
                return None

            # 过滤掉置信度低的
            for idx, item in enumerate(nms_pred):
                if item[4] > self.score_thr:  # if score > score thr
                    _ = {
                        'lx': item[0],
                        'ly': item[1],
                        'rx': item[2],
                        'ry': item[3],
                        'score': item[4],
                        'name': self.classes[labels[idx]]}

                    _ = InferenceDeepTk.__post_process(bbox=_, image_shape=self.image_shape,
                                                       input_shape=self.input_tensor_size)

                    # 处理界外的点
                    _['lx'] = self.image_shape[1] if _['lx'] > self.image_shape[1] else _['lx']
                    _['ly'] = self.image_shape[0] if _['ly'] > self.image_shape[0] else _['ly']
                    _['rx'] = self.image_shape[1] if _['rx'] > self.image_shape[1] else _['rx']
                    _['ry'] = self.image_shape[0] if _['ry'] > self.image_shape[0] else _['ry']

                    _['lx'] = 0 if _['lx'] < 0 else _['lx']
                    _['ly'] = 0 if _['ly'] < 0 else _['ly']
                    _['rx'] = 0 if _['rx'] < 0 else _['rx']
                    _['ry'] = 0 if _['ry'] < 0 else _['ry']

                    filtered_pred.append(_)

                else:
                    continue  # if nms_pred is sorted ,else continue

            if len(nms_pred) == 0:
                return None
            else:
                return filtered_pred


        elif model_type is 'mnn':
            pass
        else:
            raise TypeError()

    @staticmethod
    def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, min_wh=10):
        """
        Removes detections with lower object confidence score than 'conf_thres'
        Non-Maximum Suppression to further filter detections.
        min_wh = 2  # (pixels) minimum box width and height
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_conf, class)


        """
        # 如果只是一张图 则组成batch
        prediction = np.array([prediction])
        output = [None] * len(prediction)
        for image_i, pred in enumerate(prediction):
            # class_conf = np.max(pred[:, 4:], axis=1)
            # class_pred = np.argmax(pred[:, 4:], axis=1)
            # step 0 : 对所有框的conf进行从大到小的排序
            index = np.lexsort((pred[:, -2],))
            pred = pred[index]

            #  step 1: 去掉太小的框和置信度低的框
            i = (pred[:, 4] > conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
            pred2 = pred[i]

            # If none are remaining => process next image
            if len(pred2) == 0:
                continue

            # Select predicted classes
            # class_conf = class_conf[i]
            # class_pred = np.expand_dims(class_pred[i], 1)
            # class_conf = np.expand_dims(class_conf, 1)
            # numpy.concatenate((pred2[:, :5], class_conf, class_pred), 1)
            # Get detections sorted by decreasing confidence scores

            det_max = []
            nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)

            # 开始无类别的nms
            # Non-maximum suppression
            if nms_style == 'OR':  # default
                while pred2.shape[0]:
                    det_max.append(pred2[-1])  # save highest conf detection

                    pred2 = pred2[:-1]
                    if pred2.shape[0] == 0:  # Stop if we're at the last detection
                        break
                    iou = InferenceDeepTk.bbox_iou(det_max[-1], pred2)  # iou with other boxes
                    print(iou)
                    pred2 = pred2[iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    i = InferenceDeepTk.bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            # for c in np.unique(pred2[:, -1]):
            #     dc = pred2[pred2[:, -1] == c]  # select class c
            #     dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes
            #
            #     # Non-maximum suppression
            #     if nms_style == 'OR':  # default
            #         while dc.shape[0]:
            #             det_max.append(dc[:1])  # save highest conf detection
            #             if len(dc) == 1:  # Stop if we're at the last detection
            #                 break
            #             iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
            #             dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            #
            #     elif nms_style == 'AND':  # requires overlap, single boxes erased
            #         while len(dc) > 1:
            #             iou = InferenceDeepTk.bbox_iou(dc[0], dc[1:])  # iou with other boxes
            #             if iou.max() > 0.5:
            #                 det_max.append(dc[:1])
            #             dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            #
            #     elif nms_style == 'MERGE':  # weighted mixture box
            #         while len(dc):
            #             i = InferenceDeepTk.bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
            #             weights = dc[i, 4:5]
            #             dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
            #             det_max.append(dc[:1])
            #             dc = dc[i == 0]

            if len(det_max):
                # det_max = torch.cat(det_max)  # concatenate
                output = np.array([i.reshape(-1) for i in det_max])
                index = np.lexsort((output[:, -1],))
                output = output[index]
                output = np.flipud(output)

        return output

    @staticmethod
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
        inter_area = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0, 9999999) * \
                     (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0, 9999999)

        # Union Area
        union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        return inter_area / union_area  # iou

    @staticmethod
    def __post_process(bbox, image_shape, input_shape, bbox_type='xyxy', process_type='resize'):
        """
        对结果进行后处理

        """

        assert process_type in ('resize')  # we may add more pre process method
        assert bbox_type in ('xywh', 'xyxy')

        w_factor = image_shape[1] / input_shape[2]
        h_factor = image_shape[0] / input_shape[3]

        bbox['lx'], bbox['ly'] = int(bbox['lx'] * w_factor), int(bbox['ly'] * h_factor)
        bbox['rx'], bbox['ry'] = int(bbox['rx'] * w_factor), int(bbox['ry'] * h_factor)

        return bbox

    def __post_processes_with_decoder(self,mlvl,strides):
        """
        将mlvl的结果转化为标准的bbox 和 cls的输出
        :param mlvl: backbone + neck 的输出结果，list
        :param strides: 设定anchor的stride ，list
        :return: 标准的bbox与cls的输出
        """
        # step0：
        cls_scores = None
        bbox_preds = None
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        batch_size = cls_scores[0].shape[0]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        mlvl_priors = self._get_anchor_point(mlvl_priors, self.prior_generator.strides)
        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
        with_score_factors = False
        mlvl_score_factor = [None for _ in range(num_levels)]
        mlvl_batch_bboxes = []
        mlvl_scores = []
        batch_size = 1
        print('开始处理mlvl bboxes 和mlvl scores')
        for cls_score, bbox_pred, score_factors, priors in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # torch.Size([1, 6400, 3]) 每个点预测属于每一类的分数
            scores = cls_score.permute(0, 2, 3,
                                       1).reshape(batch_size, -1,
                                                  self.cls_out_channels)

            # 默认是要使用sigmoid的，但是我要去掉，而且也不要softmax
            # if self.use_sigmoid_cls:
            #     scores = scores.sigmoid()
            #     nms_pre_score = scores
            # else:
            #     scores = scores.softmax(-1)
            #     nms_pre_score = scores
            scores = scores.sigmoid()

            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(
                    batch_size, -1).sigmoid()

            # bbox 是torch.Size([1, 108800, 4])
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            # priors 是torch.Size([1, 6400, 4])
            priors = priors.expand(batch_size, -1, priors.size(-1))
            # Get top-k predictions
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])

            if nms_pre > 0:

                if with_score_factors:
                    nms_pre_score = (nms_pre_score * score_factors[..., None])
                else:
                    nms_pre_score = nms_pre_score

                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)

                batch_inds = torch.arange(
                    batch_size, device=bbox_pred.device).view(
                        -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                priors = priors.reshape(
                    -1, priors.size(-1))[transformed_inds, :].reshape(
                        batch_size, -1, priors.size(-1))
                pdb.set_trace()
                bbox_pred = bbox_pred.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                scores = scores.reshape(
                    -1, self.cls_out_channels)[transformed_inds, :].reshape(
                        batch_size, -1, self.cls_out_channels)
                if with_score_factors:
                    score_factors = score_factors.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)

            pdb.set_trace()
            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)
            pdb.set_trace()
            print('完成了decoder')
            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)
        # step1: decoder

        # step2:

    def _get_anchor_point(self,mlvl_priors,strides):
        """
        Converting mlvl grid anchors to mlvl points anchors,

        Created by haoran in deeptk
        Fix bug for gfl ,while converting model from pytorch to onnx,
        default anchor is grid anchor ,but point anchor is needed.

        :param mlvl_priors: multi level anchors
        :param strides: multi level anchor strides
        :return: anchor points
        """
        assert isinstance(mlvl_priors,list), 'Only support for mlvl priors'
        assert isinstance(strides,list),'Only support for mlvl priors'
        assert strides[0][0] == strides[0][1],'Only support square stride'
        new_mlvl_priors = []
        for priors,stride in zip(mlvl_priors,strides):
            new_mlvl_priors.append(self.anchor_center(priors)/stride[0])

        return new_mlvl_priors

if __name__ == '__main__':
    # img_path = 'test_data/1.jpg'
    img_dir = {'hand': '../tmp/yolo2coco/images',
               'cargoboat': '/Users/musk/Desktop/实习生工作/dataset/cargoboat_split/train_img',
               'garbage': '/Users/musk/Desktop/实习生工作/dataset/FloW_IMG/training/images',
               'sx-hand': '/Users/musk/Movies/hand检测测试数据',
               'sx-hand2': '/Users/musk/Movies/sx-hand-video2',
               'coco-hand':'/Users/musk/Desktop/实习生工作/COCO-Hand/COCO-Hand-S/COCO-Hand-S_Images',
               'sx-client-data':'/Users/musk/Desktop/实习生工作/杂类',
               }

    model_path = {
        'hand-mobv3': './model_zoo/mobv3-hand.onnx',
        'cargoboat-ssd': './model_zoo/ssd300_cargoboat.onnx',
        'garbage-ssdlite': './model_zoo/ssdlite_mobilenetv2_garbage.onnx',
        'model-s-hand':'./model_zoo/model-s-hand.onnx',
        'pipeline-small-秤':'./model_zoo/pipeline-small-秤.onnx'
    }

    input_tensor_size = {
        'hand-mobv3': (1, 3, 320, 320),
        'cargoboat-ssd': (1, 3, 300, 300),
        'garbage-ssdlite': (1, 3, 320, 320),
        'pipeline-small':(1, 3, 320, 320),
    }

    classes = {
        'hand': ['hand'],
        'cargoboat': ['cargoboart', 'otherboat'],
        'garbage': ['bottle'],
        'sx-cheng':['d','w','c']
    }

    # step1 : 选择模型
    model_path = model_path['pipeline-small-秤']

    # step2：: 设置输入tensor 的shape
    input_tensor_size = input_tensor_size['pipeline-small']
    classes = classes['sx-cheng']

    print('Testing model :', model_path)

    eval_model = InferenceDeepTk(model_path=model_path, input_tensor_size=input_tensor_size, classes=classes)
    # eval_model.inference(img='/Users/musk/PycharmProjects/yolo_v3-master/deployment/images/hand.png')
    eval_model.inference_batch(img_dir['sx-client-data'])
