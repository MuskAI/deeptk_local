"""Exports a pytorch *.pt model to *.onnx format

Usage:
    import torch
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import onnx
import torch


from tmp.yolov3 import Yolov3Tiny
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./hand-tiny_512-2021-02-19.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    # google_utils.attempt_download(opt.weights)
    img_size = 416

    a_scalse = 416. / img_size
    anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
    anchors_new = [(int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse)) for j in range(len(anchors))]

    model = Yolov3Tiny(num_classes=1, anchors=anchors_new)
    torch_model = torch.load(opt.weights,map_location='cpu')['model']
    model.load_state_dict(torch_model)
    model.eval()
    # model.fuse()

    # Export to onnx
    # model.model[-1].export = True  # set Detect() layer export=True
    _ = model(img)  # dry run
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['output'])  # output_names=['classes', 'boxes']

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
