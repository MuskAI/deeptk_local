## DeepTk流水线 模型部署

当通过流水线得到onnx或mnn模型后，需要进行推理得到不同业务所需要的结果，这就是DeepTk模型模部署类需要解决的问题

### 具体内容

```yaml
-deployment
  -images:存放要测试的单张图片
  -model_zoo:存放各种模型
  -inference_deeptk.py :深想模型推理类
  -README.md
```

1. nms后处理操作实现
2. 解析模型输出，统一成(x1,y1,x2,y2,bbox_score,cls_score)
3. 画图功能
4. 速度测试



### Usage

实例化推理类
```yaml
eval_model = InferenceDeepTk(model_path=model_path, input_tensor_size=input_tensor_size, classes=classes)
```
推理单张图片

```yaml
eval_model.inference(img='/Users/musk/PycharmProjects/yolo_v3-master/deployment/images/hand.png')
```


推理文件夹下的所有图片
```yaml
eval_model.inference_batch(img_dir['sx-hand2'])
```

