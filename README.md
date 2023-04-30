## DeepTk流水线
> mmlab框架负责训练，但在训练之前数据处理成coco deeptk格式，在训练之后需要进行详细的测试和推理
> 简单来说：mmlab之前和之后的工作都由deeptk_local完成
> 


##具体内容

```yaml

-deeptk_local
  -data_fromat_convert : 将标注数据转换成统一的deeptk coco格式
  -deployment : 在mmlab中完成模型转换后，针对转换后的模型后续的操作,比如推理、画图、性能测试、nms等
  -model_test : 按照客户要求进行各种测试
  -test_data : 测试数据
  -tmp : 不用看，暂存文件，未来会删除
  -cfg : 原本yolov3模型的参数，我们之前的hand 检测是使用yolov3的，之后会删除
  
```

关于各个文件的具体内容，将在子目录下的  ```README.md``` 文件中给出说明


