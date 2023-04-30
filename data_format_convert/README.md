## DeepTk流水线 数据集转换


统一数据集标注格式是DeepTk流水线的第一步，在这里支持下列格式的数据集转换成DeepTk COCO标注格式

```yaml
voc(xml文件) -> coco
yolo(txt文件) -> coco
```

特别的，由于不同业务的需求，针对特定数据集也实现转换

```yaml
coco-hand数据集 ->coco
egohands(matlab mat 文件) -> coco
```

### 具体内容

```yaml
-data_format_convert
  -voc2deeptkcoco.py  : 将voc数据格式转化为coco数据格式
  -yolo2deeptkcoco.py : 将yolo数据格式转化为coco数据格式
  -COCO_Image_Viewer.ipynb : 可视化转化好的coco格式数据集，Note：pycharm内显示bbox会有偏差
  
  上面为通用数据格式转换
  -----------------------
  下面为特殊数据格式转换
  
  -coco-hand2coco.py : 将coco-hand数据集转化为coco数据格式
  -egohand2coco.py : 将egohand数据集转化为coco数据格式
```

我们的代表工具输出yolo格式的标注文件
1. txt文件名与图片名相同
2. 类别 x y w h的方式
``
0 0.428125 0.619444 0.147917 0.316667
``


### 项目说明

#### 火焰检测项目

数据集处理代码在 ```furg2coco.py```

标注格式不是voc格式，而且对应的bbox坐标也有问题，因此进行如下操作：
    
1. 将格式转为标准voc
2. voc2coco
3. 可视化找到坐标的规律，并修正坐标（尝试无果，不知道标的是什么狗屁坐标，所以直接从最原始的视频中截取


```xml
<?xml version="1.0" encoding="utf-8"?>

<annotation>
  <filename>barbecue0.png</filename>
  <object>
    <name>fire</name>
    <bndbox>
      <xmin>191</xmin>
      <ymin>109</ymin>
      <xmax>288</xmax>
      <ymax>371</ymax>
    </bndbox>
  </object>
</annotation>

```
 要转换成如下
 
```xml
<annotation verified="yes">
	<folder>JPEGImages</folder>
	<filename>ship20.jpg</filename>
	<path>C:\Users\ecidi\Desktop\图片打标\sx_mask_dataset\JPEGImages\ship20.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>800</width>
		<height>600</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>otherboat</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>70</xmin>
			<ymin>197</ymin>
			<xmax>733</xmax>
			<ymax>424</ymax>
		</bndbox>
	</object>
</annotation>
```

默认是要通过xml中的filename，或者在没有指定filename的时候通过path获取。
但是在处理火焰数据的时候，voc格式不依靠filename，所有我需要增加一个从文件名中获取filename的功能

