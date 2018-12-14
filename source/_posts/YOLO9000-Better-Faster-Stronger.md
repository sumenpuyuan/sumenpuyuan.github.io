---
title: 'YOLO9000:Better,Faster,Stronger'
categories:
  - 论文阅读
date: 2018-12-09 20:32:59
tags:
---
# 更好
yolo与最先进的检测器相比，会有更多的定位误差，此外yolo和基于区域提出的方法相比，召回率相对较低。所以主要侧重定位误差和召回率，同时保证分类准确性

## batchNorm， ##
 在所有卷积层上加bachnorm，map获得超过2%的改进。并且可以移除dropout不会过拟合。
##  High Resolution Classifier
高分辨率分类器，之前是在ImageNet 224×224 上训练，在448×488检测数据集上finetune，这样直接切换可能模型可能难以快速适应高分辨率。所以v2加了个在Imagenet数据集上使用 448\数据集上使用 448×448 输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。448 输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。
## anchor Box
借鉴FasterRCNN的anchorbox，预测边界框关于先验框的偏移值。对于YOLOv1，每个cell都预测2个boxes，每个boxes包含5个值： (x, y, w, h, c) ，前4个值是边界框位置与大小，最后一个值是置信度（confidence scores，包含两部分：含有物体的概率以及预测框与ground truth的IOU）。但是每个cell只预测一套分类概率值（class predictions，其实是置信度下的条件概率值）,供2个boxes共享。YOLOv2使用了anchor boxes之后，每个位置的各个anchor box都单独预测一套分类概率值，
## Dimension Clusters维度聚类。
之前的先验框是手工选择的，我们可以使用聚类算法对真实数据边界框进行聚类，自动找到好的先验。如果使用欧几里得距离作为距离度量，那么大的边界框具有更大的误差，所以采用了以下的距离度量
$$ d(box,centroid)=1-IOU(box,centroid) $$
<img src="/images/paper/yolov201.jpg"/>
聚类结果比使用手工选择的先验结果要好
## Direaction location prediction
直接位置预测，使用anchorbox，遇到一个问题，模型不稳定，早期迭代过程中，大部分不稳定来自预测边界框位置，在区域提出网络中，边界框实际中心位置（x,y），预测偏移值\\(（t_x,t_y）\\),先验框尺度\\((w_a,h_a)\\)中心坐标\\((x_a,y_a)\\)
<img src="/images/ML/yolov202.jpg"/>

这个公式是不受限制的，所以任何锚盒都可以在图像任一点结束，而不管在哪个位置预测该边界框。随机初始化模型需要很长时间才能稳定以预测合理的偏移量。
所以，YOLOv2弃用了这种预测方式，而是沿用YOLOv1的方法，就是预测边界框中心点相对于对应cell左上角位置的相对偏移值，为了将边界框中心点约束在当前cell中，使用sigmoid函数处理偏移值，这样预测的偏移值在(0,1)范围内（每个cell的尺度看做1）。网络预测每个边界框的4个偏移\\(t_x，t_y,t_w,t_h\\)
<img src="/images/ML/yolov203.jpg"/>
从这个图可以很好理解上难免公式
<img src="/images/paper/yolov204.jpg"/>
## 细粒度功能。##
YOLOv2的输入图片大小为 416×416 ，经过5次maxpooling之后得到 13×13 大小的特征图，并以此特征图采用卷积做预测。 13×13 大小的特征图对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained Features）。因此SSD使用了多尺度的特征图来分别检测不同大小的物体，前面更精细的特征图可以用来预测小物体。YOLOv2提出了一种passthrough层来利用更精细的特征图。YOLOv2所利用的Fine-Grained Features是 26×26 大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为 26×26×512 的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个 2×2 的局部区域，然后将其转化为channel维度，对于 26×26×512 的特征图，经passthrough层处理之后就变成了 13×13×2048 的新特征图（特征图大小降低4倍，而channles增加4倍，图6为一个实例），这样就可以与后面的 13\times13\times1024 特征图连接在一起形成 13×13×3072 大小的特征图，然后在此特征图基础上卷积做预测。在YOLO的C源码中，passthrough层称为reorg layer。在TensorFlow中，可以使用tf.extract_image_patches或者tf.space_to_depth来实现passthrough层：
## 多尺度训练 ##
由于YOLOv2模型中只有卷积层和池化层，所以YOLOv2的输入可以不限于 416×416 大小的图片。为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。
