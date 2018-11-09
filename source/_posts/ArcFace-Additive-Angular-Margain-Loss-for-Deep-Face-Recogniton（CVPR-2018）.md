---
title: 'ArcFace:Additive Angular Margain Loss for Deep Face Recogniton（CVPR 2018）'
categories:
  - 论文阅读
date: 2018-11-09 20:46:10
tags:
---
主要是模式识别的seminar的看的
# 简介 #
目前深度卷积神经网络在人脸识别任务上取得了很好的效果，不同的神经网络主要在训练数据集、网络设置和损失函数三种属性有所区别，

文章据此做了相关改进，并主要有以下四个贡献

1.清洗了最大人脸公共训练数据集（MS1M）和测试数据集（MegaFace）

2.探索不同网络设置，并分析精度与速度之间的关系

**3.提出了一种几何可解释的损失函数ArcFace，并优于softmax，SphereFace和CosineFace**

4.在MegaFace人脸数据集上取得了最先进的表现
# 从softmax到arcFace #
softmax仅能做到分类，但是我们人脸识别做的不仅仅是分类，而是相同的人尽量聚在一起，不同的人尽量分开

最原始的softmax函数

<img src='/images/paper/arc1.png' width="640"/>

<img src="/images/paper/arc2.png" width='640'/>

令偏置b为0，然后权重和输入的内积用上面式子表示，用L2正则化处理Wj使得||Wj||=1，L2正则化就是将Wj向量中的每个值都分别除以Wj的模，从而得到新的Wj，新的Wj的模就是1

<img src="/images/paper/arc3.png" width='640'/>

然后一方面对输入xi也用L2正则化处理，同时再乘以一个scale参数s；另一方面将cos(θyi)用cos(θyi+m)替代

<img src="/images/paper/arc4.png" width='640'/>

# 几何解释 #
使用二分类来证明，决策边界就是在这个边界上属于正类和属于负类的概率一样的，所以可以得到以下这个表格和图像
<img src="/images/paper/arc5.png" width='640'/>

<img src="/images/paper/arc6.png" width='640'/>

一些实验

<img src="/images/paper/arc7.png" width='640'/>








