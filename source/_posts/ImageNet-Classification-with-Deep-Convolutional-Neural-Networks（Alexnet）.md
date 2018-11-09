---
title: ImageNet Classification with Deep Convolutional Neural Networks（Alexnet）
categories:
  - 论文阅读
date: 2018-11-09 21:02:33
tags:
---
ImageNet Classification with Deep Convolutional Neural Networks（Alexnet）
#1.创新的地方#
（1）使用ReLu函数作为CNN的激活函数

（2）使用dropout防止过拟合

（3）使用重叠最大池化

（4）提出了LRN层

#2.摘要#
（1）提出了一个深度卷积网络，在ImageNet LSVRC2010，top1和top5错误分别是37.5%、17.0%远远好于之前的表现

（2）神经网络有6000万个参数，650000神经元，5个卷积层（某些卷积层后面有池化层）和三个全连接层

（3）为减少全连接层的过拟合，使用了一个dropout的正则化方法

#3.图片预处理#
（1）大小归一化
将图像进行下采样到固定的256*256分辨率，给定一个矩形图像，首先缩放图像短边长度为256，然后从结果图像中裁剪中心的256*256大小的图像块
（2）减去像素平均值
所有图片的每个像素值都减去所有训练集图片的平均值
#3.架构#
##3.1ReLU非线性##
训练速度比之前的sigmod和tanh快，并解决了之前激活函数的梯度消失问题，（详见5ReLU函数）
##3.2局部响应归一化##
提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。（ps，博客上都说他其实没什么用）
##3.3重叠池化##
在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化，AlexNet全部使用最大池化，避免平均池化的模糊化效果。并且AlexNet中提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。Alexnet设置s=2，z=3，这个方案分别降低了top-1 0.4%，top5 0.3%的错误率。且论文提到重叠池化的模型更难过拟合。
##3.4整体架构##
8个带权重的层，前5层是卷积层，剩下3层是全连接层。最后一层是1000维softmax输入

<img src="/images/paper/alex1.png" width="640"/>

<img src="/images/paper/alex2.png" width="640"/>
##3.5减少过拟合##
###3.5.1.数据增强###

方法一：水平翻转和图像变换（随机裁剪）

训练时：随机裁剪224*224的图片和进行水平翻转，所以数据增大了（250-224）平方*2=2048倍

在测试时，网络通过提取5个224x224块（四个边角块和一个中心块）以及它们的水平翻转（因此共十个块）做预测，然后网络的softmax层对这十个块做出的预测取均值。

方法二：PCA Jittering（详见7 PCA Jittering）（减少了TOP1错误率 1%以上）

###3.5.2.dropout###

训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。Dropout虽有单独的论文论述，但是AlexNet将其实用化，通过实践证实了它的效果。在AlexNet中主要是最后几个全连接层使用了Dropout。（也有论文，单独看下）
朴素思想：以0.5的概率让隐藏层的输出为0，失活的神经元不再进行前向传播且不参与反向传播

alexnet：前两个全连接层使用dropout，会减少过拟合，但会使训练迭代次数翻一倍

#4.学习细节#
随机梯度下降来训练模型，batch size为128，动量是0.9（详见6.动量），权重衰减为0.0005
<img src="/images/paper/alex3.png" width="640"/>
使用均值是0，标准差是0.01的高斯分布对每一层的权重进行初始化。2,4,5卷积层和全连接层偏置初始化为1，其余层偏置初始化为0


#5.ReLu函数#
AlexNet成功使用ReLU作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题。而且训练速度也比之前的sigmod和tanh快。虽然ReLU激活函数在很久之前就被提出了，但是直到AlexNet的出现才将其发扬光大。
##5.1训练速度##
使用ReLu的四层卷积神经网络在CIFAR-10数据集上达到25%的训练误差比使用tanh神经元的等价网络（虚线）快六倍
<img src='/images/paper/alex4.png' width="640"/>
##5.2梯度消失和梯度爆炸##
<img src="/images/paper/alex5.jpg" width='640'/>
如上图所示的网络结构，我们在进行反向传播更新w1权值的时候，公式如下
<img src='/images/paper/alex6.jpg' width='640'/>

sigmod函数导数如下图所示
<img src="/images/paper/alex7.png" width='640'/>
从图中可以看出最大值是0.25，所以存在一下两种现象
<img src="/images/paper/alex8.jpg" width="640"/>
##5.3 ReLu函数##

relu函数解决问题的方式很简单，如果偏导数是1的话就不存在爆炸与消失的问题了，relu函数图像如下
<img src="/images/paper/alex9.jpg" width="640"/>
从上图中，我们可以很容易看出，relu函数的导数在正数部分是恒等于1的，因此在深层网络中使用relu激活函数就不会导致梯度消失和爆炸的问题。
#6.动量#
在物理世界，球体运动是有惯性的，所以可能翻过山头达到全局最优解
<img src="/images/paper/alex10.png" width="640"/>
把这种概念引入到梯度下降中去，每次更新时不仅考虑当前梯度影响，还要考虑上一步的梯度影响
<img src="/images/paper/alex11.png" width="640"/>
有可能会翻过去找到全局最优解
<img src="/images/paper/alex12.png" width="640"/>
#7 PCA Jittering#
##7.1PCA主成分分析##
主成分分析（Principal components analysis，以下简称PCA）是最重要的降维方法之一。PCA顾名思义，就是找出数据里最主要的方面，用数据里最主要的方面来代替原始数据。
主要计算过程：
假设有一数组
<img src="/images/paper/alex13.jpg" width="640"/>
第一步：分别求出x和y的平均值，对于每个样例减去均值得到
<img src="/images/paper/alex14.jpg" width="640"/>
第二步，求协方差矩阵得到
这是3维的协方差形式
<img src="/images/paper/alex15.jpg" width="640"/>

协方差（Covariance）在概率论和统计学中用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。（这里的方差是修正样本方差，分母是 样本数量-1）
<img src="/images/paper/alex16.jpg" width="640"/>
协方差是衡量两个变量同时变化的变化程度。协方差大于0表示x和y若一个增，另一个也增；小于0表示一个增，一个减。如果ｘ和ｙ是统计独立的，那么二者之间的协方差就是０；但是协方差是０，并不能说明ｘ和ｙ是独立的。协方差绝对值越大，两者对彼此的影响越大，反之越小。
第三步：求协方差的特征值和特征向量
<img src="/images/paper/alex17.png" width="640"/>
<img src="/images/paper/alex18.jpg" width="640"/>
在数学上，特别是线性代数总，对于一个给定的矩阵A，它的特征向量v进行经过这个线性变化后，得到新的向量仍然和原来的v保持在一条直线上，但其长度或方向或许会改变，即

Av=λν

λ是特征值，v是特征向量
（1）如果矩阵可以变换成为对角矩阵，那么它的特征值就是他对角线上的元素，而特征向量就是相应的基：例如矩阵
<img src="/images/paper/alex19.png" width="640"/>
的特征值就是2和4,2对应的特征向量就是所有形同（a，b，0）T的非零向量，4对应的特征向量是所有形同（0,0，c）T的非零向量
（2）对于一般的矩阵
<img src="/images/paper/alex20.jpg" width="640"/>
第四步：将特征值按照从大到小进行排序，选择其中最大的k个，然后将其对应的特征向量分别作为列向量组成特征向量矩阵
第五步：将样本点投影到选取的特征向量上。假设样例数为m，特征数为n，减去均值后的样本矩阵为DataAdjust(m*n)，协方差矩阵是n*n，选取的k个特征向量组成的矩阵为EigenVectors(n*k)。那么投影后的数据FinalData为
FinalData(10*1) = DataAdjust(10*2矩阵) x 特征向量(-0.677873399, -0.735178656)T
最后结果是他
<img src="/images/paper/alex21.jpg" width="640"/>
这样就把原始的n维降到了k维