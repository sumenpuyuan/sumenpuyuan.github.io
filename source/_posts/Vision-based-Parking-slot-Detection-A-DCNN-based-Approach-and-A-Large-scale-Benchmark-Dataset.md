---
title: >-
  Vision-based Parking-slot Detection:A DCNN-based Approach and A Large-scale
  Benchmark Dataset
categories:
  - 论文阅读
date: 2018-11-29 22:14:02
tags:
---
# 摘要 #
两个贡献，
1.提出了基于深度卷积神经网络的停车位检测方法，以环绕图像作为输入，主要有两个步骤，识别标记点，根据标记点形成的局部图像进行分类
2.建立了一个大规模标记数据集，是目前停车领域中最大的，包含12165个环绕图像
# 介绍 #
## A基于视觉的停车位检测 ##
车停位检测方法主要分为两种，基于自由空间和基于视觉的。基于自由空间的方法通过识别相邻车辆之间的适当空置空间来指定目标停车位置。这是最广泛使用的方法，基于自由空间的方法具有固有的缺点，即它必须依赖已经适当停放的车辆作为参考。换句话说，这种方法不能在没有车辆的开放区域内工作。此外，其准确性很大程度上取决于相邻车辆的位置和姿势。基于视觉的方法的工作原理与基于自由空间的方法的工作原理根本不同。
基于视觉的方法的目标是识别和定位由地面上绘制的停车线段定义的停车位。显然，这种方法的表现并不取决于相邻车辆的存在或姿势。此外，在大多数情况下，停车线段可以提供比“自由空间”更准确的停车信息。
<img src="/images/paper/che01.png"/>
<img src="/images/paper/che02.jpg"/>
## B深度卷积神经网络 ##
物体检测领域的发展是与我们的工作非常相关。应用DCNN解决物体检测任务始于Girshick等人的开创性工作R-CNN [50]。 R-CNN实际上是一个多阶段检测框架。给定输入图像，它首先使用对象建议算法来找到包含具有高概率的对象的边界框。然后，标准DCNN作为特征提取器应用于每个提出的边界框，最后一个分类器决定框内的对象类。根据R-CNN的框架，许多研究人员提出了改进R-CNN性能的修改，并且沿着这个方向的一些代表性方法是Fast-RCNN [51]，FasterRCNN [52]，HyperNet [53]等.R-CNN和它的所有变体都高度依赖于对象提议。因此，对象提议算法的性能成为瓶颈。
最近，一些研究人员开始在基于DCNN的物体检测系统中挑战对象提议算法的必要性，并且他们将对象检测作为回归问题制定为空间分离的边界框和相关的类概率。 这些方法的代表包括Yolo [54]，SSD [55]和YoloV2 [56]。 不同研究小组的实验结果表明，这种方法可以获得与R-CNN类似方法一样精确的边界框，但运行速度更快。 因此，在我们的停车位检测方法中，采用DeepPS，YoloV2来检测标记点（详情请参阅第III-A节）。
总体流程
# DEEPPS：基于DCNN的方法 #
## A标记点检测 ##
为了训练探测器，我们需要准备训练样本。在训练样本准备阶段，在给定的环绕视图图像上，手动标记其所有标记点的位置。对于每个标记点pi，以pi为中心的固定大小p×p的方框被认为是pi的groundtruth边界框。

非常希望训练的标记点检测器具有旋转不变性。为了实现这一目标，我们通过旋转每个原始标记图像来增强训练集，以生成许多旋转版本。具体地，从每个原始标记图像I，我们可以获得其J个旋转版本\\(\{I_j\}^{J-1}_{j=0}\\)，其中\\(I_j\\)通过以\\(\frac {360}{J} \* j\\)度旋转I来生成。当然，标记点的标记坐标以相同的方式旋转。用于数据增强的这种想法通过图5中所示的示例来说明。图5（a）是原始标记图像，而图5（b）是通过将图5（a）旋转30度而生成的。标记点表示为紫色点，相关的边界框显示为黄色方块。在实施过程中，**基于YoloV2的标记点检测器D由在VOC数据集上训练的模型进行了调整，该模型由YoloV2的作者提供。。 对于微调，小批量大小设置为64; 学习率从0.0001开始，每50000次迭代除以10。 我们使用了0.0005的重量衰减和0.9的动量。 **经过训练的标记点检测器在测试阶段表现良好。 在Sect。 IV-C，我们将定量评估其性能。
<img src="/images/paper/che03.jpg"/>
## B局部图像模式分类 ##
在测试图像上应用标记点检测器D后，具有大于δ1的置信度得分的点将被视为标记点。假设p1和p2是两个检测到的标记点。我们需要验证它们是否可以形成有效的入口线。首先，如果---→p1p2可以是有效的入口线候选者，则p1和p2之间的距离应该满足一些约束。如果---→p1p2是平行停车位的入口线候选者，则需要满足t1 <∥p1p2∥<t2;如果---→p1p2是垂直或倾斜停车槽的入口线候选者，它应该满足t3 <∥p1p2∥<t4。参数t1，t2，t3和t4基于关于各种类型的停车位的入口线长度的先验知识来设置。
然后，我们需要进一步处理满足距离约束的标记点对。首先，对于一对标记点，尽管它可以满足距离约束，但很可能它们仍然不能形成有效的入口线。例如，在图6中，p1和p2之间的距离满足作为平行停车槽入口线的距离约束;然而，很明显---→p1p2不是有效的入口线，因为它通过另一个标记点。另外，假设---→p1p2是有效的入口线。我们需要确定相关的停车槽是在其顺时针侧还是在其逆时针侧，并确定该停车槽是直角还是倾斜1。所有这些问题都可以通过将由p1和p2定义的局部图像模式分类为预定类之一来解决。如图7（a）所示，如下提取由环绕视图像上的两个标记点p1和p22定义的局部图像图案。首先，建立局部坐标系，以p1和p2的中点为原点，以---→p1p2为X轴。因此可以确定其Y轴。在该坐标系中，我们定义了矩形区域R，其对应于X轴和Y轴。对于R，沿X轴的边长设定为∥p1p2∥+Δx，沿Y轴的边长设定为Δy。我们从环绕视图图像中提取由R覆盖的图像区域，将其标准化为尺寸w×h，并将得到的图像块视为由p1和p2定义的局部图像图案。
在训练阶段，基于标签数据，我们可以获得包括由成对标记点定义的所有局部图像图案的集合C.根据相关停车位的特点，我们将C中的样本分为7类，“直角逆时针3”，“逆时针倾斜，急停车角4”，“逆时针倾斜，钝角停车”，“右 - 顺时针方向“，”顺时针倾斜，有一个钝角停车角“，”顺时针方向倾斜，有一个急停车角“，”无效“。属于这些类别的代表性范例分别如图7（b）〜（h）所示。在构造C时，我们遇到了一个实际问题，即类不平衡，这意味着特定类相对于其他类具有非常少量的实例。在我们的例子中，这个问题的根本原因是，在我们收集的数据集中，倾斜停车位的数量远小于直角停车位的数量。为了解决这个问题，我们采用了SMOTE [58]对少数民族进行过度抽样。从C开始，我们可以训练分类模型M来预测从测试环绕视图图像中提取的看不见的局部图像模式的类标签。

<img src="/images/paper/che04.png"/>

<img src="/images/paper/che05.jpg"/>
定制的DCNN架构旨在解决我们的分类任务。如图8所示，我们的网络采用灰度48×192图像作为输入，输出层有7个节点，对应于7类本地图像模式。在图8中，“conv”表示它是卷积层，“ReLU”表示它是一个完整的线性单元[60]层，“max-pool”表示它是一个最大池层，“BN”表示它是一个批量标准化层，“FC”表示它是完全连接层。对于每个“conv”，“max-pool”和“FC”层，其参数设置及其输出的维度（该层的特征映射）如表I所示。“kernel：[kernel h，kernel w] “指定每个滤波器的高度和宽度，”pad：[pad h，pad w]“指定要添加到输入每一侧的像素数，”stride：[stride h，stride w]“指定应用滤波器的时间间隔输入和“num output”指定过滤器的数量。**定制的DCNN模型首先在ImageNet 2012分类数据集上进行了训练，然后针对我们的特定任务进行了调整。对于微调，将mini-batchsize设置为256;学习率从0.002开始，每5000次迭代除以2。我们使用了0.0005的重量衰减和0.9的动量。**
## C停车位推断 ##
在自助泊车系统中，停车位通常被认为是平行四边形，并且通常由其四个顶点的坐标表示。在大多数情况下，两个非标记点顶点是不可见的，它们的坐标只能通过推理获得。为此，我们需要假设停车位的“深度”事先已知为先验知识。如图9所示，假设垂直，平行和倾斜的停车槽的深度分别为d1，d2和d3。支持p1和p2被改为标记点，并且由p1和p2定义的局部图像模式被分类为“顺时针顺时针”或“直角逆时针”。如果是这种情况，则可以容易地计算两个非标记点顶点p3和p4的坐标。例如，在图9（a）中，由p1和p2定义的局部图像模式是“顺时针直角”和长度---→p1p2表示该停车槽应该是垂直的（不是平行的），因此它的“深度”是d1。因此，它的p3和p4被推断为，
<img src="/images/paper/che06.jpg"/>
<img src="/images/paper/che07.jpg"/>
 在图9（c）中，由p1和p2定义的局部图像模式被分类为“具有急性停车角的逆时针倾斜”; 为了估计两个非标记点顶点的位置，需要估计停车角α。 为解决此问题，使用基于模板匹配的策略。 如图9（d）所示的一组理想的“T形”模板\\(\{T_{θ_j}\}^M_{j=1}\\)，其中θj是模板j的两条线之间的角度，M是总数模板。 每个模板的大小为s×s，为零均值。 在测试阶段，分别提取以p1和p2为中心的两个s×s图像块I1和I2。 I1和I2都与---→p1p2对称。 然后，停车角α可以自然地估计为，
<img src="/images/paper/che08.jpg"/>
在计算了停车角度之后，可以直接计算两个非标记点的坐标。例如，在图9（c）中，p3和p4的坐标可以表示为，
<img src="/images/paper/che09.jpg"/>
<div id="all"/>
**训练阶段**
**输入**：一个集合S，包括带有标记标记点和停车位的环绕视图图像
**输出**：标记点检测器D和图像模式分类器M
1.数据增强：通过图像旋转得到增强数据集S'
2.对于每个标记点\\(p_i\\)，生成一个p×p的边界框\\(B_i\\)
3.resize到416 × 416，输入S'和\\({B_i}\\)给yoloV2，进行训练，得到分类器D
4.得到数据集C，进行7类图像模式划分（以p1 p2重点为原点，p1-p2作为x轴，矩形框x轴长∥p1p2∥+Δx，y轴边长Δy，切割形成局部图像，标准化尺寸为w×h）
5.使用SMOTE对C的少数类进行过采样
6.C输入DCNN,开始训练，得到分类器M

**测试阶段**
**输入**：一个环绕图像I，标记点检测器D，图像分类器M
**输出**：一个包含从I得到的停车位集合P
1.I输入给D，得到一个标记点集合\\(\{p_i\}^L_{i=1}\\)
2.
for i=1:L-1
&emsp;&emsp;for j =i+1:L
&emsp;&emsp;&emsp;&emsp;if ||\\(p_i,p_j\\)||不满足有效入口线的条件（如果是平行停车位，t1<||\\(p_i,p_j\\)||<t2,如果是垂直或者倾斜，t3<||\\(p_i,p_j\\)||<t4）
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;continue
&emsp;&emsp;&emsp;&emsp;end if
&emsp;&emsp;&emsp;&emsp;根据pi，pj提取局部图像，使用M进行分类得到结果c
&emsp;&emsp;&emsp;&emsp;if c == "invalid"
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;continue
&emsp;&emsp;&emsp;&emsp;end if
&emsp;&emsp;&emsp;&emsp;if c=="直角顺时针" or "直角逆时针"
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;根据公式1得到非标记点\\(p_{ij}^3\\)和\\(p_{ij}^4\\)
&emsp;&emsp;&emsp;&emsp;else
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;根据公式2估计停车角度\\(\alpha\\)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;根据公式3得到非标记点\\(p_{ij}^3\\)和\\(p_{ij}^4\\)
&emsp;&emsp;&emsp;&emsp;end if
&emsp;&emsp;&emsp;&emsp;\\(ps=[p_i,p_j,p_{ij}^3,p_{ij}^4]\\)
&emsp;&emsp;&emsp;&emsp;if 如果ps中的顶点按逆时针顺序排列
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\\(ps=[p_j,p_i,p_{ij}^4,p_{ij}^3]\\)
&emsp;&emsp;&emsp;&emsp;end if
&emsp;&emsp;&emsp;&emsp;ps加入P
&emsp;&emsp;end for
end for