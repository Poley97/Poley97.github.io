---
layout: post
title: 'Point-cloud based 3D object detection and classification methods for self-driving applications: A survey and taxonomy'
date: 2021-05-26
author: Poley
cover: '/assets/img/20210526/PCReview.png'
tags: 论文阅读
---

> 论文链接： https://www.sciencedirect.com/science/article/pii/S1566253520304097


个人感想：这是一篇非常全面的综述，完整仔细阅读一遍基本相当于整理了一遍我之前所看过的论文，值得一读。


# Introduction

目前随着3D传感器技术的快速发展，激光雷达LiDAR已经呗广泛使用，每个LiDAR都可以产生一副点云，包含了周围环境的几何信息。但是激光雷达的limitations和复杂性：

1. 环境的多样性
2. 物体遮挡和阶段
3. 不同类别间和size区别得到的不相似表示，以及同一种类别物体，距离不同得到的不同结果
4. 表现可靠性

其余自动驾驶中常用的传感器也有各自的优势和劣势
1. Camera-based：可以提供高密度的像素信息，可以捕捉到形状和纹理信息。但是单目相机缺少深度信息，TOF和stereo cameras可以提供深度信息，但是要付出昂贵的计算成本。stereo cameras的深度信息的误差随着距离的上升指数上升，TOF的容易受到光照的影响。
2. RADAR，相比LiDAR具有更低的分辨率和更小的视野。它的特殊性使得它更适合其他任务，比如速度判定，短距离目标检测或者自动停车。

![](/assets/img/20210526/PCReviewT1.png)

点云处理的难点：

1. 点云数据的特性，高维、系数、非结构化的数据；
2. 高性能要求，自动驾驶需要从点云中提取特征并进行检测和分类，在实时的条件下。典型的场景是10Hz；
3. 安装的限制，安装在车上的处理模块是资源有限的。

## 点云处理的Pipeline 和分类

主要的流程分为三个模块：
1. **Data Representation**：Voxels,Frustums，Pillars，2Dprojection or raw point cloud
2. **Feature Extraction**: 见图
3. **Detection Network modules**：见图
4. **Predictions Refinement Network**:见图

![](/assets/img/20210526/PCReviewF1.png)

![](/assets/img/20210526/PCReviewF2.png)

本文的目录结构也按上述的分类展开。

# Data representation approaches

点云是非结构化的数据，为了应用基于CNN的目标检测方法，点云自然需要一个结构化的表示来应用卷积操作。

## Point-based

直接对点云处理来得到稀疏的表示。典型就是**PointNet**，后续被拓展到提取局部和全局特征，并启发了其他的工作，比如**Voxel Feature Extractor**(VoxelNet提出)，并被用于一系列文献，such as IPO, STD, PointRCNN, PointRGCN, LaserNe, PointFusion, RoarNe and PointPaiting, resort to this scheme of point cloud representation。

## Voxel-based

通过体素化减小点云的维度，节省内存，帮助特征提取网络更高效的利用一组点提取计算局部和全局（底层和高层）特征，而不是逐点计算。

## Frustum-based

将点云切割到一个Frustum里，比如Frustum PointNet,Frustum ConvNet and SIFRNet。

## Pillar-based

相当于简化的体素化，忽略了z轴，比如PointPillars。

## Projection-based

将3D投影到2D，来减小高维的计算量。有多种投影方式，比如**front biew(FV), range view(RV), bird's eye vies(BEV)**。他们分别相当于沿着z轴（前），x轴（右）和y轴（上）进行压缩。

其中**BEV应该是最适合自动驾驶的**，因为感兴趣目标都在同一个地面上，具有比较小的方差，所以这也是使用的**最广泛的方法**。相比FV具有一些优点，比如解决了遮挡问题，并且可以保持物体本身的物理尺寸，具有较小的方差，这都是FV所不具备的能力。

# Data feature extraction methods

特征提取可以分为三个阶段
1. Local features，也就是low-level features，具有丰富的局部信息
2. Global features，也就是high-level features，编码了点及其邻域点的几何特征
3. Contextual feature, 在提取的最后阶段被提取出来，被期望具有丰富的定位和语义信息，并被提供给模型的final tasks。
   
## point-wise

PointNet。为了提供permutation invariance（排列不变形）,使用了对称函数来提取和转换特征，比如maxpooling 和 avgpooling，缺点在于不能捕捉局部结构信息（邻域点之间）。
![](/assets/img/20210526/PCReviewF3.png)
Pointnet++，为了克服PointNet的缺点而提出，通过**Set Abstraction layer**解决。同样也被用于calssification, region proposal 和 segmentation。
## segment-wise

先将点云分割成若干个空间尺度的scenes，然后再提取特征，典型的就是**Voxel Feature Extractor**。和点特征提取不同，这种特征往往直接用于volumetric representation of the point cloud，比如voxels,pillars or frustums。使用这种特征提取的算法有VoxelNet, Second, Voxel-FPN, and HVNet.

虽然segment-wise和point-wise关联性很强，**但是segment-wise的特征效率更高，更鲁邦（因为用了一组点而不是一个），允许提取更复杂的3D局部形状信息**。

但是为了解决点云在体素间分布不均匀的问题，算法中对体素中的点随机采样，并限制了体素中点的个数。这样就会导致点多的体素丢弃一些点，**造成信息损失，是的网络预测的结果可能不稳定**，点少的体素又需要0填充，**增加了计算量和计算资源的需求**。体素的大小也需要平衡，**大的体素增加推断速度，而小的可以提供更丰富的信息**，因此，VEFs也被应用到提取单size voxel特征以及多size voxel中。

![](/assets/img/20210526/PCReviewF4.png)
## Object-wise
利用2D检测器来过滤点云。通过2D BBox 检测得到物体，然后转换为一个3D BBox，减小了空间感兴趣区域的搜索范围和需要处理的点数量。常见于LiDAR与其他传感器的结合方法中。
## Convolution neural networks

其中有些方法是用于处理2D检测的，然后被转移到处理点云上，有些是单独为3D空间设计的。

### 2D backbone

一些传统的，常用的，比如ResNet，DetNet,Hourglass Network等等。

### 3D backbone

直接使用3D卷积，但是这并不划算，因为复杂度高，而且体素特征一般是稀疏的，对于这一点应该加以利用。

### Sparse convolutional networks

当输入点有activate的点时，才计算该点的卷积。但是这样会逐层减小数据稀疏的程度，因为一个activate的点总是会使得多个输出activate，造成dilation。这使得其在深度网络的应用效果一般。为了解决这个问题，提出了Vaild Sparse Convolution ，也就是Submanifold Sparse Convolution。其对比如下图所示
，其只计算activate点对应输出位置的点的响应。


![](/assets/img/20210526/PCReviewF5.png)

### CNN networks based on voting scheme

另一种稀疏卷积的方法，不详细介绍，同样会导致dilation。

![](/assets/img/20210526/PCReviewF6.png)

### Graph convolution network 

图神经网络可以引入一些特征，比如
1. residual skip connections
2. dynamic receptive fields
3. dilatation
   
但是由于GCN的高计算和内存负担，但自动驾驶又需要实时性，因此GCNs只用在网络流程的第四阶段，也就是refine中，因为一般此时输入的点数远少于原始输入，此时只包含proposal出来的点。

用于自动驾驶的GCNs算法包括 **EdgeConV,MRGCN**,其余还有**PointRGCN**等。

## Feature extraction paradigms in 3D object detection models

这部分和通用目标检测没啥区别，主要将几种常见的提取特征的网络结构。

![](/assets/img/20210526/PCReviewF7.png)

上述若干结构的问题在于会损失信息，由于不断降采样，导致位置信息误差的增大。受到FPN的启发，出现了很多top-down结构的特征提取网络可以弥补这一点。底层的位置信息可以直接通过lateral connection连接到top-down maps上。

![](/assets/img/20210526/PCReviewF8.png)


# Detection and prediction refinement networks

检测是上述总流程的第三步。可以有很多分类方法，比如结构、物体在点云中的定位表示方法、产生rpoposal的方法、Mutil-scale feature、是否refine等等。如下图所示
![](/assets/img/20210526/PCReviewF9.png)


## Detector network architecture

从2D目标检测出发，检测器可以分为*dual-stage detectors and single-stage detector*。这没啥好说的。其结构大致如下
![](/assets/img/20210526/PCReviewF10.png)
## Detector settings

使用长方体或者segmentation-masks来做检测。

第一种就是传统的使用预定义anchors等方向来直接回归。

第二种是先预测一个mask，之后再通过pixel-wise mask来做Objects segmented。但是在视觉任务中一班选择用mask来回归其附近的BBox。

检测编码一遍对BBox-level的定位做编码，需要输入特征图，并且依赖BBox的标注信息来训练网络的回归能力。

Mask level 的同样需要mask的标注，在3D数据集中，这可以直接通过3D BBox获得（因为目标没有重叠）。为了达到检测目的，往往还需要加一个satge在分割之后做回归得到BBox。


本文总结了多个经典网络的特征提取方法，这个个人认为还是很具有参考性的。

![](/assets/img/20210526/PCReviewT2.png)

## Detector module techniques

### Region proposal-based frameworks

首先介绍了经典的几个R-CNN，这部分略了。传统的Faster R-CNN使用2k个proposal，而18年的AVOD用了80k个proposal，这体现了3D巨大的搜多空间对模型运行时间的负面影响。

之后介绍了诸多使用RPN做检测的3D目标检测算法，这在3D目标检测中应用非常广泛，在此不一一赘述。

### Silding window-based

主要是Vote3D,比较原始。

### Anchorless detectors

已经被提出的方法中包括**HotSpotNet,PointRCNN,PIXOR,VoteNet,SGPN and 3D-BoNET**，它们可以提供更高的性能表现，当点云物体收到occlusion or truncation。这些方法不使用anchor-based regression，取而代之的是每个点都对3D几何重建做出自己的贡献。

anchor-based 的方法缺点在于正负样本的不平衡，而Anchorless则可以克服这一点。

Point-RCNN展示anchor-free的solution依旧可以使用一套类似于RPN的框架。其同时进行点云分割和边界框的回归。其分割的任务就使得网络必须去捕捉点云的contextual information。

PointRGCN和PointRCNN非常相似，只是回归头用了基于GCN的方法。

Point A^2 也和PointRCNN很相似，但是体素的回归框具有更高的召回率，所以用其代替了PointNet。

### Hybrid detectors

STD，PVRCNN，同时利用点特征和体素特征，两者完全是相反的。

## Prediction refinement network

Dual-stage object detection models, such as Patch, FastPoint-RCNN, PV-RCNN, Part-A2 net, STD,PointRCNN, and PointRGCN。

传统的两阶段方法都使用特征提取器得到的特征来给出Proposal，但是由于连续的卷积和下采样使得原本存在于点云中的准确的定位信息小时了，而那个是保证目标最佳定位的基础。**从2018年以来的工作一般都会考虑全部的信息，来自于first stage(语义特征)以及原始点云信息（这样原始的位置信息就可以被保留并且增强）。最终得到更好的定位效果。**

**Refinement process大致可以分为四步**
1. RoI 的内点呗随机的采样，然后转换到canonical coordinates 来保证旋转和平移不变形以及适当的local spatial feature learning；
2. 引入额外的信息，比如点的到传感器的距离、反射强度、语义分割mask、cononical坐标等等，这些参数用来弥补远处的物体的点远少于近处的物体所带来的信息损失；
3. 上述的global和local feature concatenated到一起，输入contextual feature network，比如PointNet,PointNet++,GCN-based or VFE等等；
4. 判别和回归。

**实验表明，提取特征的时候包含ROI附近更大范围内的特征是必要的，对于结果又显著的提高。但是不是越大的距离代表越好的性能，实验表明最高的性能在margin of 1 meter处。**

总之，Refinement subnetwork主要是为了在更小的损失下捕捉contextual information，这可以通过结合local 和global features来实现。

![](/assets/img/20210526/PCReviewF11.png)

![](/assets/img/20210526/PCReviewT3.png)

# Methodologies for the development of an objext detection model

## benchmarks

对自动驾驶数据集，关心的是stereo, optical flow, visual odometry/SLAM, 3D object detection and 3D tracking。一般通过在一个或者多个区域的交通密集区驾驶采集得到。

自动驾驶数据集主要有**Waymo,A2D2,KITTI,unScenes**，其中，典型的是10Hz采样。Waymo和unScenes提供2Hz和20Hz采样。**Waymo 具有最多的标注帧和标注目标，A2D2和KITTI则少一些。**。

共同的局限是不同class的目标数量不平衡。

类别数：
1. KITTI : 3类
2. unScene : 23类
3. Waymo : 4类
4. A2D2 ： 14类

他们的对比如下
![](/assets/img/20210526/PCReviewT4.png)

## Learning strategies

per-object transformation，可能引入BBox之间的重叠，因此需要进行碰撞检测。

数据增强可以分为两种策略：
1. 实时增强，数据无需被保存在磁盘上。
2. 额外的GT BBox中的点，从一个点云中提取然后放到另一个点云里去。
   
对GT实例的transformations主要包括
1. 沿xyz轴平移
2. 绕z轴旋转
3. 实例缩放

对全局的transformations主要包括
1. 沿x轴镜像
2. 沿z轴旋转
3. 缩放
4. 全局xyz平移

为了避免一些类别中的GT数过少的问题，增强方法有
1. GT-AUG, 已经被广泛使用。首先建立一个类别实例的database。然后从中随机采样并插到别的点云中去。
2. 复制类别少的samples,根据其和别的类别的比例。

## Imbanlance sampling

类别不平衡是目标检测中的大问题。

Fast RCNN中对负样本随机采样，维持正样本和负样本的固定比例以及总体2k个proposal。SSD使用hard negative sampling ，选择confidence loss最高的negative bbox for each bbox。常用的方法还有OHEM等。RetinaNet提出了focal loss，受其启发又出现了gradient harmonizing mechanism(GHM)。

## Localisation refinement

Predict Refinement Network，也就是流程中的模块4对于位置的回归具有很大帮助。很多方法选择在此阶段融合点和之前网络中提取的global features。还有一些是完全独立的，比如Patch和STD,其在这个阶段的特征完全独立于之前的阶段。另一种强化位置回归的方法是增强RPN的判别能力。


## Evaluation metrics

自动驾驶的对于有方向的目标从4个方面评判：检测，定位，分类和推理时间。

检测，指准确率和召回率；定位，指IoU(KITTI),Average Translation Error (ATE), AverageScale Error (ASE) and Average Orientation error (AOE)(unScenes)。

分类：AP和mAP，通过PR曲线计算得到。

推理时间：字面意思，用ms或者Hz衡量。

# Object detection models comparision

## Fusion-based solution
主要是点云和图像相结合的方法，不太关心所以忽略。

## LiDAR-based solutions

将点云做BEV表示是合理而且高效的（对于都在同一个地面上的物体而言），一般使得算法具有较高的效率。但是精度一般不如Volumetric和Point data表示的方法。

Anchorless的方法相比rpn-based的需要的内存更低，因为不用定义大量的先验框，尤其是做多类别检测的时候。Anchorless的方法一般预测点到目标中心的偏移，对大目标情况下，Anchorless的效果一般弱于rpn-based的，比如Car，因为偏移更大，不好学习。但是在小目标上，在Patr A2 Net中，在行人和骑车人目标上达到了和rpn-based类似的效果。

大部分方法都用rpn-based，尤其是基于体素的；而有一些基于点或者投影的方法一般使用Anchorless方法。
![](/assets/img/20210526/PCReviewT7.png)

![](/assets/img/20210526/PCReviewT8.png)


# Research challenges and opportunities

## On the sparsity of the data and on the extraction of features

由于点云的稀疏性其不好利用传统的卷积，因此现在Backbone一般用submanifold and spatially sparse convolutions。专注点一般都在网络结构整体上，而很少有新的Backbone提出。 Graph CNNS体现了一种新的思路，voting CNN-based solution没有引起太多的关注，即使它的效率更高。

近年来出现了一些新的卷积方法。Depthwise Separation Convolutional Kernels; Mix Convs; Switchable Atrous Convolutions;Recurrent Feature Pyramid Networks
等等。

另一个就是Attention，将CNN基础的Backbones和attention结合或者代替。比如Attentional PointNet。
## On the data representation 
由于数据是稀疏的，所以可以思考改变表达方式来提高效率，比如voxels,pillars等等。

## On the occlusion and truncation issues

物体有可能被别的物体完全遮挡或者遮挡一部分。融合方法可以利用RGB数据来缓解这个问题，但是LiDAR-based方法讨论这个的不多。HotSpotNet对这个进行了一些讨论。


## On model training

LiDAR的数据总体还不是很多，数据增强或者迁移学习可以帮助克服这一点。一些新的数据增强的思路已经呗提出，但是还没有用到点云上。

还有一些提升数据效率的方法，比如Active learning等等。但是这些方法在激光雷达点云和自动驾驶数据上的应用还需要进一步验证。

## On the evolution of multimodal perception

多模态感知，结合RGB信息。主要问题是多传感器的校准和同步。
## On the possibilities for motion information integration 
通过动作（或者跟踪）信息来进一步加速模型并增强模型的精度和鲁棒性。SLAM可能会有所帮助。

## On the transparency and explainability of the models
自动驾驶会带来法律和伦理学的问题，以及安全的问题。因此可解释性和内部的判别机理是值得研究的。