---
layout: post
title: 'Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph'
date: 2022-10-12
author: Poley
cover: '/assets/img/20221008/GR.jpg'
tags: 论文阅读
---

作者认为，现有的二阶段模型一般通过grid或者keypoint来提取roi特征。对于不均匀且稀疏的户外点云并不高效。具体来说，这三种方法的代表性工作为：
1. PointRCNN,提取proposal附近的关键点，并提取其特征；
2. Part $A^2$，将proposal划分为regular voxel grid进行后续处理；
3. PV-RCNN，在proposal内生成均匀的grid point来提取特征。
作者认为，现有方法存在三个问题：
1. 忽略了点在proposal中的非均匀分布；
2. 点的内部关系没有被充分利用；
3. 稀疏点云只能提供有限的注意信息；

作者提出三个创新：
1. 动态点聚合，通过patch search快速搜索局部区域内的点，并且体素大小根据距离而调整，以适应非均匀分布；
2. RoI-graph Pooling：通过建立局部图来更好的建模上下文信息和挖掘point relations
3. Visual Features Augmentation： 通过fusion策略来补偿具有有限语义线索的lidar point。

所提出的网络结构整体如下图所示
![](/assets/img/20221008/GRF2.jpg)

## Dynamic Point Aggregation
![](/assets/img/20221008/GRF3.jpg)
具体来说，在采样方法上，传统方法遍历所有点决定其是否在proposal里。复杂度O（MN），对于大场景（比如waymo）很不友好。这里直接通过proposal占用的patch来实现快速采样。得到proposal的点后，将其划分为均匀的体素网格，使用最远体素采样。保留被采样体素中的点。考虑到点云的不均匀性，使用根据距离变化的体素尺寸，对于近处的目标，体素大；对于远处的，体素小，尽可能保留完整的信息。

这部分具体分为两部分，patch search和dynamic farthest voxel sampling;

**patch search**主要是为了降低proposal内点的搜索成本（复杂度O（MN），M为proposal的数量，N为总点数）。本方法通过将空间划分为若干个patch，并维护patch与point和box之间的索引，并通过索引实现并行的快速采样。为了简化计算，这里也将proposal简化为了axis aligned的最大内切矩形，如上图所示。通过这种方法是，提取proposal内点的操作计算复杂度可以下降至O（QK），其中Q是所有被占用patch包含的点数，K是每个patch的最大被占用次数（可能被多个proposal）占用。

**dynamic farthest voxel sampling**则是为了解决上述提到的roi内点分布不均匀的问题。由于roi中大部分位置没有点，这些均匀分布的Grid采样可能会造成浪费。因此这里提出使用voxel sampling来平衡效率与性能。即将proposal内的空间划分成若干voxel，使用non-empty voxel中的随机一点表示该体素的位置，并进行fps采样，以保证物体的各个part信息都得以保留，同时复杂度又得以大大降低。注意到voxel会导致采样点的密度降低，这里作者提出根据距离调整体素大小，远小近大，以尽可能完整的保留远处物体的点云而不造成不必要的信息损失。体素的尺寸和距离成负相关，作者提出的计算方式如下
$$
\begin{equation}
V_i=\lambda \cdot e^{-\frac{\sqrt{x_i^2+y_i^2+z_i^2}}{\delta}}
\end{equation}
$$
整个采样过程中可以使用GPU并行，但是建立体素索引时，体素太稀疏会导致空索引太多，在并行的时候需要对齐voxels的数量，造成不必要的内存浪费。这里使用哈希表，并使用二次探查法来解决地址冲突。

## RoI-graph Pooling
上述得到的采样点数少，为了避免信息损失，因此首先聚合采样点周围的邻域。也就是说，这个graph实际上是local信息之间的graph。local信息使用radius近邻的pointnet实现聚合。注意到，因为每个proposal中的点已经被提前划分出来，所以radius query的搜索空间很小，进而实现很小的计算代价。

一个node（即一个采样点与其对应的local）可能被多个proposal使用（为什么），导致ambiguous。因此这里添加了proposal的局部角点来差异化不同proposal里的node特征。作者认为两个角点是足够的。相当于一种位置编码吧个人觉得。


提取了采样点之后，为了弥补信息损失，先用PointNet做一个局部信息聚合。采样点之间建立knn graph来提取特征。最后，融合多个level的node特征作为robust的roi feature。

综上，一个node的初始特征可以表示为
$$
\begin{equation}
s_j^0=\left[x_j, y_j, z_j, r_j, f_j, u_j, w_j\right] \text {, }
\end{equation}
$$

之后，通过KNN近邻搜索建立graph，使用Edgeconv提出的聚合方法进行聚合（就是DGCNN）。如下
$$
\begin{equation}
s_j^{t+1}=\max _{v_k \in \mathcal{N}_{v_j}} \phi_\theta\left(\left[s_k^t-s_j^t, s_j^t\right]\right)
\end{equation}
$$
最后，引入一个简单的attention来进行reweight。似乎这么逐级融合RoI特征具有不错的效果，类似的CasA也是逐级融合的。如下图所示
![](/assets/img/20221008/GRF5.jpg)
## Visual Features Augmentation
关于数据增广，copy and paste是常用的方法。但是本方法作用在RPN上，对其并不敏感。因此其只用于预训练RPN（感觉意思应该是用来训练一阶段网络），并在训练整个框架时disable了。
融合比较简单，通过2D卷积提取图像特征，将node投影到feature map上插值即可。这样实际上可能也有点浪费图像信息（采样率太低）。
但是总的来说，纯点云方案确实引入很多ambiguous的问题（比如几何形状相似的物体），因此引入视觉特征确实可以很好的降低FP，如下所示。

![](/assets/img/20221008/GRF1.jpg)

最终，在性能上，本方法也体现出了非常显著的优势。

![](/assets/img/20221008/GRT1.jpg)

