---
layout: post
title: 'TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221114/Transfusion.jpg'
tags: 论文阅读
---

本文提出仅仅通过LiDAR point和图像像素建立关联是不太鲁邦的。传统方法通过hard assosiation来建立跨模态关系，比如投影。这样对传感器和时间同步的要求都比较高，并且由于特征是concat或者sum的，因此Lidar模态的特征也很容易收到camera模态中一些恶劣条件的影响，比如bad illumination和sensor misalignment等等。这毫无疑问降低了模型的鲁棒性。总的来说，point-based融合的两大缺点：简单的加或concat会导致lidar特征被low-quality 的图像特征degrades。其次，这也浪费了很多密集的图像特征，并高度依赖于时间和空间同步的准确性。

因此，本文提出一种基于transformer的融合方法，通过soft assosiation的方法来建立跨模态特征的联系。并设计了一种image-guided的query initialization方法来使得模型更好的索引图像上的信息（因为融合方法中一般都是Lidar为主模态，因此不太需要额外的设计）。

本文设计的结构如下图所示

![](/assets/img/20221114/TransfusionF2.jpg)

首先在双模态上各自使用一个特征提取网络来独立提取各自的特征。

之后，这里为了减少query的数量，首先对BEV特征预测了一个heatmap，并选取TOPK的位置来初始化query的位置。随后使用两个cross attention模块。

关于Lidar的部分比较简单，直接使用3D位置编码做query即可。对于图像的稍微复杂点。对于3D点和多视角图像，一个3D点所对应的图像特征并不会出现在所有的图上（自动驾驶场景下，一般指出现在一个图像上），因此，这里提出，query只需要和其所在的feature map上进行attention即可。为了防止模型聚合到太多与Object无关的特征（transformer灵活的感受野允许这种情况的发生），这里作者额外设计了一个mask，根据query的位置为对应的图像特征赋予权重，如下所示。

$$
\begin{equation}
M_{i j}=\exp \left(-\frac{\left(i-c_x\right)^2+\left(j-c_y\right)^2}{\sigma r^2}\right)
\end{equation}
$$

即引入一个位置嵌入的高斯加权，使其主要关注图像中的投影位置附近的部分。

综上，一个query依次索引lidar和camera模态的特征，并最终解码，得到最终的预测。注意到，这里在两个cross-attention后都使用了预测头，这也是多层decoder的常用操作，使用辅助Loss来确保每层都具有一定的预测能力。

最终，在检测结果上，本方法也是达到了当时的SOTA,不过现在已经被更优的方法所刷新。

![](/assets/img/20221114/TransfusionT1.jpg)