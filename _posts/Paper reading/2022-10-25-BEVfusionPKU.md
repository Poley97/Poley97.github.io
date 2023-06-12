---
layout: post
title: 'BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221025/BEVFusionPKU.jpg'
tags: 论文阅读
---
本文是现在很火的关于BEV做多传感器感知融合的论文。目前有两篇均称为BEVFusion的工作，这篇是北大产出的，还有一篇的MIT产出的。相比MIT那篇，这篇其实更关注与多模态融合的可插拔特性。 


现有融合方法一般以lidar为主去query图像特征。但是鲁棒性不好，当Lidar失效，整个pipline就不能正常工作（比如很多方法使用lidar投影到image上索引对应的图像特征作为Lidar特征的扩充，即还是以lidar为主。此时lidar失效当导致整个Pipeline的失效）。本文主要针对鲁棒性进行了改进，使得融合方法对于lidar malfunction的situation更加robust。

本文相当于引入了可插拔的理念。即Fusion方法不应该因为一个模态的缺失而完全失效。如果用lidar去query image feature的话，实际上是一种不可插拔的设计。本文和一些先前工作的结构对比如下

![](/assets/img/20221025/BEVFusionPKUF1.jpg)

本文具体的网络结构如下

![](/assets/img/20221025/BEVFusionPKUF2.jpg)

其实和MIT的BEVFusion没什么区别。同样是使用一个LLS的backbone进行Multi-view image向BEV的投影，然后和3D backbone压缩得到的BEV特征进行一个element wise level的融合。只不过这里重新设计了一个fusion的模块，如下所示，比较简单。

![](/assets/img/20221025/BEVFusionPKUF3.jpg)

同时，在图像feature的提取上，这里设计了一个Adaptive Module，用于更灵活的提取图像的多尺度特征。如下所示

![](/assets/img/20221025/BEVFusionPKUF5.jpg)

总的来说，这个网络相比其他的融合方法，相当于把更多的重心放到了camera的modual而不是Lidar。从而提高了在lidar失效情况下的检测鲁棒性。但是实际上这里的实验结果并没有完全验证这一点。如下表所示

![](/assets/img/20221025/BEVFusionPKUT3.jpg)

这里作者只屏蔽了一部分FOV的Lidar而不是全部屏蔽，这样造成的性能下降是可以预料的。但是作者也没有进一步的去分析，屏蔽完FOV后，在无Lidar的fov区域的具体检测性能。因此，此表格仍然不能准确衡量在lidar mulfunction区域的具体检测性能。