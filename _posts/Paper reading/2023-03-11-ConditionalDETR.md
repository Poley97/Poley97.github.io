---
layout: post
title: 'Conditional DETR for Fast Training Convergence'
date: 2023-3-11
author: Poley
cover: '/assets/img/20230311/CDETR.jpg'
tags: 论文阅读  
---

本文发表于ICCV2021,主要创新为对detr的改进，学习conditional spatial query使得其可以快速attend到distinct region。进而显著的增加了模型收敛的速度。本工作受到的启发主要来自于，content embedding的重要作用和pos embedding的极小贡献


![](/assets/img/20230311/CDETRF1.jpg)

可以看到，DETR在50e的时候其权重突出的区域和真正的目标关联性很弱。而Conditional DETR在50e的时候，四个offset的感兴趣区域已经是BBOX四个边界的位置了，这是合理的,并且基本与DETR训练500e的感兴趣区域一致。


作者认为，DETR 收敛慢有两个原因：1、query并没有探索关于特定图像的信息
2、短时间的训练，query无法同时匹配空间key和内容key，这增加了对高质量content embeeding的依赖，加大了训练难度。

而这里提出的conditional query预测方法是将用于回归的信息映射到embedding空间进行预测，和2d坐标映射的空间一致。其示意图如下所示

![](/assets/img/20230311/CDETRF3.jpg)

原版DETR的回归可以表示为如下
$$
\begin{equation}
\mathbf{b}=\operatorname{sigmoid}\left(\operatorname{FFN}(\mathbf{f})+\left[\begin{array}{lll}
\mathbf{s}^{\top} & 0 & 0
\end{array}\right]^{\top}\right)
\end{equation}
$$
原版DETR的query的refpoints $s$是0 0，这里提出两种预案：1、作为参数学习，2、从对应的object query产生（预测）

这里首先分析了DETR的cross attention。其中具有两个emb，分别是来自decoder self attention的content embedding和表达位置的pos embedding。两者一般相加作为qeury的embedding，此时其相似度计算如下

$$
\begin{equation}
\begin{aligned}
& \left(\mathbf{c}_q+\mathbf{p}_q\right)^{\top}\left(\mathbf{c}_k+\mathbf{p}_k\right) \\
= & \mathbf{c}_q^{\top} \mathbf{c}_k+\mathbf{c}_q^{\top} \mathbf{p}_k+\mathbf{p}_q^{\top} \mathbf{c}_k+\mathbf{p}_q^{\top} \mathbf{p}_k \\
= & \mathbf{c}_q^{\top} \mathbf{c}_k+\mathbf{c}_q^{\top} \mathbf{p}_k+\mathbf{o}_q^{\top} \mathbf{c}_k+\mathbf{o}_q^{\top} \mathbf{p}_k .
\end{aligned}
\end{equation}
$$

可以看到，其中除了content和pos与其各自的相似度。作者认为这样实际上使得embedding需要兼顾两边的交互，这对于content embeddinig提出了更高质量的需求，这可能是导致detr收敛慢的原因（因为2d pos embedding相对固定，因此只能是content embedding去学习的更好）。

这里作者首先拆分了ca的embeeding,去除了content和position的相互作用。作者发现ca的关键在于decoder embedding和ref point。因此想办法将其映射到同一空间中组成query。如下所示
$$
\begin{equation}
\mathbf{c}_q^{\top} \mathbf{c}_k+\mathbf{p}_q^{\top} \mathbf{p}_k
\end{equation}
$$

进一步的，DETR的回归分为两部分：1、 根据decoder embedding回归ref point的offset，非归一化；2、归一化到0-1。因此，ref和emb对于detr的感兴趣区域是由这两者共同决定的。因此，这里选择根据这两者预测一个query出来。如下所示


$$
\begin{equation}
(\mathbf{s}, \mathbf{f}) \rightarrow \mathbf{p}_q
\end{equation}
$$

首先，直观的，根据ref points生成query是合理的。
$$
\begin{equation}
\mathbf{p}_s=\operatorname{sinusoidal}(\operatorname{sigmoid}(\mathbf{s}))
\end{equation}
$$
为了和content进行强关联，这里设计通过content预测一个矩阵来对ref points的项链进行变换，为了简化，这里预测的矩阵简化为对角矩阵，即变成两个向量的点乘。
$$
\begin{equation}
\mathbf{p}_q=\mathbf{T} \mathbf{p}_s=\lambda_q \odot \mathbf{p}_s
\end{equation}
$$

进一步的，作者还对attention注意力中的空间注意力和内容注意力进行了可视化。
空间注意力更关注物体的边缘，而内容注意力更关注物体本身。两者结合可以屏蔽不关键的部位，准确的获取到物体本身的Infomation。如下图所示

![](/assets/img/20230311/CDETRF4.jpg)

使用conditional spatial query的效果：1、将感兴趣区域转移到object box内，2；自适应的缩放highlings的区域大小，目标大则大，小目标则小。这些实际上都是通过利用content来对refpoint做变换而实现的。

其性能和收敛性相比DETR都有很大的提升

![](/assets/img/20230311/CDETRT1.jpg)