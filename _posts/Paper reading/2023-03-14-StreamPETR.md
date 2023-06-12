---
layout: post
title: 'Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection'
date: 2023-3-14
author: Poley
cover: '/assets/img/20230311/StreamPETR.jpg'
tags: 论文阅读  
---
时序融合分为两类：BEV temporal 和 perspective temporal。BEV方法从历史的feature中warp信息。但是由于其高度结构化，因此不擅长建模运动物体，这需要通过大感受野来弥补。perspective方法则是DETR-based的方法，使用query来和过去的img交互，造成了成倍的计算量。而这里提出centric-based，使用query作为物体的隐状态传递，可以有效建模运动物体。

大体结构，使用一个memory query来存储历史信息，以及一个propagation transformer用于与当前的query交互。同时提出一个motion-aware layer normalization来隐式的针对运动进行补偿。这样的优势是不需要处理dense feature map，而只使用一小部分query就可以，大大降低了计算量。

基于bev的方法具有intermediate feature，而query based的没有。本文结合了两者的优点。
相比MOTR等工作直接继承qeury不同，这里的历史query主要作为token为当前query提供信息。同时，2D检测中无需考虑运动问题，而3D中则需要考虑如何在query的隐空间中解决物体运动的问题。这几种时序方法的对比如下 

![](/assets/img/20230311/StreamPETRF1.jpg)

本文首先分析了几种时序建模方式。总的来所，时序融合一般从三个方面融合，2D特征，BEV特征和物体特征。如下所示
$$
\begin{equation}
\tilde{F}_{\text {out }}=\varphi\left(F_{2 d}, F_{\text {bev }}, F_{\text {obj }}\right)
\end{equation}
$$

基于BEV的方法一般只融合历史的BEV特征，如下所示
$$
\begin{equation}
\tilde{F}_{\text {bev }}^t=\varphi\left(F_{b e v}^{t-1}, F_{b e v}^t\right)
\end{equation}
$$
拓展到长时序一般有两种方法。一种是对齐历史帧并一起处理，如下
$$
\begin{equation}
\tilde{F}_{\text {bev }}^t=\varphi\left(F_{\text {bev }}^{t-k}, \cdots, F_{\text {bev }}^{t-1}, F_{\text {bev }}^t\right)
\end{equation}
$$
一种是用迭代的方式，用一个bev来融合之前的所有BEV状态，如下所示。
$$
\begin{equation}
\tilde{F}_{b e v}^t=\varphi\left(\tilde{F}_{b e v}^{t-1}, F_{b e v}^t\right)
\end{equation}
$$
缺点是静态的BEV忽略了物体的运动，可能造成错误定位。


基于perceprctive的方法是使用当前的query与过去的feature进行交互，即从分别从不同的图像中提取信息，如下所示。
$$
\begin{equation}
\tilde{F}_{o b j}^t=\varphi\left(F_{2 d}^{t-k}, F_{o b j}^t\right) \cdots+\varphi\left(F_{2 d}^t, F_{o b j}^t\right)
\end{equation}
$$
缺点是，随着时序变长，需要交互的feature越来越多，计算量也增大。
这里提出的Object-centric方法主要是先对历史的Object feature补偿，然后再和当前的Object融合。

这里提出的Object-centric方法主要是先对历史的Object feature补偿，然后再和当前的Object融合。首先针对目标特征的补偿
$$
\begin{equation}
\tilde{F}_{o b j}^{t-1}=\mu\left(F_{o b j}^{t-1}, M\right)
\end{equation}
$$
在将object-level的特征进行融合
$$
\begin{equation}
\tilde{F}_{o b j}^t=\varphi\left(\tilde{F}_{o b j}^{t-1}, F_{o b j}^t\right)
\end{equation}
$$

接下来是本方法的具体介绍，本方法的流程图如下所示
![](/assets/img/20230311/StreamPETRF3.jpg)

首先对历史query的ref points进行偏移。
$$
\begin{equation}
\tilde{Q}_p^t=E_{t-1}^t \cdot Q_p^{t-1}
\end{equation}
$$
同样的，也需要的query的特征进行偏移。但是这只能在隐式的高维空间中进行，因此这里采用了预测layer normalization的方式。这是收到一些task specific control生成模型的启发。

$$
\begin{equation}
\begin{aligned}
& \gamma=\xi_1\left(E_{t-1}^t, v, \triangle t\right) \\
& \beta=\xi_2\left(E_{t-1}^t, v, \triangle t\right)
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
\tilde{Q}_{p e}^t & =\gamma \cdot L N\left(\psi\left(Q_p^t\right)\right)+\beta \\
\tilde{Q}_c^t & =\gamma \cdot L N\left(Q_c^{t-1}\right)+\beta
\end{aligned}
\end{equation}
$$

分别对content和pe进行分布的重新调整。同时当前帧也和历史帧做相同操作，只不过速度和时间都为0，为了保持一致性。

最后，在性能上，确实在camera-based方法上具有较大的性能提升和优势，如下

![](/assets/img/20230311/StreamPETRT1.jpg) 