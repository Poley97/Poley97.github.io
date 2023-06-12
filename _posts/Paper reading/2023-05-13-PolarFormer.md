---
layout: post
title: 'PolarFormer Multi-camera 3D Object Detection with Polar Transformer'
date: 2023-5-13
author: Poley
cover: '/assets/img/20230513/PolarFormer.jpg'
tags: 论文阅读  
---
本文提出一种新的camera 3D 检测Pipeline，作者认为Polar coordinate system是更适合camera 3D检测的，因为这是camera自身的视角。但是类似的方法不多，LIDAR中探索了一些。这些方法贡献有限，因为卷积网络限制了矩形的grid structure和局部感受野。
本文首先提出学习polar ray的表征。之后将其光栅化到bev Polar表征。为了解决卷积对输入grid的要求，这里使用cross attention代替。为了解决目标尺度在polar坐标下变化剧烈的问题，引入多尺度Polar BEV表征。

首先，一般的基于camera的3D检测都需要在camera的3D 笛卡尔坐标系中进行，其从3D坐标的转换可以表示为
$$
\begin{equation}
s\left[\begin{array}{c}
x^{(I)} \\
y^{(I)} \\
1
\end{array}\right]=\left[\begin{array}{ccc}
f_x & 0 & u_0 \\
0 & f_y & v_0 \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
x^{(C)} \\
y^{(C)} \\
z^{(C)}
\end{array}\right],
\end{equation}
$$

$$
\begin{equation}
\begin{gathered}
\phi^{(P)}=\arctan \frac{x^{(C)}}{z^{(C)}}=\arctan \frac{x^{(I)}-u_0}{f_x} \\
\rho^{(P)}=\sqrt{\left(x^{(C)}\right)^2+\left(z^{(C)}\right)^2}=z^{(C)} \sqrt{\left(\frac{x^{(I)}-u_0}{f_x}\right)^2+1}
\end{gathered}
\end{equation}
$$

这里的极坐标为什么只有两个坐标？ 从方位角的角度来看，其与Z轴无关，因此可以将image的每一列与一个azimuth一一对应起来。此时每个column就是一个H*C的特征。query也按column的形式进行，并进行cross-attention交互。之后，由于在每个方向角上都有一个query，就可以将多个image的方向角特征投影到bev上，形成bev polar feature map。由于多尺度在方向角上的分辨率是一致的，因此可以直接cat。作者认为，这样的query可以隐式学习到深度信息。

上述的图像特征被编码成若干个不同azimuth的形式，之后使用预先定义好的Polar query来作为Encoder提取特征，如下所示

$$
\begin{aligned}
\mathbf{p}_{n, u, w} & =\operatorname{MultiHead}\left(\dot{\mathbf{p}}_{n, u, w}, \mathbf{f}_{n, u, w}, \mathbf{f}_{n, u, w}\right) \\
& =\text { Concat }\left(\operatorname{head}_1, \ldots, \operatorname{head}_{\mathrm{h}}\right) \mathbf{W}_u^O,
\end{aligned}
$$
where
$$
\operatorname{head}_{\mathrm{i}}=\operatorname{Attention}\left(\dot{\mathbf{p}}_{n, u, w} \mathbf{W}_{i, u}^Q, \mathbf{f}_{n, u, w} \mathbf{W}_{i, u}^K, \mathbf{f}_{n, u, w} \mathbf{W}_{i, u}^V\right) \text {, }
$$

由此，得到了定义在极坐标系下的query特征。接下来需要将其投影到统一的BEV空间，以方便融合Multi-camera进行3D检测。这里类似于PETR，在空间中撒点以柱坐标的形式撒点，并投影到极坐标中，寻找对应的polar feature。每个点对应的各视图中的Polar feature取平均得到此点的feature。感觉类似于LSS。

这个过程可以表示如下
$$
\begin{equation}
\left[\begin{array}{c}
s x_{i, j, k, n}^{(I)} \\
s y_{i, j, k, n}^{(I)} \\
s \\
1
\end{array}\right]=\left[\begin{array}{cc}
\boldsymbol{\Pi}_n & 0 \\
0 & 1
\end{array}\right] \mathbf{E}_n\left[\begin{array}{c}
\rho_i^{(P)} \sin \phi_j^{(P)} \\
\rho_i^{(P)} \cos \phi_j^{(P)} \\
z_k^{(P)} \\
1
\end{array}\right],
\end{equation}
$$
同时在平均的时候把Z轴也平均掉了，因此将其成为BEV，表示如下。但实际上他是一个Percepctive的视图。
$$
\begin{equation}
\begin{aligned}
& \mathbf{G}_u\left(\rho_i^{(P)}, \phi_j^{(P)}\right)=\frac{1}{\sum_{n=1}^N \sum_{k=1}^{\mathcal{Z}} \lambda_n\left(\rho_i^{(P)}, \phi_j^{(P)}, z_k^{(P)}\right)} \\
& \cdot \sum_{n=1}^N \sum_{k=1}^{\mathcal{Z}} \lambda_n\left(\rho_i^{(P)}, \phi_j^{(P)}, z_k^{(P)}\right) \mathcal{B}\left(\mathbf{P}_{n, u},\left(\bar{x}_{i, j, k, n}^{(I)}, \bar{r}_{i, j, n}\right)\right)
\end{aligned}
\end{equation}
$$
Percepctive的BEV视图如下所示

![](/assets/img/20230513/PolarFormerF5.jpg)

网络的整体结构如下

![](/assets/img/20230513/PolarFormerF2.jpg)

整体结构并不复杂，主要是是使用了现有的Attention模块，只是在特征的编码方式上有所变化，相对应的，在target assign上也需要变换到Polar坐标中，如下图所示

![](/assets/img/20230513/PolarFormerF3.jpg)

消融实验表明，feature和prediction以及pe都使用基于Polar的可以带来显著的增益，并且所提出的Pipeline的多尺度增益也很明显。

![](/assets/img/20230513/PolarFormerT2.jpg)