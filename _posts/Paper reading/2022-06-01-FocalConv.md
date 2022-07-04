---
layout: post
title: 'Focal Sparse Convolutional Networks for 3D Object Detection'
date: 2022-5-30
author: Poley
cover: '/assets/img/20220530/FocalConv.png'
tags: 论文阅读
---


传统稀疏卷积减小稀疏性并且模糊特征的判别性。而子流型卷积因为限制了输出而影响了必要的信息流，尤其是空间上不相连的体素。这导致所有的体素并不平等。这点和笔者自己论文的思路非常近似。但是笔者论文得解决方法是通过引入动态感受野来使得体素之间更加“平等”，而这片文章作者则提出将注意力更多的关注到重要的体素上。

![](/assets/img/20220530/FocalConvF1.png)

相对于子流型卷积和普通稀疏卷积，本文相当于是对两者进行了一个折中，形成了一种通用表达。即对部分包含重要语义信息的体素进行dilate，而对大部分无用体素保持submainfold。为此，作者引入了一个cubic importance来对体素的重要性进行判断。

对于普通卷积，其可以表示为
$$
\begin{equation}
\mathrm{y}_{p}=\sum_{k \in K^{d}} \mathrm{w}_{k} \cdot \mathrm{x}_{\bar{p}_{k}}
\end{equation}
$$
而对于稀疏卷积则如下
$$
\begin{equation}
\mathrm{y}_{p \in P_{\mathrm{out}}}=\sum_{k \in K^{d}\left(p, P_{\mathrm{in}}\right)} \mathrm{W}_{k} \cdot \mathrm{X}_{\bar{p}_{k}}
\end{equation}
$$
其中
$$
\begin{equation}
K^{d}\left(p, P_{\mathrm{in}}\right)=\left\{k \mid p+k \in P_{\mathrm{in}}, k \in K^{d}\right\}
\end{equation}
$$
$$
\begin{equation}
P_{\mathrm{out}}=\bigcup_{p \in P_{\mathrm{in}}} P\left(p, K^{d}\right)
\end{equation}
$$
$$
\begin{equation}
P\left(p, K^{d}\right)=\left\{p+k \mid k \in K^{d}\right\}
\end{equation}
$$
当$P_out=P_in$时，上式变为子流型卷积。

本文将上述过程一般化，变为
$$
\begin{equation}
P_{\mathrm{out}}=\left(\bigcup_{p \in P_{\mathrm{im}}} P\left(p, K_{\mathrm{im}}^{d}(p)\right)\right) \cup P_{\mathrm{in} / \mathrm{im}}
\end{equation}
$$
其中im表示重要体素，即具有丰富信息的（前景）体素。而这里的卷积主要分为三步
+ cubic importance prediction
+ important input selection,
+ dynamic output shape generation.

通俗的说，即先预测每个体素自身和邻域的重要性。选择自身重要性高的体素作为important input，并对这些体素进行dilate。而dilate的形状不仅取决于卷积核，也取决于这些important input为中心预测的cubic importance prediction的阈值，即选择预测重要性高的邻域作为dialta的形状。上述过程可以表达为

重要性选取
$$
\begin{equation}
P_{\mathrm{im}}=\left\{p \mid I_{0}^{p} \geq \tau, p \in P_{\mathrm{in}}\right\}
\end{equation}
$$

$$
\begin{equation}
K_{\mathrm{im}}^{d}(p)=\left\{k \mid p+k \in P_{\mathrm{in}}, I_{k}^{p} \geq \tau, k \in K^{d}\right\}
\end{equation}
$$

这样确定了$P_out$的active voxel之后，再从$P_in$的对应位置卷积即可。在实际网络结构中，作者进一步引入了图像信息来辅助进行重要性预测和dilation shape的预测，进一步提高了网络的性能，如下图所示

![](/assets/img/20220530/FocalConvF3.png)

总的来说，本文的创新在于使用一个轻量化卷积来在传统稀疏卷积dialtion引入过多不必要计算量和子流型卷积切断必要的信息流之间做了一个trade-off，并取得了不错的效果。