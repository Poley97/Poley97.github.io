---
layout: post
title: 'On Tht Relationship Between Self-Attention And Cconvolutional Layers'
date: 2021-12-31
author: Poley
cover: '/assets/img/20211228/TandC.png'
tags: 论文阅读
---

这篇论文分析了Transformer和Convolution的关系，得出结论，2D Transformer可以模拟任何2D Convlution的情况，并给出了理论和实验证明。论文略微有点复杂，笔者没有非常仔细的阅读，而是了解了一下大致概念和思路。

Transformer最大的优势和不同在于其可以同时接触到输入序列的所有部分（大感受野，long-range denpendencies）。如今Transformer在图像处理上也达到了很好的性能。于是问题来了，自注意力层处理图像的方式是否类似于卷积？作者给出的结论是:
**从理论和实践上证明self-attention可以被表示为任何卷积，并且具有类似的处理方式。并且，实验表明局部的卷积可能是对于图像处理网络开始若干层的正确归纳偏置。**

首先阐述自注意力和卷积的数学形式

MultiHead Self-Attention Layer可以表示为如下形式（1D）
$$
\begin{equation}
\text { Self-Attention }(\boldsymbol{X})_{t,:}:=\operatorname{softmax}\left(\boldsymbol{A}_{t,:}\right) \boldsymbol{X} \boldsymbol{W}_{v a l}
\end{equation}
$$
多头形式如下
$$
\begin{equation}
\operatorname{MHSA}(\boldsymbol{X}):=\underset{h \in\left[N_{h}\right]}{\operatorname{concat}}\left[\text { Self-Attention }_{h}(\boldsymbol{X})\right] \boldsymbol{W}_{\text {out }}+b_{\text {out }}
\end{equation}
$$
而2D卷积可以表示为如下
$$
\begin{equation}
\operatorname{Conv}(\boldsymbol{X})_{i, j,:}:=\sum_{\left(\delta_{1}, \delta_{2}\right) \in \mathbb{\Delta}_{K}} \mathbf{X}_{i+\delta_{1}, j+\delta_{2},:} \mathbf{W}_{\delta_{1}, \delta_{2},,,:}+\boldsymbol{b}
\end{equation}
$$
其中
$$
\begin{equation}
\Delta_{K}:=\left[-\left\lfloor\frac{K}{2}\right\rfloor, \cdots,\left\lfloor\frac{K}{2}\right\rfloor\right] \times\left[-\left\lfloor\frac{K}{2}\right\rfloor, \cdots,\left\lfloor\frac{K}{2}\right\rfloor\right]
\end{equation}
$$
代表卷积核的相对位移。

为了和1D的Self-Attention保持一致，这里将2D 自注意力的脚标变为二维元组$p=(i,j)$，则其可以表示为如下
$$
\begin{equation}
\text { Self-Attention }(\boldsymbol{X})_{\boldsymbol{q},:}=\sum_{\boldsymbol{k}} \operatorname{softmax}\left(\mathbf{A}_{\boldsymbol{q},:}\right)_{\boldsymbol{k}} \mathbf{X}_{\boldsymbol{k},:} \boldsymbol{W}_{\text {val }}
\end{equation}
$$
多头形式与1D同理。

为了解决Transformer的顺序不变形问题，一般通过引入位置编码来解决，一种相对位置编码方式如下
$$
\begin{equation}
\mathbf{A}_{q, k}^{\mathrm{rel}}:=\mathbf{X}_{q,:}^{\top} \boldsymbol{W}_{q r y}^{\top} \boldsymbol{W}_{k e y} \mathbf{X}_{k,:}+\mathbf{X}_{q,:}^{\top} \boldsymbol{W}_{q r y}^{\top} \widehat{\boldsymbol{W}}_{k e y} r_{\delta}+\boldsymbol{u}^{\top} \boldsymbol{W}_{k e y} \mathbf{X}_{k,:}+\boldsymbol{v}^{\top} \widehat{\boldsymbol{W}}_{k e y} r_{\delta}
\end{equation}
$$

在这种位置编码下，可以证明
![](/assets/img/20211228/TandCL1.png)

其原理如下图所示
![](/assets/img/20211228/TandCF1.png)

这样，多头注意力机制可以简化为如下的形式
$$
\begin{equation}
\operatorname{MHSA}(\boldsymbol{X})=b_{\text {out }}+\sum_{h \in\left[N_{h}\right]} \operatorname{softmax}\left(A^{(h)}\right) \boldsymbol{X} \underbrace{\boldsymbol{W}_{\text {val }}^{(h)} \boldsymbol{W}_{\text {out }}\left[(h-1) D_{h}+1: h D_{h}+1\right]}_{\boldsymbol{W}^{(h)}}
\end{equation}
$$

注意到两个矩阵相乘的情况。在满足rank条件的情况下，即
Note that each head's value matrix $\boldsymbol{W}_{\text {val }}^{(h)} \in \mathbb{R}^{D_{\text {in }} \times D_{h}}$ and each block of the projection matrix $\boldsymbol{W}_{\text {out }}$ of dimension $D_{h} \times D_{\text {out }}$ are learned. Assuming that $D_{h} \geq D_{\text {out }}$, we can replace each pair of matrices by a learned matrix $\boldsymbol{W}^{(h)}$ for each head. We consider one output pixel of the multi-head self-attention:
$$
\operatorname{MHSA}(\boldsymbol{X})_{\boldsymbol{q},:}=\sum_{h \in\left[N_{h}\right]}\left(\sum_{\boldsymbol{k}} \operatorname{softmax}\left(\mathbf{A}_{\boldsymbol{q},:}^{(h)}\right)_{\boldsymbol{k}} \mathrm{X}_{\boldsymbol{k},:}\right) \boldsymbol{W}^{(h)}+b_{\text {out }}
$$

通过引入上述认为设定的自注意力矩阵，可以得到
$$
\begin{equation}
\operatorname{MHSA}(\mathbf{X})_{q}=\sum_{h \in\left[N_{h}\right]} \mathbf{X}_{q-\boldsymbol{f}(h),:} \boldsymbol{W}^{(h)}+b_{\text {out }}
\end{equation}
$$

此时可以发现，多头注意力具有了和卷积一样的数学形式，即可以在对卷积进行模拟。为了达到上述特殊的注意力矩阵的条件，这里作者通过证明得到其对应的注意力参数和位置编码如下（证明略）
$$
v^{(h)}:=-\alpha^{(h)}\left(1,-2 \Delta_{1}^{(h)},-2 \Delta_{2}^{(h)}\right) \quad r_{\delta}:=\left(\|\delta\|^{2}, \delta_{1}, \delta_{2}\right) \quad W_{q r y}=W_{k e y}:=0 \quad \widehat{W_{k e y}}:=I
$$
The learned parameters $\Delta^{(h)}=\left(\Delta_{1}^{(h)}, \Delta_{2}^{(h)}\right)$ and $\alpha^{(h)}$ determine the center and width of attention of each head, respectively. On the other hand, $\delta=\left(\delta_{1}, \delta_{2}\right)$ is fixed and expresses the relative shift between query and key pixels.

并且对于卷积的Padding， Stride和Dilation参数，同样可以通过如下方式进行模拟
- Padding: a multi-head self-attention layer uses by default the "SAME" padding while a convolutional layer would decrease the image size by $K-1$ pixels. The correct way to alleviate these boundary effects is to pad the input image with $\lfloor K / 2\rfloor$ zeros on each side. In this case, the cropped output of a MHSA and a convolutional layer are the same.
- Stride: a strided convolution can be seen as a convolution followed by a fixed pooling operation-with computational optimizations. Theorem 1 is defined for stride 1 , but a fixed pooling layer could be appended to the Self-Attention layer to simulate any stride.
- Dilation: a multi-head self-attention layer can express any dilated convolution as each head can attend a value at any pixel shift and form a (dilated) grid pattern.


在实验部分，作者发现在多注意力的前几层，及时没有使用随机初始化的可学习位置编码，在前几层注意力模块上，注意力模块仍然会倾向于收集local上信息，这说明卷积所引入的聚合Local信息的归纳偏置在图像分类任务的浅层上是非常正确的。作者在实验中可视化了attention head的中心，如下图所示。可以发现在浅层 attention基本在收集Local特征，而到高层才开始手机远距离的long-range dependencies。

![](/assets/img/20211228/TandCF4.png)

整体论文比较复杂，由于时间有限，笔者值大概浏览其关键证明和概念。实验部分没有仔细浏览。