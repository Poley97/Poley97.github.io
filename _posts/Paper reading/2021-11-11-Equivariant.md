---
layout: post
title: 'Equivariant Point Network for 3D Point Cloud Analysis'
date: 2021-11-11
author: Poley
cover: '/assets/img/20211110/Equivariant.png'
tags: 论文阅读
---

本文提出一种可分离点卷积（separable point convolution）产生等变特征，以及一个attention layer来利用这些等变特征产生不变特征。使得网络可以根据下游任务灵活的提取点云特征



## 相关知识：等变与不变（Equivariant and Invariant）
 > 参考博客： https://zhuanlan.zhihu.com/p/41682204

 + 等变性 equivariant：对于一个函数/特征$f$以及一个变换$g$， 如果我们有：$f(g(x))=g(f(x))$则称$f$对变换$g$有等变性。
 + 不变形 invariant: 假设我们的输入为$x$，函数为$f$, 此时我们先对输入做变换$g:g(x)=x'$，此时若有：$f(x)=f(x')$，则称$f$对变换$g$具有不变性。

简单的来说，CNN中的卷积操作中的参数共享使得它对**平移操作有等变性**，而一些**池化操作对平移有近似不变性**。
先来说前者， 我们举个很简单的例子，我们都知道CNN的第一层往往可以解释为一些简单的线条处理，比如竖直/水平线条检测等等，那么如果图像平移，显然并不会影响到这一层线条检测的功能，但是其输出也会做相应平移。
后者的之所以说是近似不变性，是因为池化层并非能保持完全不变，例如我们使用max池化，只要变换不影响到最大值，我们的池化结果不会收到影响，对于一个NxN的filter，只有一个值的变动会影响到输出， 其他的变换都不会造成扰动。 平均池化的近似不变性就稍弱些。这里池化的其实是一个非常强的先验，等于是忽视了这一步维数约简带来的信息损失而保证了近似不变性。

我们换个角度来说，CNN是既具有不变性，又具有等变性。 可以这么理解，如果我们的输出是给出图片中猫的位置，那么我们将图片中的猫从左边移到右边，这种平移也会反应在输出上，我们输出的位置也是从左边到右边，那么我们则可以说CNN有等变性；如果我们只是输出图片中是否有猫，那么我们无论把猫怎么移动，我们的输出都保持"有猫"的判定，因此体现了CNN的不变性。

## Introduction

CNN并不容易直接拓展到点云分析上。同时，相比2D图像的几何变换群，3D的聚合变换群要更复杂，因为3D物体经常可以被通过任意的3D位移和旋转来变换。尽管目前有很多group-invariant 操作可以对于几何变换群得到相同的特征，但是这样得到的特征不能应对一些内在对称情况，比如数字'6'和'9'的区分。相比之下，等变特征可以更好地表示和保留输入变换群中的几何变换信息。因此，可以得到一个对于$SE(3)$ group transformations等变的特征就有益。

处理对于$SE(3)$等变的特征的主要有两个难点：
+ 处理量大，卷积需要在6个维度进行($SE(3)$的自由度是6)。本文中受到可分离卷积思路的启发，将$SE(3)$空间的卷积分离为两个卷积，分别在$\mathbb{R}^3$和$SO(3)$空间内进行，因为$SE(3)$和$\mathbb{R}^3\times SO(3)$是同胚的。但这也不好进行，因$SE(3)$空间是非交换和非紧的(non-commutative and non-compact)。本文利用了有限旋转群（比如正二十面体）的优势，来将$SO(3)$拆分出来，大大减少卷积的计算量。
+ 在不损失关键结构信息和低计算量的的情况下充分的利用等变信息。特别的匹配两个群等变特征的匹配一般需要很多操作，比如匹配度计算，姿态估计，以及PnP问题求解等等。另一种利用等变特征的选择是直接通过池化来将其变为不变特征然后直接比较。本文认为原生的池化并不能很好的利用特征的等变结构，因此提出了一种新的池化方法，利用attention机制。
![](/assets/img/20211110/EquivariantF1.png)
相比不变特征，等变特征保留了更多空间结构信息，因此可以得到more discriminative representation。这种等变特征可以利用在推断3D旋转上，可以显著提高网络shape alignment task任务的性能。

## Method 

$SE(3)$的定义如下
$$
\begin{equation}
\mathrm{SE}(3)=\left\{A \mid A=\left[\begin{array}{cc}
R & t \\
0 & 1
\end{array}\right], R \in \mathrm{SO}(3), t \in \mathbb{R}^{3}\right\}
\end{equation}
$$
由于其是$\mathbb{R}^3\times SO(3)$的同胚空间，因此可以定义$SE(3)$空间中的连续特征表示函数表示为$\mathcal{F}(x_i,g_i): \mathbb{R}^3\times SO(3) \rightarrow \mathbb{R}^D$。具有对$SE(3)$中的等变形，应该满足$\forall A \in \operatorname{SE}(3), A(\mathcal{F} * h)(x, g)=$ $(A \mathcal{F} * h)(x, g)$

故卷积可以表示为如下形式

$$
\begin{aligned}
&(\mathcal{F} * h)(x, g) \\
=& \int_{x_{i} \in \mathbb{R}^{3}} \int_{g_{j} \in \operatorname{SO}(3)} \mathcal{F}\left(x_{i}, g_{j}\right) h\left(g^{-1}\left(x-x_{i}\right), g_{j}^{-1} g\right)
\end{aligned}
$$
where $h$ is a kernel $h(x, g): \mathbb{R}^{3} \times \mathrm{SO}(3) \rightarrow \mathbb{R}^{D}$.

其满足等分性的证明如下：
对于旋转$mathcal{R} \in SO(3)$和平移$\mathcal{T}\in \mathbb{R}^3$

旋转等变性：For convenience of notation, let $x_{i}^{\prime}=\mathcal{R}^{-1} x_{i}$, and $g_{j}^{\prime}=\mathcal{R}^{-1} g_{j}$
$$
\begin{aligned}
&\mathcal{R}\left(\mathcal{F} * h_{1}\right)(x, g)=\left(\mathcal{F} * h_{1}\right)(\mathcal{R} x, \mathcal{R} g) \\
&=\int_{x_{i} \in \mathbb{R}^{3}} \int_{g_{j} \in \operatorname{SO}(3)} \\
&\quad \mathcal{F}\left(x_{i}, g_{j}\right) h\left((\mathcal{R} g)^{-1}\left(\mathcal{R} x-x_{i}\right), g_{j}^{-1} \mathcal{R} g\right) \\
&=\int_{x_{i} \in \mathbb{R}^{3}} \int_{g_{j} \in \operatorname{SO}(3)} \\
&\quad \mathcal{F}\left(x_{i}, g_{j}\right) h\left(g^{-1}\left(x-\mathcal{R}^{-1} x_{i}\right),\left(\mathcal{R}^{-1} g_{j}\right)^{-1} g\right) \\
&=\int_{x_{i}^{\prime} \in \mathbb{R}^{3}} \int_{g_{j}^{\prime} \in \operatorname{SO}(3)} \\
&\quad \mathcal{F}\left(\mathcal{R} x_{i}^{\prime}, \mathcal{R} g_{j}^{\prime}\right) h\left(g^{-1}\left(x-x_{i}^{\prime}\right), g_{j}^{\prime-1} g\right) \\
&=\left(\mathcal{R} \mathcal{F} * h_{1}\right)(x, g)
\end{aligned}
$$

平移等变性： Let $x_{i}^{\prime}=\mathcal{T}^{-1} x_{i}$. Because $\mathcal{T}\left(x-x_{i}\right)=x-x_{i}$ :
$$
\begin{equation}
\begin{aligned}
&\mathcal{T}\left(\mathcal{F} * h_{1}\right)(x, g)=\left(\mathcal{F} * h_{1}\right)(\mathcal{T} x, g) \\
&=\int_{x_{i} \in \mathbb{R}^{3}} \int_{g_{j} \in \mathrm{SO}(3)} \\
&=\int_{x_{i} \in \mathbb{R}^{3}} \int_{g_{j} \in \mathrm{SO}(3)} \\
&\quad \mathcal{F}\left(x_{i}, g_{j}\right) h\left(g^{-1} \mathcal{T}\left(x-\mathcal{T}^{-1} x_{i}\right), g_{j}^{-1} g\right) \\
&=\int_{x_{i}^{\prime} \in \mathbb{R}^{3}} \int_{g_{j} \in \mathrm{SO}(3)} \\
&\quad \mathcal{F}\left(\mathcal{T} x_{i}^{\prime}, g_{j}\right) h\left(g^{-1}\left(x-x_{i}^{\prime}\right), g_{j}^{-1} g\right) \\
&=\left(\mathcal{T} \mathcal{F} * h_{1}\right)(x, g)
\end{aligned}
\end{equation}
$$

通过给点有限点集$\mathcal{P}$和有限旋转群$G$,可以得到上述卷积的离散化结果
$$
\begin{equation}
(\mathcal{F} * h)(x, g)=\sum_{x_{i} \in \mathcal{P}} \sum_{g_{j} \in G} \mathcal{F}\left(x_{i}, g_{j}\right) h\left(g^{-1}\left(x-x_{i}\right), g_{j}^{-1} g\right)
\end{equation}
$$

并可以等效成如下形式
$$
\begin{equation}
\begin{aligned}
(\mathcal{F} * h)(x, g) &=\sum_{x_{i} \in \mathcal{P}} \sum_{g_{j} \in G} \mathcal{F}\left(g^{-1}\left(x-x_{i}\right), g_{j}^{-1} g\right) h\left(x_{i}, g_{j}\right) \\
&=\sum_{x_{i}^{\prime} \in \mathcal{P}_{g}} \sum_{g_{j} \in G} \mathcal{F}\left(x-x_{i}^{\prime}, g_{j}^{-1} g\right) h\left(g x_{i}^{\prime}, g_{j}\right) .
\end{aligned}
\end{equation}
$$

不失一般性，这里假设坐标是表示在局部坐标系内的，其和参考帧的旋转为$g$，则有$g^{-1}x=x$,上述旋转之后的点集可以表示为
$$\begin{equation}
\mathcal{P}_{g}:\left\{g^{-1} x \mid x \in \mathcal{P}\right\}
\end{equation}
$$

不难看出，此时核函数$h$是可以通过有限的点集大小和有限旋转群来进行参数化的。如果离散点集大小为$|\mathcal{P}|$, 有限旋转群大小为$|G|$，则可以得到核函数的参数（kernel size）为$|\mathcal{P}| 
\times |G|$。

参考可分离卷积的思路，这里同样将卷积核拆分，成为一个$|\mathcal{P}|\times \{\mathbf{I}\}$的卷积核以及一个$|\{\mathbf{0}\}\times |G|$
$$
\begin{equation}
\begin{aligned}
\left(\mathcal{F} * h_{1}\right)(x, g) &=\sum_{x_{i}^{\prime} \in \mathcal{P}_{g}} \mathcal{F}\left(x-x_{i}^{\prime}, g\right) h_{1}\left(g x_{i}^{\prime}, \mathrm{I}\right) \\
\left(\mathcal{F} * h_{2}\right)(x, g) &=\sum_{g_{j} \in G} \mathcal{F}\left(x, g_{j}^{-1} g\right) h_{2}\left(\mathbf{0}, g_{j}\right)
\end{aligned}
\end{equation}
$$

作者将这两个分离出来的卷积分别成为$SE(3)$ *point convolution* 以及 $SE(3)$ *group convolution*， 合并称为 $SE(3)$ *separable point convolution (SPConv)*。

### SE(3) point convolution
通过近邻搜索得到近邻合集
$$
\begin{equation}
\mathcal{N}_{x}=\left\{x_{i} \in \mathcal{P} \mid\left\|x-x_{i}\right\| \leqslant r\right\}
\end{equation}
$$

点卷积可以表示为
$$
\begin{equation}
\left(\mathcal{F} * h_{1}\right)(x, g)=\sum_{x_{i}^{\prime} \in \mathcal{N}_{g x}} \mathcal{F}\left(x-x_{i}^{\prime}, g\right) h_{1}\left(g x_{i}^{\prime}\right)
\end{equation}
$$
其中$\mathcal{N}_{g x}=\left\{g^{-1}\left(x-x_{i}\right) \mid x_{i} \in \mathcal{N}_{x}\right\}$，上述核函数$h_1$由于不需要旋转信息输入，因此可以直接定义在canonical neighbor space $\mathcal{B}^3_r$。上述表达式用于计算在旋转$g$下的空间关系信息，这样核函数可以轻易拓展到任意的空间核函数。典型的形式可以分为显式和隐式两种，此处不详细展开

+ 显式：通过一个correlation function来加权
$$
\begin{equation}
h_{1}\left(x_{i}\right)=\sum_{k}^{K} \kappa\left(x_{i}, \tilde{y}_{k}\right) W_{k}
\end{equation}
$$

+ 隐式：将特征和局部坐标串联并加权等：

$$
\begin{equation}
\begin{aligned}
h_{1}(\mathcal{F}(x, g)) &=\sum_{x_{i} \in \mathcal{N}_{x}} h_{1}\left(\mathcal{F}\left(x_{i}, g\right), g^{-1} x_{i}\right) \\
&=\sum_{x_{i} \in \mathcal{N}_{x}}\left[\begin{array}{c}
\mathcal{F}\left(x_{i}, g\right) \\
g^{-1} x_{i}
\end{array}\right] W
\end{aligned}
\end{equation}
$$
### SE(3) group convolution

类似点卷积，对于给定的离散旋转群$G$,寻找其最相近的K个旋转（相当于旋转空间内的K近邻，得到旋转群的一个子集）。即$\mathcal{N}_{g}=\left\{g_{j} \in G\right\}_{K}$ and $\left\{W_{j} \in\right.$ $\left.\mathbb{R}^{D_{\text {in }} \times D_{\text {out }}}\right\}_{K}$, with the kernel size $K=\left|\mathcal{N}_{g}\right|$。得到卷积形式如下： 
$$
\begin{equation}
\left(\mathcal{F} * h_{2}\right)(x, g)=\sum_{g_{j} \in \mathcal{N}_{g}} \mathcal{F}\left(x, g_{j}^{-1} g\right) h_{2}\left(g_{j}\right)
\end{equation}
$$

通过上述拆分，计算量可以大大减少，尤其是当两个维度的核大小较大的时候。

![](/assets/img/20211110/EquivariantF2.png)

## Group Attentive Pooling

为了让rotation-equivariant 变为 invariant，那么应该要求 Attention是针对 rotation的，即
$$
\begin{equation}
A: G \rightarrow \mathbb{R}, A(g)=\left\{a_{g} \mid \sum_{g \in G} a_{g}=1\right\}
\end{equation}
$$

得到attention如下
$$
\begin{equation}
\mathcal{F}_{i n v}=\frac{\sum_{g} \exp \left(a_{g} / T\right) \mathcal{F}_{G}(g)}{\sum_{g} \exp \left(a_{g} / T\right)}
\end{equation}
$$