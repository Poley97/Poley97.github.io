---
layout: post
title: 'Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics'
date: 2021-12-06
author: Poley
cover: '/assets/img/20211206/UWL.png'
tags: 论文阅读
---
> 参考博客：https://blog.csdn.net/cdknight_happy/article/details/102618883

多任务联合学习可以提升各任务的学习效果，因为多任务可以共享数据集、共享低层特征。但多任务联合学习时，该如何对各子任务的损失函数进行加权才能取得最优的训练效果，这是本文所关心的问题。

多任务学习的性能高度依赖于一个在各个任务损失之间合适的权重。手动调整带来的成本太高，而作者发现各任务的最优权重取决于measurement和最终的任务噪声水平。本文提出的方法在学习场景的几何和语义上比较有优势，这里作者应用到了三个task上，分别是语义分割、实例分割、和深度估计。其结构如下所示
![](/assets/img/20211206/UWLF1.png)

一般的多任务学习中，通过手动设置的权重对于不同的损失进行加权求和。

$$
\begin{equation}
L_{\text {total }}=\sum_{i} w_{i} L_{i}
\end{equation}
$$

但是如下图所示，多任务学习的效果和权重高度相关。
![](/assets/img/20211206/UWLF2.png)

## Homoscedastic uncertainty as task-dependent uncertainty

贝叶斯模型中，可以建模的不确定度主要分为两种：
+ Epistemic uncertainty(认知不确定度): 模型本身由于知识不够而产生的不确定度，可以通过增加训练数据量来缓解；
+ Aleatoric uncertainty（偶然不确定度）: 表示数据中不能解释的信息的不确定度；

其中 Aleatoric uncertainty又可以分为以下两种：
+ Data-dependent or Heteroscedastic(异方差性): 由输入数据带来的不确定度，最终以模型预测结果的形式体现；
+ Task-dependent or Homoscedastic(同方差性): 不取决于输入数据的不确定度，并不是模型输入，而是一个量，对于所有的输入数据相同而对不同的任务不同。

故后者包含了任务之间的相对置信度，反映了各个任务内在的不确定度。因此其较适合用来给任务分配权重。

## Multi-task likelihoods
对于回归任务，将网络的输出的似然函数是高斯函数，即
$$
\begin{equation}
p\left(\mathbf{y} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)=\mathcal{N}\left(\mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma^{2}\right)
\end{equation}
$$
对于分类，可以直接得到
$$
\begin{equation}
p\left(\mathbf{y} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)=\operatorname{Softmax}\left(\mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)
\end{equation}
$$

一般在多任务模型中，一般讲似然在输出熵分解，假设模型的输出是分布的充分统计量，得到的似然如下
$$
\begin{equation}
p\left(\mathbf{y}_{1}, \ldots, \mathbf{y}_{K} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)=p\left(\mathbf{y}_{1} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \ldots p\left(\mathbf{y}_{K} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)
\end{equation}
$$

在最大似然的优化策略下，最大化模型的似然概率，那么对于回归问题，等效于最大化如下函数
$$
\begin{equation}
\log p\left(\mathbf{y} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \propto-\frac{1}{2 \sigma^{2}}\left\|\mathbf{y}-\mathbf{f}^{\mathbf{W}}(\mathbf{x})\right\|^{2}-\log \sigma
\end{equation}
$$

为了最大化对数自然，网络需要调整模型参数$\mathbf{W}$和噪声参数$\sigma$。

假设两个输出向量$\mathbf{y}_1,\mathbf{y}_2$输出服从高斯分布，则模型输出的似然概率为
$$
\begin{equation}
\begin{aligned}
p\left(\mathbf{y}_{1}, \mathbf{y}_{2} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) &=p\left(\mathbf{y}_{1} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \cdot p\left(\mathbf{y}_{2} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \\
&=\mathcal{N}\left(\mathbf{y}_{1} ; \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_{1}^{2}\right) \cdot \mathcal{N}\left(\mathbf{y}_{2} ; \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_{2}^{2}\right)
\end{aligned}
\end{equation}
$$
则通过最小化损失你是函数或者最大化似然可以得到
$$
\begin{equation}
\begin{aligned}
&=-\log p\left(\mathbf{y}_{1}, \mathbf{y}_{2} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \\
&\propto \frac{1}{2 \sigma_{1}^{2}}\left\|\mathbf{y}_{1}-\mathbf{f}^{\mathbf{W}}(\mathbf{x})\right\|^{2}+\frac{1}{2 \sigma_{2}^{2}}\left\|\mathbf{y}_{2}-\mathbf{f}^{\mathbf{W}}(\mathbf{x})\right\|^{2}+\log \sigma_{1} \sigma_{2} \\
&=\frac{1}{2 \sigma_{1}^{2}} \mathcal{L}_{1}(\mathbf{W})+\frac{1}{2 \sigma_{2}^{2}} \mathcal{L}_{2}(\mathbf{W})+\log \sigma_{1} \sigma_{2}
\end{aligned}
\end{equation}
$$

上述式子意味着当一个task的方差越大，权重则越小。上述是回归任务的推导，下面是分类任务。

$$
\begin{equation}
p\left(\mathbf{y} \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma\right)=\operatorname{Softmax}\left(\frac{1}{\sigma^{2}} \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right)
\end{equation}
$$

可以将其视为一个玻尔兹曼分布，而里面这个尺度因子$\sigma^2$经常被称作问题 

故对于分类任务，其对数似然函数如下
$$
\begin{equation}
\begin{aligned}
\log p\left(\mathbf{y}=c \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma\right) &=\frac{1}{\sigma^{2}} f_{c}^{\mathbf{W}}(\mathbf{x}) \\
&-\log \sum_{c^{\prime}} \exp \left(\frac{1}{\sigma^{2}} f_{c^{\prime}}^{\mathbf{W}}(\mathbf{x})\right)
\end{aligned}
\end{equation}
$$

那么联合损失如下：
$$
\begin{equation}
\begin{aligned}
=&-\log p\left(\mathbf{y}_{1}, \mathbf{y}_{2}=c \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x})\right) \\
=&-\log \mathcal{N}\left(\mathbf{y}_{1} ; \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_{1}^{2}\right) \cdot \operatorname{Softmax}\left(\mathbf{y}_{2}=c ; \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_{2}\right) \\
=& \frac{1}{2 \sigma_{1}^{2}}\left\|\mathbf{y}_{1}-\mathbf{f}^{\mathbf{W}}(\mathbf{x})\right\|^{2}+\log \sigma_{1}-\log p\left(\mathbf{y}_{2}=c \mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma_{2}\right) \\
=& \frac{1}{2 \sigma_{1}^{2}} \mathcal{L}_{1}(\mathbf{W})+\frac{1}{\sigma_{2}^{2}} \mathcal{L}_{2}(\mathbf{W})+\log \sigma_{1} \\
&+\log \frac{\sum_{c^{\prime}} \exp \left(\frac{1}{\sigma_{2}^{2}} f_{c^{\prime}}^{\mathbf{W}}(\mathbf{x})\right)}{\left(\sum_{c^{\prime}} \exp \left(f_{c^{\prime}}^{\mathbf{W}}(\mathbf{x})\right)\right)^{\frac{1}{\sigma_{2}^{2}}}} \\
\approx & \frac{1}{2 \sigma_{1}^{2}} \mathcal{L}_{1}(\mathbf{W})+\frac{1}{\sigma_{2}^{2}} \mathcal{L}_{2}(\mathbf{W})+\log \sigma_{1}+\log \sigma_{2},
\end{aligned}
\end{equation}
$$

这就可以看做是对两个损失函数的自动加权，当一个损失函数的方差变大的时候，其对网络的贡献就会减少。同时，后两项也防止$\sigma_1,\sigma_2$会变得过大的问题（这样前两项就明显变小）。

在实际应用中，作者的做法是通过网络预测对数方差$\log \sigma^2$来实现上述方法。

![](/assets/img/20211206/UWLT1.png)