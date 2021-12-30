---
layout: post
title: 'Balanced Meta-Softmax
for Long-Tailed Visual Recognition'
date: 2021-12-06
author: Poley
cover: '/assets/img/20211206/BMS.png'
tags: 论文阅读
---

本文由商汤科技新加坡团队提出。本文提出对于最常用的softmax函数，在长尾问题情况下会给出有偏的梯度。本文提出Balanced Softmax，一个简洁的softmax的无偏拓展。再进一步和Meta Sampler复合，可以进一步提升模型对长尾问题的学习能力。

相关的工作主要有一下几个方面：
+ Data Re-Balancing: 重采样是尝试通过重采样来修复数据集的不平衡分布。但是这样可能会导致尾类别的过拟合问题。重加权通过改变样本在损失中的权重来模拟平衡的分布，由此带来的是不稳定问题，在极度不平衡条件下产生的过大的权重可能会导致训练不稳定。
+ Loss Function Engineering: 提出一些新的损失函数和机制来解决长尾问题。本文属于这一类的方法。
+ Meta-Learning: 使用元学习方法来估计样本的权重，实现sample-based re-weight approach。这一般需要一个干净无偏的数据集，一般是development set，用于学习估计权重。
+ Decoupled Training: 分离训练对于长尾问题是一个简单有效的方法。但是其对于高度不平衡的数据集效果不够充分。但是对于本文提出的BALMS可以带阿里额外的提升。

## Balanced Meta-Softmax


### Balance Softmax
#### Label Distribution Shift
多类别Softmax回归一般都感兴趣与估计条件概率，可以建模成一个多项式分布如下：
$$
\begin{equation}
\phi=\phi_{1}^{1\{y=1\}} \phi_{2}^{1\{y=2\}} \cdots \phi_{k}^{1\{y=k\}} ; \quad \phi_{j}=\frac{e^{\eta_{j}}}{\sum_{i=1}^{k} e^{\eta_{i}}} ; \quad \sum_{j=1}^{k} \phi_{j}=1
\end{equation}
$$
其中每个类别的条件概率可以表示如下
$$
\begin{equation}
\phi_{j}=p(y=j \mid x)=\frac{p(x \mid y=j) p(y=j)}{p(x)}
\end{equation}
$$

这样，当训练数据集和测试数据集从同一个条件概率$p(x|y=j)$中产生，其在训练集和数据集之间仍然存在差异，当类别先验概率$p(y=j)$和数据先验概率$p(x)$不同的时候。我们定义$\phi$和$\phi$分别是平衡的测试集和不平衡的训练集上的条件分布，可以得出结论，Softmax给出了$\phi$的有偏估计。

对于平衡数据集，其后验概率可以表示为
$$
\begin{equation}
\phi_{j}=p(y=j \mid x)=\frac{p(x \mid y=j)}{p(x)} \frac{1}{k}
\end{equation}
$$
其中K是类别数。同理，对于不平衡的数据后验概率可以表示为
$$
\begin{equation}
\hat{\phi}_{j}=\hat{p}(y=j \mid x)=\frac{p(x \mid y=j)}{\hat{p}(x)} \frac{n_{j}}{\sum_{i=1}^{k} n_{i}} .
\end{equation}
$$
则如果$\phi$用标准softmax function表示，可以设（恒等）
$$
\begin{equation}
\eta_{j}=\log \left(\frac{\phi_{j}}{\phi_{k}}\right)
\end{equation}
$$
对于上式两边同减
$$
\begin{equation}
-\log \left(\phi_{j} / \hat{\phi}_{j}\right)
\end{equation}
$$
得到
$$
\begin{equation}
\eta_{j}-\log \frac{\phi_{j}}{\hat{\phi}_{j}}=\log \left(\frac{\phi_{j}}{\phi_{k}}\right)-\log \left(\frac{\phi_{j}}{\hat{\phi}_{j}}\right)=\log \left(\frac{\hat{\phi}_{j}}{\phi_{k}}\right)
\end{equation}
$$
进而
$$
\begin{equation}
\begin{gathered}
\phi_{k} e^{\eta_{j}-\log \frac{\phi_{j}}{\hat{\phi}_{j}}}=\hat{\phi}_{j} \\
\phi_{k} \sum_{i=1}^{k} e^{\eta_{i}-\log \frac{\phi_{i}}{\phi_{i}}}=\sum_{i=1}^{k} \hat{\phi}_{i}=1 \\
\phi_{k}=1 / \sum_{i=1}^{k} e^{\eta_{i}-\log \frac{\phi_{i}}{\phi_{i}}}
\end{gathered}
\end{equation}
$$
带回可以得到
$$
\begin{equation}
\hat{\phi}_{j}=\phi_{k} e^{\eta_{j}-\log \frac{\phi_{j}}{\hat{\phi}_{j}}}=\frac{e^{\eta_{j}-\log \frac{\phi_{j}}{\hat{\phi}_{j}}}}{\sum_{i=1}^{k} e^{\eta_{i}-\log \frac{\phi_{i}}{\hat{\phi}_{i}}}}
\end{equation}
$$
注意到
$$
\begin{equation}
\phi_{j}=p(y=j \mid x)=\frac{p(x \mid y=j)}{p(x)} \frac{1}{k} ; \quad \hat{\phi}_{j}=\hat{p}(y=j \mid x)=\frac{p(x \mid y=j)}{\hat{p}(x)} \frac{n_{j}}{n}
\end{equation}\begin{equation}
\phi_{j}=p(y=j \mid x)=\frac{p(x \mid y=j)}{p(x)} \frac{1}{k} ; \quad \hat{\phi}_{j}=\hat{p}(y=j \mid x)=\frac{p(x \mid y=j)}{\hat{p}(x)} \frac{n_{j}}{n}
\end{equation}\begin{equation}
\phi_{j}=p(y=j \mid x)=\frac{p(x \mid y=j)}{p(x)} \frac{1}{k} ; \quad \hat{\phi}_{j}=\hat{p}(y=j \mid x)=\frac{p(x \mid y=j)}{\hat{p}(x)} \frac{n_{j}}{n}
\end{equation}
$$
故
$$
\begin{equation}
\log \frac{\phi_{j}}{\hat{\phi}_{j}}=\log \frac{n}{k n_{j}}+\log \frac{\hat{p}(x)}{p(x)}
\end{equation}
$$
最终得到
$$
\begin{equation}
\hat{\phi}_{j}=\frac{e^{\eta_{j}-\log \frac{n}{k n_{j}}-\log \frac{\hat{p}(x)}{p(x)}}}{\sum_{i=1}^{k} e^{\eta_{i}-\log \frac{n}{k n_{i}}-\log \frac{\hat{p}(x)}{p(x)}}}=\frac{n_{j} e^{\eta_{j}}}{\sum_{i=1}^{k} n_{i} e^{\eta_{i}}}
\end{equation}
$$
则非平衡数据集$\hat{\phi}$可以表示为如下所示：
$$
\begin{equation}
\hat{\phi}_{j}=\frac{n_{j} e^{\eta_{j}}}{\sum_{i=1}^{k} n_{i} e^{\eta_{i}}}
\end{equation}
$$
故定义Balanced Softmax function为
$$
\begin{equation}
\hat{l}(\theta)=-\log \left(\hat{\phi}_{y}\right)=-\log \left(\frac{n_{y} e^{\eta_{y}}}{\sum_{i=1}^{k} n_{i} e^{\eta_{i}}}\right)
\end{equation}
$$

#### Generalization Error Bound
泛化误差界指的是对于模型给定的训练误差，其测试误差的上界。对于尾类别来说，其具有更大的泛化误差界，下面证明优化Balanced Softmax function等价于最小化模型的泛化误差界。

一般来说，更大的margin代表更小的泛化误差上界（个人人为类似于SVM）。Margin bound 通常限制0-1错误
$$
\begin{equation}
e r r_{0,1}=\operatorname{Pr}\left[\theta_{y}^{T} f(x)<\max _{i \neq y} \theta_{i}^{T} f(x)\right]
\end{equation}
$$
但是0-1误差不好求导优化，因此将其松弛成连续函数可以得到
$$
\begin{equation}
\operatorname{err}(t)=\operatorname{Pr}\left[t<\log \left(1+\sum_{i \neq y} e^{\theta_{i}^{T} f(x)-\theta_{y}^{T} f(x)}\right)\right]=\operatorname{Pr}\left[l_{y}(\theta)>t\right]
\end{equation}
$$
其中$l_y$是一个标准的Softmax的负对数似然，比如交叉熵。$t$是一个阈值，那么一般把类别$j$的maigin表示为
$$
\begin{equation}
\gamma_{j}=t-\max _{(x, y) \in S_{j}} l_{j}(\theta)
\end{equation}
$$

对于任意的$t>0$，对于所有$\gamma_j>0$，在至少$1-\delta$的概率下成立
$$
\begin{equation}
\operatorname{err}_{b a l}(t) \lesssim \frac{1}{k} \sum_{j=1}^{k}\left(\frac{1}{\gamma_{j}} \sqrt{\frac{C}{n_{j}}}+\frac{\log n}{\sqrt{n_{j}}}\right) ; \quad \gamma_{j}^{*}=\frac{\beta n_{j}^{-1 / 4}}{\sum_{i=1}^{k} n_{i}^{-1 / 4}},
\end{equation}
$$
在限制各类margin的总和为$\beta$，最优值可以通过柯西施瓦茨不等式得到。进一步为了获得更优的margin，可以得到每个类别的预期训练损失如下
$$
\begin{equation}
\hat{l}_{j}^{*}(\theta)=l_{j}(\theta)+\gamma_{j}^{*}, \quad \text { where } \quad l_{j}(\theta)=-\log \left(\phi_{j}\right)
\end{equation}
$$
进一步可以得到
![](/assets/img/20211206/BMSFO10.png)
在实验中，发现将常数$1/4$替换为$1$效果更好，这可能说明上述的约束并不需要是紧的。
### Meta Sampler
重采样是一种解决类别不平衡的常用方法，其中最常用的是Class-balanced sampler。

而结合Blanced Softmax 和 CBS效果却不佳，经过作者的分析（略），发现这是由于发生了过平衡问题，即每个类别的的权重从数量倒数$1/n_j$变为了$1/n_j^2$，从而偏离的最优的分布。

为了解决上述过平衡的问题，本文引入了Meta Sampler。为了得到每个类别的最优采样率，这里设置了一个双层（双阶段）的学习策略，即对于采样分布的参数和分类器的参数分开学习，固定一个优化另一个，如下所示
![](/assets/img/20211206/BMSFO13.png)

故其总流程如下所示
![](/assets/img/20211206/BMSA1.png)

## Experiments
![](/assets/img/20211206/BMSF1.png)
![](/assets/img/20211206/BMST2.png)

对于类别边缘概率$p(y)$的可视化如下
![](/assets/img/20211206/BMSF2.png)