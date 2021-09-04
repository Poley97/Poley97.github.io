---
layout: post
title: 'Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results'
date: 2021-08-17
author: Poley
cover: '/assets/img/20210817/MT.png'
tags: 论文阅读
---


提出一种mean teacher方法，通过对模型权重而不是预测标签来进行平均来训练模型。

在深度学习中，手动对训练数据添加高质量的标签往往是高成本的，因此使用正则化方法来利用未标注数据来有效的减小模型的过拟合问题在半监督学习中是非常必要的。

下图是对于一个二分类问题图示，展示了添加噪声以及未标注数据对于训练的影响。
![](/assets/img/20210817/MTF1.png)

通过添加噪声，可以使得决策边界尽可能的原理标注数据点，如(b)所示。由于未标注数据没有标签，所以一些方法提出通过对有噪声和没有噪声的数据预测，并使用cosistency loss来建立两者之间的联系。这里，一个模型同时担任了teacher和students的作用。由于产生的标签并不准确，对他们赋予太多的权重可能会阻止模型学习到新的知识。也就是受困于**confirmation bias**，解决这个困境的方法就是提供更高质量的targets。

有两种方法可以提高target的质量。一种是谨慎的选择添加的噪声，而不是仅仅简单的使用加性或者乘性噪声。另一种方法就是谨慎的选择teacher模型，而不是直接复制student模型。

这两种方法是互补的，第一种方法的代表是**Virtual Adaversarial Training**，该方法具有很好的性能，但不在本文的讨论范围之内。

本文的目标是建立一个更好的teacher模型，而不需要额外的训练。考虑到softmax输出一般不能提供训练数据之外的准确的预测，这里可以通过在预测的时候加入噪声来环节这一点，也就是一个noisy teacher可以得到更准确的targets。

$\Pi$模型通过对预测值做EMA来实现上述目标。但是其具有较高的时空复杂度，不适合在线训练。

为了解决这个问题，本文提出对权重进行EMA，如下所示。

![](/assets/img/20210817/MTF2.png)

本模型的优点在于：
+ teacher和student之间的准确标签反馈更快
+ 实现在线训练

consistency cost J

$$
\begin{equation}
J(\theta)=\mathbb{E}_{x, \eta^{\prime}, \eta}\left[\left\|f\left(x, \theta^{\prime}, \eta^{\prime}\right)-f(x, \theta, \eta)\right\|^{2}\right]
\end{equation}
$$

参数更新 

$$
\begin{equation}
\theta_{t}^{\prime}=\alpha \theta_{t-1}^{\prime}+(1-\alpha) \theta_{t}
\end{equation}
$$

## 分析

### 训练曲线

实验结果表明，EMA加权的模型可以给出更准确的预测，相比单纯的student models。

同时，使用EMA加权的模型也提升了半监督训练的效果。体现出一个虚拟的反馈循环，即teacher通过consistency cost来提升students的性能，而students通过EMA来提升 teacher的性能。
### 结论
1. 噪声仍然是必须的（数据增强下可以不用噪声）

2. 对于EMA decay和consistency weight是敏感的。在训练初期可以使用较小的decay，让模型快速忘记之前的知识（0.99），而之后换成大的decay（0.999）

3. classification和consistency还不能完全decoupling。同时使用两者的输出可能有更好的结果。（可能理解不正确，之后再进一步理解一下）

4. 使用KL散度损失的效果不如MSE。
