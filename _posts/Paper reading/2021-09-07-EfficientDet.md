---
layout: post
title: 'EfficientDet: Rethinking Model Scaling for Convolutional Neural Networks'
date: 2021-09-07
author: Poley
cover: '/assets/img/20210907/EffDet.png'
tags: 论文阅读
---

> 参考博客：https://zhuanlan.zhihu.com/p/96773680
提出了一种新的FPN结构，BiFPN，在简化的PANet上添加了**shortcut**。
![](/assets/img/20210907/EffDet.png)

同时，以往的特征融合方法对所有输入特征一视同仁，在BiFPN中引入了加权策略，类似于Attention

最简单的加权如下
$$
\begin{equation}
O=\sum_{i} w_{i} \cdot I_{i}
\end{equation}
$$

但是不归一化的权重（没有限制）容易导致训练不稳定，因此可以使用softmax得到权重
$$
\begin{equation}
O=\sum_{i} \frac{e^{w_{i}}}{\sum_{j} e^{w_{j}}} \cdot I_{i}
\end{equation}
$$

softmax速度较慢，可以用一种快速的归一化方法
$$
\begin{equation}
O=\sum_{i} \frac{w_{i}}{\epsilon+\sum_{j} w_{j}} \cdot I_{i}
\end{equation}
$$

之后在第六层使用加权的特征

$$
\begin{equation}
\begin{aligned}
P_{6}^{t d} &=\operatorname{Conv}\left(\frac{w_{1} \cdot P_{6}^{i n}+w_{2} \cdot \operatorname{Resize}\left(P_{7}^{i n}\right)}{w_{1}+w_{2}+\epsilon}\right) \\
P_{6}^{\text {out }} &=\operatorname{Conv}\left(\frac{w_{1}^{\prime} \cdot P_{6}^{i n}+w_{2}^{\prime} \cdot P_{6}^{t d}+w_{3}^{\prime} \cdot \operatorname{Resize}\left(P_{5}^{\text {out }}\right)}{w_{1}^{\prime}+w_{2}^{\prime}+w_{3}^{\prime}+\epsilon}\right)
\end{aligned}
\end{equation}
$$

EfficientDet组合了EfficentNet,BiFPN以及box prediction net。形成了自己的整体结构如下


![](/assets/img/20210907/EffDetF3.png)

之后同样使用EfficientNet中提出的复合模型扩张来决定最优的参数选择。

实验结果如下

![](/assets/img/20210907/EffDetT2.png)
