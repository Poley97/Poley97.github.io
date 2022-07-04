---
layout: post
title: 'Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning'
date: 2022-1-13
author: Poley
cover: '/assets/img/20220111/VAT.png'
tags: 论文阅读
---

这是一篇发表在2019TPAMI上的文章，非常经典。本文提出了虚拟对抗损失（Virtual Adversarial Training），是对抗损失（Adversarial Training）的进一步发展，使其可以用于半监督学习的过程当中，以高效显著的增加模型的鲁棒性和泛化性。

虚拟对抗损失（Virtual adversarial loss，VAT）被定义为模型在输入数据附近对抗局部扰动时，其条件概率分布的鲁棒性度量。其可以被用于半监督情况，并且不需要太多次的前向-反向传播即可。

神经网络模型的训练受制于两个问题，欠拟合和过拟合。为了解决这个问题，人们引入了正则化的概念，来缓解训练误差和测试误差之间不可避免的性能差距。而本文引入的VAT相当于是对于模型最敏感方向引入的一种正则化。

正则化相当于引入了我们对于模型的一种一种先验知识（或者是先验分布）。一种最常见的先验知识就是，一般自然出现的系统，其输出对于时间和空间的输入来说一般是平滑的。这一般是由于物理性质导致现实中的系统一般可以被描述为微分方程的形式。**当我们把这种假设引入模型中，我们一般要求其条件概率$p(y|x)$对于输入$x$来说是平滑的。**

目前已经有很多工作证明，在训练中引入随机的局部扰动对于训练时有益的。但是作者认为这仍然有一些不足之处。**作者发现通过这些随机噪声或者数据增强得到的各向同性平滑的预测器经常会对某一个方向上的扰动特别敏感，也就是对抗方向（adversarial direction），即类别预测（条件概率）$p(y=k|x)$最敏感的扰动方向。** $L1,L2$的正则化很容易受到这样敏感方向的影响，及时这些扰动小到人眼都分辨不出来。

之前来解决这个问题的一种方法是adversarial training对抗学习。**其作者提出，局部各向同性的输出分布不能通过使得模型对于各向同性的噪声鲁棒来获得**。这在直观上也很好理解，对于局部输出大多方向上已经平滑的模型，使用各向同性的噪声相当于分散了正则的约束力到各个方向上。然而大部分方向上是不需要这些正则的（本身已经平滑）。

本文将对抗学习引入到半监督问题上，对抗学习寻找一个可以减少模型正确分类概率的扰动方向，**而本方法寻找一个大幅扰动当前模型输出概率分布的扰动方向**。

首先简单介绍一下对抗学习，对抗学习的思路很简单，使得模型收到干扰后的输出分布和正确分布的差异尽可能小，即

$$
\begin{equation}
L_{\mathrm{adv}}\left(x_{l}, \theta\right):=D\left[q\left(y \mid x_{l}\right), p\left(y \mid x_{l}+r_{\mathrm{adv}}, \theta\right)\right]
\end{equation}
$$
其中
$$
\begin{equation}
\text { where } r_{\mathrm{adv}}:=\underset{r ;\|r\| \leq \epsilon}{\arg \max } D\left[q\left(y \mid x_{l}\right), p\left(y \mid x_{l}+r, \theta\right)\right]
\end{equation}
$$

其中$D$是用于衡量两个分布差异的函数，比如交叉熵或者KL散度。在有正确标签的情况下，对抗方向很好计算，如下所示，即输出分布差异对于输入梯度最大的方向（在$L2$范数下）
$$
\begin{equation}
r_{\text {adv }} \approx \epsilon \frac{g}{\|g\|_{2}}, \text { where } g=\nabla_{x_{l}} D\left[h\left(y ; y_{l}\right), p\left(y \mid x_{l}, \theta\right)\right]
\end{equation}
$$
如果在$L_\infty$下则可以得到更简单的形式，这是更加常用的方式。
$$
\begin{equation}
r_{\text {adv }} \approx \epsilon \operatorname{sign}(g)
\end{equation}
$$

在虚拟对抗学习中，目标仍然和上述类似，如下
$$
\begin{equation}
\begin{aligned}
&D\left[q\left(y \mid x_{*}\right), p\left(y \mid x_{*}+r_{\mathrm{qadv}}, \theta\right)\right] \\
&\text { where } r_{\mathrm{qadv}}:=\underset{r ;\|r\| \leq \epsilon}{\arg \max } D\left[q\left(y \mid x_{*}\right), p\left(y \mid x_{*}+r, \theta\right)\right]
\end{aligned}
\end{equation}
$$
但是问题在于缺少对于无标签数据的直接信息，因此需要将无变迁数据的$q(y|x)$替换成当前的模型输出（估计）$p(y|x,\theta)$。注意到当训练样本足够多的时候，上述两者的近似程度会比较高。故上式变为
$$
\begin{equation}
\begin{aligned}
\operatorname{LDS}\left(x_{*}, \theta\right) &:=D\left[p\left(y \mid x_{*}, \hat{\theta}\right), p\left(y \mid x_{*}+r_{\text {vadv }}, \theta\right)\right] \\
r_{\text {vadv }} &:=\underset{r ;\|r\|_{2} \leq \epsilon}{\arg \max } D\left[p\left(y \mid x_{*}, \hat{\theta}\right), p\left(y \mid x_{*}+r\right)\right],
\end{aligned}
\end{equation}
$$

对全部样本的损失平均可以得到最终的对抗损失
$$
\begin{equation}
\mathcal{R}_{\text {vadv }}\left(\mathcal{D}_{l}, \mathcal{D}_{u l}, \theta\right):=\frac{1}{N_{l}+N_{u l}} \sum_{x_{*} \in \mathcal{D}_{l}, \mathcal{D}_{u l}} \operatorname{LDS}\left(x_{*}, \theta\right) .
\end{equation}
$$
并加以有监督的负对数似然（交叉熵），可以得到模型的最终损失
$$
\begin{equation}
\ell\left(\mathcal{D}_{l}, \theta\right)+\alpha \mathcal{R}_{\mathrm{vadv}}\left(\mathcal{D}_{l}, \mathcal{D}_{u l}, \theta\right)
\end{equation}
$$
模型在这种损失下的预测演变过程如下所示，可以看到其通过寻找敏感边界可以很好的对无标注数据进行分类
![](/assets/img/20220111/VATF1.png)

需要解决的问题是如何找到对抗方向$r$可以看到上述LDS损失对于$r$求导是没有意义的，因其损失值和其导数在$r=0$时衡为$0$，因此这里作者转而求助损失的泰勒展开的二阶近似，即
$$
\begin{equation}
D(r, x, \hat{\theta}) \approx \frac{1}{2} r^{T} H(x, \hat{\theta}) r
\end{equation}
$$
很明显，当时海森矩阵在最大特征值对应的特征向量时，具有最大的损失$D$
$$
\begin{equation}
\begin{aligned}
r_{\mathrm{vadv}} & \approx \arg \max _{r}\left\{r^{T} H(x, \hat{\theta}) r ;\|r\|_{2} \leq \epsilon\right\} \\
&=\epsilon \overline{\epsilon u(x, \hat{\theta})}
\end{aligned}
\end{equation}
$$

然而有两个问题：
+ 海森矩阵不好求
+ 特征值也不好求（计算复杂度高）

这里作者引入了power iteration method 和 finite difference method来解决这个问题。

根据power iteration method的原理，随意去一个和主特征向量不垂直的向量，通过
$$
\begin{equation}
d \leftarrow \overline{H d}
\end{equation}
$$
的不断迭代可以得到非常近似的主特征向量。这样就避免了求特征值的高复杂度运算。为了避免海森矩阵的计算，这里引入了差分近似，即
$$
\begin{equation}
\begin{aligned}
H d & \approx \frac{\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=\xi d}-\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=0}}{\xi} \\
&=\frac{\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=\xi d}}{\xi},
\end{aligned}
\end{equation}
$$

那么就可以得到
$$
\begin{equation}
d \leftarrow \overline{\left.\nabla_{r} D(r, x, \hat{\theta})\right|_{r=\xi d}}
\end{equation}
$$

通过一阶求导就可以得到其对抗方向的方法，至此，问题就变得简单，可以使用和对抗学习类似的形式来求得对抗方向如下
$$
r_{\mathrm{vadv}} \approx \epsilon \frac{g}{\|g\|_{2}}
$$
where $g=\left.\nabla_{r} D[p(y \mid x, \hat{\theta}), p(y \mid x+r, \hat{\theta})]\right|_{r=\xi d}$.

最终的算法流程如下
![](/assets/img/20220111/VATA1.png)
