---
layout: post
title: 'A Comprehensive Survey on Transfer Learning'
date: 2021-08-19
author: Poley
cover: '/assets/img/20210819/TL.png'
tags: 论文阅读
---
![](/assets/img/20210819/TL.png)

本文是目前最新的全面的关于迁移学习的综述，包含了超过40个具有代表性的迁移学习的工作。同样也简要的介绍了迁移学习的应用。在实验中使用了超过20种迁移学习方法，并在不同的数据集上使用。

本文中出现的一些符号如下所示

![](/assets/img/20210819/TLS1.png)

![](/assets/img/20210819/TLS2.png)


# Introduction
迁移学习主要关注 transfering the knowledge across domains。这个概念最早来自于心理学、

**迁移过去的知识并不总是对于另一个tasks具有正面的帮助。** 比如法语和西班牙语同源，但是一个语言中养成的习惯可能导致另一个语言中错误的词汇或者连词用法。在心理学中，这种之前的经验会对新任务产生消极影响的现象称为**negative transfer**，在迁移学习领域也一样。造成这种现象的原因可能集中在几个因素上，比如：
+ 源域和目标域的差异
+ 学习者找到可迁移的以及有帮助的知识的能力

在相关文献中，具有更详细的分析。

总的来说，根据不同域之间的差异，迁移学习主要可以分为两类。**Homogeneous and heterogeneous transfer learning**。前者一般主要为了解决**domains are of the same feature space**的情况，这种情况下，许多研究假设doamins通常只有边缘分布不同。因此，这里通过**correcting the sample selection bias or covariate shift**来做自域适应。

上述情况不总是成立，比如情绪分类问题，一个词可能具有不同的意思，在不同的情况下。这称为**context feature bias**。为了解决这种问题，一些研究进一步去适应条件分布。Hetergeneous transfer learning一般为了解决不同特征空间情况的域适应问题，故其需要**feature space adaptation**，使得问题更加复杂。

以下内容主要分为7个部分

+ 明确迁移学习和其他机器学习技术的不同
+ 介绍常用符号以及迁移学习的定义
+ 从数据和模型角度分别说明迁移学习方法
+ 迁移学习的应用
+ 实验
+ 总结
  
# Related work
本部分主要声明迁移学习和其他常见技术的区别

## Semi-supervised learning

半监督学习中，标注数据和未标注数据是属于同一个分布的。这点和迁移学习相反，迁移学习的两个domain一般是不同的。但是，半监督学习的主要假设：**smoothness,cluster and manifold assumptions**经常用于迁移学习中。值得注意是，半监督迁移学习的说法是一个矛盾项。

## Multiview Learning

多视角学习主要关于于从多视角数据。比如视频中的信息可以通过图像和音频两种视角来得到。简要地说，多视角学习通过学习多个视角上的描述，来得到冗余的信息从而提升性能。常见的方法有**subspace learning , multikernel learning 以及 co-training**。Multiview Learning也可以被用到一些迁移学习方法中。

## Multitask Learning

多任务学习一半同时学习多个相关的tasks。其通过利用不同task的内部相关性来强化对于每一个task的学习，也就是考虑到了不同task之间的关联和区别。这样，每个task的泛化性本身就已经增强了。多任务学习和迁移学习的主要区别是，迁移学习单纯的转移相关域的知识，而多任务通过同时学习相关的task来转换不同域之间的知识。**也就是，多任务学习对于不同的task保持相同的关注，而迁移学习主要关注target task的性能，而不是source task。**但两者依然有很多相似点，他们都通过知识迁移来提升学习期的性能，并且也经常使用一些相同的构建模型的策略，比如**feature transformation and parameter sharing**。这两者也经常结合起来。

# Overview 

## Definition

首先给出了一些定义，用于方便进行后续的说明

+ **Domain**:一个域$\mathcal{D}$由两部分组成，特征空间$\chi$和边缘分布$P(X)$。即$D=\{\chi,P(X)\}$
+ **Task**: 一个任务$\mathcal{T}$由两部分组成，标签空间$\mathcal{Y}$和判别函数$f$，即$\mathcal{T}=\{\mathcal{Y},f\}$
+ **Transfer Learning**: 给定一些源域的观测，比如$\left\{\left(\mathcal{D}_{S_{i}}, \mathcal{T}_{S_{i}}\right) \mid i=1, \ldots, m^{S}\right\}$，其中$ m^{S}\in \mathbb{N}^+$代表源域的数量，再给定一系列目标域的观测$\left\{\left(\mathcal{D}_{T_{i}}, \mathcal{T}_{T_{i}}\right) \mid i=1, \ldots, m^{T}\right\}$，其中$ m^{T}\in \mathbb{N}^+$代表目标域的数量，利用源域的知识来提升目标域学习到的决策函数$f^{T_{j}}\left(j=1, \ldots, m^{T}\right)$的性能。

上述最常见的情况是两个$m$均为1的情况。常见情况是在源域中具有大量的标注样本，而目标域中只有少量的标注样本。
另一个在迁移学习中常用的就是**域适应 domain adaptation**，迁移学习经常依赖于域适应过程，尝试去减小域之间的差异。

## Categorization of Transfer Learning

从数据标注情况，可以分为三类
+ Transductive transfer learning : 源域有标签，目标域无标签
+ Inductive transfer learning : 源域目标域都有标签
+ Unsupervised transfer learning : 源域和目标域都没有标签

从域的一致性上可以分为homogeneous 和 hetergeneous，前文说过，不再赘述。

从另一个角度来说，迁移学习可以分为四类
+ Instance-Based : 主要通过instance weighting strategy方法
+ Feature-Based ： 进一步分为两种，对称与非对称。非对称方法转换源域特征到目标域，而对称方法寻找两者共同的隐特征，使得源域和目标域特征一起转换到一个新的特征表示
+ Parameter-Based ： 转换模型或者参数中包含的知识
+ Relational-Based ： 转换逻辑或规则等等
  
总结如下图所示
![](/assets/img/20210819/TLF2.png)

# Data-Based Interpretation

这部分主要对应与比较泛化层面的Instance-Based和Feature-Based迁移学习。主要通过调整和转换数据来实现知识的迁移。

![](/assets/img/20210819/TLF3.png)

本综述里更关注于hemogeneous transfer learning，其目的就是减少两个域之间的分布差异。从数据层面上来说，主要有个方法，分别是Instance Weighting Strategy和Feature transformation。

## Instance Weighting Strategy

假设一种简单的情况，目前具有很多的标注数据在源域中，而只有少量的目标域数据是可用的。两者唯一的区别是边缘分布不同，即$P^{S}(X) \neq P^{T}(X)$，$\left.P^{S}(Y \mid X)=P^{T}(Y \mid X)\right)$。在这种情况下很自然的想法是去适应目标域数据的边缘分布，一个简单的方法就是通过对源域数据的loss进行加权（也就是对数据加权）。

$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{(\mathbf{x}, y) \sim P^{T}}[\mathcal{L}(\mathbf{x}, y ; f)] &=\mathbb{E}_{(\mathbf{x}, y) \sim P^{S}}\left[\frac{P^{T}(\mathbf{x}, y)}{P^{S}(\mathbf{x}, y)} \mathcal{L}(\mathbf{x}, y ; f)\right] \\
&=\mathbb{E}_{(\mathbf{x}, y) \sim P^{S}}\left[\frac{P^{T}(\mathbf{x})}{P^{S}(\mathbf{x})} \mathcal{L}(\mathbf{x}, y ; f)\right]
\end{aligned}
\end{equation}
$$

更泛化的形式如下

$$
\begin{equation}
\min _{f} \frac{1}{n^{S}} \sum_{i=1}^{n^{S}} \beta_{i} \mathcal{L}\left(f\left(\mathbf{x}_{i}^{S}\right), y_{i}^{S}\right)+\Omega(f)
\end{equation}
$$

Kernel mean matching(KMM)是其中一种方法，如下

$$
\begin{equation}
\begin{aligned}
&\underset{\beta_{i} \in[0, B]}{\arg \min }\left\|\frac{1}{n^{S}} \sum_{i=1}^{n^{S}} \beta_{i} \Phi\left(\mathbf{x}_{i}^{S}\right)-\frac{1}{n^{T}} \sum_{j=1}^{n^{T}} \Phi\left(\mathbf{x}_{j}^{T}\right)\right\|_{\mathcal{H}}^{2} \\
&\text { s.t. }\left|\frac{1}{n^{S}} \sum_{i=1}^{n^{S}} \beta_{i}-1\right| \leq \delta
\end{aligned}
\end{equation}
$$

通过匹配源域和目标域的在kernel Hilbert space(RKHS)中的距离来解决参数估计问题。一般可以转换为二次规划问题，一旦参数确定，学习器就可以在源域加权的数据上学习，实现迁移。

还有一些方法，比如KL重要性估计过程等。通过这些工作，有一些instance-based transfer learning框架或者算法被提出，比如 multisource framework termed two-stage weighting framework for multisource-domain adaptation  (2SW-MDA)，主要分为两个阶段
+ Instance weighting : 通过加权来减小边缘分布的差距 
+ Domain weighting ： 通过加权各个域来减小条件分布差距

## Feature Transformation Strategy

主要目标就是找到一个隐特征，作为一个迁移学习的桥梁。建立新特征的目标包括最小化他们的边缘分布和条件分布区别，保持潜在的数据结构，并且找到特征之间的联系。主要可以分为
+ Feature augmentation
+ Feature reduction,又可以分为Feature mapping feature clustering,feature selection and feature encoding
+ Feature Alignment

### Disruibution Difference Metric 
特征变换的一个重要的目标就是减小不同域之间的特征分布差异，因此，如何度量特征分布的差异就是一个重要问题。

![](/assets/img/20210819/TLT1.png)

在迁移学习中常用的一种称为**maximunm mean discrepancy(MMD)**

$$
\begin{equation}
\operatorname{MMD}\left(X^{S}, X^{T}\right)=\left\|\frac{1}{n^{S}} \sum_{i=1}^{n^{S}} \Phi\left(\mathbf{x}_{i}^{S}\right)-\frac{1}{n^{T}} \sum_{j=1}^{n^{T}} \Phi\left(\mathbf{x}_{j}^{T}\right)\right\|_{\mathcal{H}}^{2}
\end{equation}
$$

其通过计算RKHS内的Instance平均的距离来衡量两个分布的区别。

### Feature Augmentation

常见的方法有feature replication and feature stacking。略

### Feture Mapping

传统的机器学习里，有一些典型的Mapping方法，比如PCA和核PCA。但是这些方法主要关注的数据的方差而非数据的分布区别。为了解决数据分布上的区别，可以简单构建一个目标函数如下

$$
\begin{equation}
\min _{\Phi}\left(\operatorname{DIST}\left(X^{S}, X^{T} ; \Phi\right)+\lambda \Omega(\Phi)\right) /\left(\operatorname{VAR}\left(X^{S} \cup X^{T} ; \Phi\right)\right)
\end{equation}
$$

其中$\Phi$是一个低维度映射。这个目标函数的目的就是找到一个映射函数$\Phi$来最小化两者之间的边缘分布差异，同时最大化数据的方差。但是找到显式的映射$\Phi$并不容易。一般通过线性映射或者核技巧来实现。主要有三种方法：

+ Mapping learning + feature extraction： 学习向高维的映射，再用PCA降维
+ Mapping construction + mapping learning ： 指定像高维的映射，学习向低维的映射
+ Direct low-dimensional mapping learning ：通常很难找到，除非添加一些约束，比如线性。

还有一些方法同样希望匹配条件分布，并且保留数据的结构。其目标函数可以泛化如下

$$
\begin{equation}
\begin{aligned}
&\min _{\Phi} \mu \operatorname{DIST}\left(X^{S}, X^{T} ; \Phi\right)+\lambda_{1} \Omega^{\mathrm{GEO}}(\Phi)+\lambda_{2} \Omega(\Phi) \\
&\quad+(1-\mu) \operatorname{DIST}\left(Y^{S}\left|X^{S}, Y^{T}\right| X^{T} ; \Phi\right) \\
&\text { s.t. } \Phi(X)^{\mathrm{T}} H \Phi(X)=I, \quad \text { with } H=I-(\mathbf{1} / n) \in \mathbb{R}^{n \times n}
\end{aligned}
\end{equation}
$$

通过$\mu$来调解两者之间的平衡，而$\Omega$则是一个控制集合结构的正则项。约束则是保持数据的协方差不变。

在一些task中，目标域并没有标签，或者只有有限的标签。这种标签的缺乏使得其很难对分布进行估计，这种情况下，一些方法使用伪标签(pseudo-label)的策略来分配标签。

上式根据$\mu$和$\lambda$的不同，可以衍生出很多方法，此处略。

### Feature Cluster

待完成


# Model-Based Interpretation

待完成




