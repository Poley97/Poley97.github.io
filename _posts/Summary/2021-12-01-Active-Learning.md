---
layout: post
title: '主动学习方法'
date: 2021-12-01
author: Poley
cover: '/assets/img/mycat3.jpg'
tags: 总结
---
>参考博客:https://zhuanlan.zhihu.com/p/377045943

## 主动学习概念
监督学习问题中，存在标记成本较为昂贵且标记难以大量获取的问题。 针对一些特定任务，只有行业专家才能为样本做上准确标记。在此问题背景下，主动学习（Active Learning, AL）尝试通过选择性的标记较少数据而训练出表现较好的模型。

主动学习最重要的假设是不同样本对于特定任务的重要程度不同，所以带来的表现提升也不全相同。选取较为重要的样本可以使当前模型以较少的标记样本数得到较好的表现。 在这一过程中，主动学习的本质是对样本的重要性（/信息度/期望带来的表现等）等进行评估。绝大多数的工作都是围绕如何评估样本来展开。

## 基础问题场景

这里我们只讨论最基础的分类和回归问题。在这两类问题下，有着三种不同的主动学习场景：

+ Pool-based scenario：此类场景通常提供一个未标记的数据池，主动学习策略在数据池中选取相应样本进行标记。
+ Stream-based scenario：此类场景中，数据以数据流的形式输入，主动学习策略需要确定对当前数据进行标记还是直接用现有模型预测。
+ Query synthesis scenario：此类场景较为少见，一个未标记的数据池通常也被提供，但是主动学习策略并不是在数据池中挑选样本进行查询，而是自行生成新样本进行查询。

## 技术角度分类

+ 设计原理角度：希望读者对不同种策略有一定的认识。
+ 适用模型角度：希望读者可以直接找到适用于自己模型的策略。

### 设计原理分类
此分类针对于Pool-based Classification场景。主动学习的设计原理（主动学习对于样本的评估方法）可以分为五大类：
+ Informativeness：一般来说指模型对选取样本取值的不确信程度，但忽略了数据分布的影响
+ Representativeness-impart：考虑了选取样本是否可以对数据分布起到代表作用，但通常来说和informativeness一起使用，此类方法和批选取方法可能会有重叠
+ Expected Improvements：考虑选取样本能为当前模型带来多少性能提升，但此类评估通常较为耗时
+ Learn to score：不人为启发式地设计选取策略，而是学习一个选取策略	
+ Others：有一些工作较难分类到上述的类别中	

其中的一些小类包括

+ Informativeness
  + Uncertainty-based sampling
  + Disagreement-based sampling
+ Expected Improvements
+ Representativeness-impart sampling
  + Cluster-based sampling
  + Density-based sampling
  + Alignment-based sampling
  + Expected loss on unlabeled data
+ Learn to Score
+ Others

### 从适用模型上分类
有些主动方法仅仅适用于一个或者一类模型，因此按常用的模型分类可以得到
+ SVM/LR
+ Bayesian/Probabilistic Models
+ Gaussian Progress
+ Decision Trees
+ Neural Network