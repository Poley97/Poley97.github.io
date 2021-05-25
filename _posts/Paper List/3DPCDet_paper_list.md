---
layout: post
title: '3D点云检测论文清单'
date: 2021-05-20
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 论文清单
---


# Attention

1. **Attentional ShapeContextNet for Point Cloud Recognition(CVPR2018)** [论文链接](http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.html) :  *将Shapecontext的思想引入深度学习。为了进一步简化，利用深度学习的能力，将注意力（transformer）引入，完成selection and aggregation的功能。*
2. **PCAN: 3D Attention Map Learning Using Contextual Information for Point
Cloud Based Retrieval(CVPR2019)** [论文链接](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_PCAN_3D_Attention_Map_Learning_Using_Contextual_Information_for_Point_CVPR_2019_paper.html) : *相当用类似PointNet++中SAG+FP的方法，提取每个点的局部特征，再转换为attention，用于PointNet的点特征→全局特征的加权*


# Detection

1. **What You See is What You Get: Exploiting Visibility for 3D Object Detection(CVPR2020)** [论文链接](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.html): *引入visualbility的概念，考虑到显示中视线的遮挡同样会对点云数据产生相同的遮挡效果，进而对paste数据增强进行一定的限制，并同时作为一种额外的特征加入pointpillar特征中，丰富特征的信息。*
2. 