---
layout: post
title: '3D点云检测论文清单'
date: 2021-08-01
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 论文清单
---

**最后更新于2021/8/1，但是还有很多没整理的。**
# Attention

1. **Attentional ShapeContextNet for Point Cloud Recognition(CVPR2018)** [论文链接](http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.html) :  *将Shapecontext的思想引入深度学习。为了进一步简化，利用深度学习的能力，将注意力（transformer）引入，完成selection and aggregation的功能。*
2. **PCAN: 3D Attention Map Learning Using Contextual Information for Point     Cloud Based Retrieval(CVPR2019)** [论文链接](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_PCAN_3D_Attention_Map_Learning_Using_Contextual_Information_for_Point_CVPR_2019_paper.html) : *相当用类似PointNet++中SAG+FP的方法，提取每个点的局部特征，再转换为attention，用于PointNet的点特征→全局特征的加权*


# Detection

1. **What You See is What You Get: Exploiting Visibility for 3D Object Detection(CVPR2020)** [论文链接](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.html): *引入visualbility的概念，考虑到显示中视线的遮挡同样会对点云数据产生相同的遮挡效果，进而对paste数据增强进行一定的限制，并同时作为一种额外的特征加入pointpillar特征中，丰富特征的信息。*
   
2. **Joint 3D Instance Segmentation and Object Detection for Autonomous Driving(CVPR2020)**[论文链接](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Joint_3D_Instance_Segmentation_and_Object_Detection_for_Autonomous_Driving_CVPR_2020_paper.pdf)
*使用使用类似vote的方法，在给出前景点分类的同时，计算每个前景点的偏移，通过聚类得到proposal，同时结合了bbox和segmentation。两部分内容可以相互boost，相当于引入了更多的监督。由于通过聚类得到proposal,因此无需nms，每个聚类（目标）只有一个proposal，效率更高。*

3. **What You See is What You Get: Exploiting Visibility for 3D Object Detection(CVPR2020)**[论文链接](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.html):*在其他目标检测的基础上，根据激光雷达的特性，引入visibility，将其作为一种额外的特征，和pillar特征cat起来，得到增强的特征，以及最后检测性能的提升。同时，利用visibility的信息，可以在gt-paste增强中可以得到更合理的增强结果，更利于训练。*

4. **Attentional PointNet for 3D-Object Detection in Point Clouds(CVPR2019)**[论文链接](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Paigwar_Attentional_PointNet_for_3D-Object_Detection_in_Point_Clouds_CVPRW_2019_paper.pdf):*将整个sence分为多个小块，分别处理。预测方式比较独特，是通过预测刚性变换矩阵（平移旋转）来将小区域内的目标挪到坐标系中心，并根据预测的size得到目标。其中使用了GRU来结合之前的特征来预测当前的刚性变换矩阵。*
   
5. **Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots(ECCV2020)**[论文链接](https://arxiv.org/abs/1912.12791)：*本文利用compositional part-based models的思想，将点云的目标表示为composition of their interior non-empty voxels，将目标的非空体素视为一个spots,选择一个小子集作为hotspots。该方法通过在每个object中选择有限数量的hotspot来balancing不同目标之间的点样本数量。同时，anchor-free没有认为定义的anchor-size，回归会更加困难，将其视为一个regression target imbalance问题，并通过一个soft argmin来解决*

6. **Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection(2021AAAI)**[论文链接](https://arxiv.org/abs/2012.15712v2):*作者提出了一个新的观点，其认为原始点的精确位置并不是高性能3D目标检测所必须的，而粗粒度的体素同样可以提供足够的检测精度。之所以体素方法精度低是因为在检测过程中忽略了3d信息。主要提出了Voxel RoI pooling，类似于PV-RCNN的pooling方法，只是换成了体素。取得了较快的速度和较好的性能。*

7. **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds(CVPR2020)**[论文链接](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf):*重于设计一种内存和计算都高效的网络结构来直接处理大规模3D点云,主要关注高效的采样，以及有效的feature-learner。通过ramdom sample来降低大规模采样的计算复杂度，对于由此带来的信息损失，使用提出的Local feature aggregation来弥补，取得了很好的效果。*

8. **Range Adaptation for 3D Object Detection in LiDAR(ICCV2019)**[论文链接](http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Wang_Range_Adaptation_for_3D_Object_Detection_in_LiDAR_ICCVW_2019_paper.pdf):*激光雷达的数据特点决定了其数据点随着距离的增加而明显的degrades，导致在远处物体处检测的性能。本文提出一种cross-range adaptation，来使得远处的物体得到和近处物体类似的性能。将近距离和远距离的点云视为两个domain，将这个远近点云检测的问题转换为domain adapatation问题。*

9. **Density Based Clustering for 3D Object Detection in Point Clouds(CVPR2020)** [论文链接](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ahmed_Density-Based_Clustering_for_3D_Object_Detection_in_Point_Clouds_CVPR_2020_paper.pdf):*处理不规则点云数据的瓶颈在于处理点的数量。点的数量直接影响了模型的大小和复杂度，以及输出特征的质量。因此本文使用增量的减少点云的数量来解决这个问题。在检测上，使用了无监督的DBSCAN+回归的方式生成proposal，再进一步进行二阶段refine。*

10. **LiDAR R-CNN: An Efficient and Universal 3D Object Detector(CVPR2021)**[论文链接](https://arxiv.org/abs/2103.15297): *本文使用了点处理方法来进行检测。但是发现了一个未被注意到的问题，（在proposal中）单纯的提取点特征，比如PointNet，会使得学习到的点特征忽略size of proposal。本文主要解决这个问题，并且得到了显著的提升。使用Virtual Points来解决Size Ambiguity Problem。*

11. **ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection(CVPR2021)**[论文链接](https://arxiv.org/abs/2103.05346): 首先在source domain上训练，使用作者提出的*random object scaling strategy*来消除source domain bias的负面影响。
之后在target domain上交替执行 updating memory bank 和 curriculum data augmentation实现在target domain上的无监督domain-adaptation。

12. **3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection(CVPR2021)**[论文链接](https://arxiv.org/abs/2012.04355): *提出了一种3D object detection的半监督方法，同时适用于室内和室外场景。利用一个**teach-student mutual learning framework**来在有标注和无标注数据传播信息，通过伪标签的方式。但是由于目标检测任务相对复杂，伪标签可靠性，对此提出了若干filtering标签的方法，得到可靠的伪标签，从而进行训练。*

13. **HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection(CVPR2021)**[论文链接](https://arxiv.org/abs/2104.00902)： *为了结合体素方法的速度优势和点方法的精度优势，在训练时使用两种特征融合，而在推断时，使用一个点特征memory来代替实际的点特征提取，在保证精度的同时提高了速度。同时，在提取特征时加入了空间注意力。*

14. **Pillar-based Object Detection for Autonomous Driving(ECCV2020)**[论文链接](https://arxiv.org/abs/2007.10323v2): *提出一种anchor-free的方法，对每个pillar进行一次预测，而不是每个anchor。同时，本方法融合了MVF和PointPillar方法，将cyclinder view和BEV view结合，形成融合特征，再通过PointPillar预测。由于特征融合需要有一个pillar to points的过程，作者使用双线性插值代替常用的最近邻插值。*
    
15. **An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds(ECCV2020)**[论文链接](https://arxiv.org/abs/2007.12392):* 将LSTM引入3D目标检测，应用到自动驾驶场景下的连续帧检测中。检测融合了3帧的数据，获得了更大的点云范围。并将3D系数卷积融入LSTM，提出3D Sparse Conv LSTM。*
    
16. **Improving 3D Object Detection through Progressive Population Based Augmentation(ECCV2020)**[论文链接](https://arxiv.org/abs/2004.00831): *将3D目标检测中的超参数设置视为一个hyperparameter schedule learning的问题，通过 Progressive Population Based Search来局部的搜索，压缩搜索空间来实现高效的晁灿搜索。*

17. **PC-RGNN: Point Cloud Completion and Graph Neural Network for 3D Object Detection(AAAI2021)**[论文链接](https://arxiv.org/abs/2012.10412): *提出一种proposal点云补全方法，来应对点云的不平衡性。同时提出一种新的图神经网络，来更好的对形状特性进行编码。*

18. **CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud(AAAI2021)**[论文链接](https://arxiv.org/abs/2012.03015):*为了解决单阶段检测器特征和目标框不对齐的问题，提出confidence rectification module。同时，利用类似FPN的结构来提取特征，增强网络的特征提取能力。最后，提出Distance-Variant IoU-Weighted NMS，来得到更好的NMS效果。*

19. **CPCGAN: A Controllable 3D Point Cloud Generative Adversarial Network with Semantic Label Generating(AAAI2021)**[论文链接](https://www.aaai.org/AAAI21Papers/AAAI-4341.YangX.pdf): *一个两阶段的点云补全网络，使用structure gan来生成点云的骨架，再用final gan来将点云密集补全。提出一种Controllable Generation method来控制点云的生成类别。截至阅读论文时，代码还没开源。*

20. **SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud(CVPR2021)** [论文链接](http://openaccess.thecvf.com/content/CVPR2021/html/):*继承CIA-SSD的结构，但是提出了一种teacher-student训练的方法，来强化对数据集中hard sample的学习（大多就是远处的目标），将远近目标视为一个domain adaptation问题。在速度和精度上均有很大优势。*

21. **Sytle-based Point Generator with Adversarial Rendering for Point Cloud Competion(CVPR2021)**[论文链接](https://openaccess.thecvf.com/content/CVPR2021/html/Xie_Style-Based_Point_Generator_With_Adversarial_Rendering_for_Point_Cloud_Completion_CVPR_2021_paper.html):*主要包含channel-attentive EdgeConv, Style-based Point Generator, and Adversarial Point Rendering，分别用于点云的特征提取、生成以及对抗。其通过部分点云的特征来作为生成点云的style-control code,思路值得学习。其通过多视角投影来增强生成点云的视觉质量的思路也比较新颖。*

22. **BoW Pooling: A Plug-and-Play Unit for Feature Aggregation of Point Clouds(AAAI2021)**[论文链接](https://www.aaai.org/AAAI21Papers/AAAI-1163.ZhangX.pdf)：*结合memory和bag of words的思想，通过学习更新memory来更新words，并使用bag of words思想来生成对于点云的特征表示。*

23. **Semantic Consistency Networks for 3D Object Detection(AAAI2021)**：[论文链接](https://www.aaai.org/AAAI21Papers/AAAI-4979.WeiW.pdf): *作者认为目标检测中目标框内的语义分割结果应该和目标框本身的语义属性一致，也就是Semantic Consistency。故通过引入一个语义分割分支，以及对应的Semantic Consistency Loss，使得检测性能有所提升。*

24. **PointConv: Deep Convolutional Networks on 3D Point Clouds(CVPR2019)**[论文链接](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.html): *从连续卷积的数学原理出发，对于点云这种非结构化数据，提出一种真正意义上的点云卷积。同时，对于推导运算结果进行了理论上可以证明的简化，加快了网络运行速度。* 

# Review

1. **Point-cloud based 3D object detection and classification methods for self-driving applications: A survey and taxonomy**[论文链接](https://www.sciencedirect.com/science/article/pii/S1566253520304097): *综述，基于点云的自动驾驶3D目标检测和分类方法。*   
2. **Automotive LiDAR Technology: A Survey**[论文链接](https://ieeexplore.ieee.org/abstract/document/9455394/):*综述，介绍了激光雷达基本原理以及目前的几种技术路线，以及面临的挑战。*

3. **Sensor and Sensor Fusion Technology in Autonomous Vehicles: A Review**[论文链接](https://www.mdpi.com/1424-8220/21/6/2140): *综述，介绍了自动驾驶场景下的若干传感器的原理，作用以及相关产品，主要包括摄像头、毫米波雷达以及激光雷达。*

# Dataset

1. **nuScenes: A multimodal dataset for autonomous driving(CVPR2020)**[论文链接](http://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.html): *nuScenes数据集发表的论文，详细介绍了数据集的情况。*

2. **Scalability in Perception for Autonomous Driving: Waymo Open Dataset(CVPR2020)**[论文链接](http://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.html): *Waymo数据集发表的论文，详细介绍了数据集的情况。*

# Feature(Classification)

1. **Attentional ShapeContextNet for Point Cloud Recognition(CVPR2018)**[论文链接](http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.html): *参考了传统的Shape Context Kernel，由传统的网格划分变为直接由DL计算点和Bins的亲和度，从而构建类似于Shape Context Kernel的特征。后续由进一步引入self-attention提升了性能。*

2. **A Closer Look at Local Aggregation Operators in Point Cloud Analysis(ECCV2020)**[论文链接](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680324.pdf):*本文首先回顾了多重local aggregation operators，并且使用相同的深度残差结构来研究他们的性能。并提出了**Position Pooling(PosPool)**,相比现有的复杂方法具有类似甚至更好的性能。*

3. **PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds(CVPR2021)**[论文链接](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_PAConv_Position_Adaptive_Convolution_With_Dynamic_Kernel_Assembling_on_Point_CVPR_2021_paper.html):*提出一种利用Weight Bank对点进行直接卷积的方法，通过Weight Bank中的多重权重以及自适应选择，来对于点之间的不同相对位置关系（不同的ralationship），来选择其最适合的权重。计算复杂度低，同时也可以更灵活的捕捉点之间的信息。*