---
layout: post
title: '环境配置笔记'
date: 2021-05-19
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 环境配置
---

> 本文档用于记录环境配置中遇到的各种问题，遇到问题并解决了就会更新到这上面来。

# 3D pointcloud detection

1、在配置一些cpp extension中遇到的问题，常见的工具包比如 iou3d,pointnet2,points_op等等。

  - 这些问题一般由cuda版本引起由于新版cuda有些函数名发生了变化。我遇到的错误主要有以下两个：

     1. AT_CHEAK函数报错，需要替换成TORCH_CHEAK

     2. THCState_getCurrentStream(state)函数报错，需要替换成at::cuda::getCurrentCUDAStream()
     3. 编译程序读不到cudnn version，因此判断没有cudnn环境，报错：需要cudnn 7以上版本(大概意思)。通过查看源码可以看到，源码所寻找的文件是cudnn.h和cudnn_version.h。在旧版本中，cudnn的版本信息保存在cudnn.h中，而新版本中，cudnn.h中没有包含任何具体信息，而只是include了其他的头文件，而版本信息存在cudnn_version.h中。而在cudnn安装配置中可能并没有复制cudnn_version.h文件到cuda路径中。因此只需要手动复制过去就可以了。 

# 2D detection

1. coco数据集所使用的工具包pycocotools，实际上已经多年没有维护。因此，mmdection项目对其进行了升级，用于mmdection中。更新后的包以如下方式安装
   ```
   pip install mmpycocotools
   ```
    需要注意的是，两者在python中的调用均为
    ```
    import pycocotools
    ```
    因此如果不更新，往往不会提示不能Import这个包，而是出现一些其他的错误。

2. mmdetection: RuntimeError: nms is not compiled with GPU support

安装配置mmdetection2.0后运行出现错误：RuntimeError: nms is not compiled with GPU support，出现错误的原因是编译过mmdetection后又重装了pytorch和torchvision所以需要重新编译mmdetection并且编译之前记得将mmdetection下的build文件删除

实际是因为登陆节点没有gpu，所以mmcv安装的时候没有用cuda编译。需要到计算节点去编译。

注意，如果已经安装过mmcv的，会在~/.cache/pip中留下缓存，这使得下一次安装不再进行重新编译。这样即使进入了gpu环境安装也不行，记得把缓存删了。

多卡训练需要用mmdet提供的脚本，而不能直接使用train.py进行。