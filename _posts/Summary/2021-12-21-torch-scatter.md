---
layout: post
title: 'torch-scatter'
date: 2021-12-21
author: Poley
cover: '/assets/img/mycat3.jpg'
tags: 总结
---
点云中往往需要用到点云和体素之间的映射与整合。比如在计算完每个点的体素坐标之后，将输入相同体素的点特征整合到一起，再继续进行后续操作，比如mean,max,pointnet等等。一般来说，在Openpcdet中，由于体素分配使用的是hard方法，即
+ 限定了最大体素数
+ 限定了体素内的最大点数
因此相当于使用一个固定大小的预分配空间来完成点到体素的分配。同时由于这里体素化的过程不涉及任何的运算，因此在OpenPCDet中，这部分工作是在预处理阶段进行的，即不在torch中，而是通过spconv模块来实现。这样即体素化的过程也不能参与到网络的求导中（不能把体素化放在网络过程的中间）

Pillar-od方法就需要这种操作，其先分别在BEV和CYLINDER view上通过卷积提取特征，再重新插值变为点特征，再进一步体素化变成Pilllar或Voxel特征，这就需要点云→体素的过程可导。即将点云特征分配到体素的过程可导。


首先我们可以通过一些现成的方法计算出每个点云对应的体素坐标。此时点云的个数是不确定的，非空体素的数量和每个体素中点的数量也是不确定的。为了包含所有的点，我们尽量使用动态体素化的过程（Dynamic Voxelization），这样可以充分利用每个点的特征，并且在优化过程中也可以利用到所有点特征的梯度。

在已知点坐标和对应的体素坐标时，我们可以使用torch的scatter函数来完成动态的体素分配。值得注意的是，上述所提到的所有操作在实际运行时，由于动态体素化情况下，每次点云输入的尺寸以及对应的体素尺寸都不同，因此，很关键的一点是在进行这些处理的时候需要关闭cudnn，因为cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题：如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

torch的scatter函数介绍如下所示，其实实现动态体素等方法的关键所在。其文档地址为[totch_scatter.scatter](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)。其具有多种不同的rudece方式，和体素化的过程不谋而合。体素化最常用的reduce方式是mean，通过此函数，可以省略点特征整合的步骤，直接生成体素特征。

在已知点特征和体素坐标的前提下，实现动态体素的代码实例如下（笔者实现），主要思想是先通过unique确定非空体素的数量以及建立点到非空体素索引的映射，之后通过scatter直接像对应的索引分配点特征并完成reduce，得到最终的体素特征:
```
voxel_coords_list = []
voxel_feature_list = []
voxel_num_points_list = []
for i in range(xyz_view_voxels['coords'].shape[0]):

    scatter_coord = xyz_view_voxels['coords'][i][torch.logical_not(xyz_view_voxels['paddings'][i]),:]
    sample_voxel_coord, sample_voxel_scatter_index, sample_points_per_sample = \
        scatter_coord.unique(dim=0, return_counts=True, return_inverse=True)
    sample_valid_point_features = x[i][torch.logical_not(xyz_view_voxels['paddings'][i]),:]
    sample_voxels = scatter(sample_valid_point_features,sample_voxel_scatter_index,dim=0,reduce='mean')
    batch_id = i * torch.ones_like(sample_points_per_sample).unsqueeze(1).cuda()
    voxel_coords_list.append(torch.cat([batch_id,sample_voxel_coord],dim=1))
    voxel_feature_list.append(sample_voxels)
    voxel_num_points_list.append(sample_points_per_sample)
voxel_coords = torch.cat(voxel_coords_list,dim = 0)[:,[0,3,2,1]]

voxel_num_points = torch.cat(voxel_num_points_list,dim = 0)
voxel_features = torch.cat(voxel_feature_list,dim = 0)
```

PS：torch的gather和scatter是两个作用相反，用法类似的函数。两者应该都经常用于点云检测中。可以有效提高效率。