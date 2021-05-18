---
layout: post
title: 'Faster RCNN 源码阅读'
date: 2021-05-18
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 目标检测，深度学习，FasterRCNN
---

> 代码参考 https://github.com/chenyuntc/simple-faster-rcnn-pytorch

# 模型结构实现

网络结构非常简单，使用的是一个VGG16的backbone，特征提取层对应16倍降采样。之后一个rpn做objectness和bbox的预测，得到roi。输入roi head，预测得到最终的结果。

具体模块在 **/model/** 中。
# 训练
## Loss

上述网络结构都比较简单，主要关注一下源码中的loss是如何产生的。主要是如何通过gt获得训练使用的target。训练部分内容在源码 **/trainer.py**中。

具体来关注一下 **Faster RCNN中这些loss对应的target是如何产生的，以及roi等的具体传递方式** 


可以看到，**FasterRCNNTrainer**的forward方法如下

```python
    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        #rpn偏移量，得分，roi(角坐标)，roi对应的图片(在batch中的序号)，anchor
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # 这里指定了batchsize为 1
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input

        # 返回值 roi 按指定数量和比例采样的正负样本，对应的真值偏移，以及他们的label。
        # 这里计算出来的gt是针对roi的偏移和标签，即这是对二阶段回归和分类的标签
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#

        #这是对一阶段回归的分类和标签
        #这里使用的是全部anchor和对应的gt来做损失。
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
```

可以看到，其主要流程是：

1. 通过backbone提取单层特征，这很简单，没什么好说的
```python
    features = self.faster_rcnn.extractor(imgs)
```

2. 通过rpn提取rpn_map的偏移量，得分，roi，roi对应的图片序号（batch），以及anchor。
```python
rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)
```

3. 将roi输入head中，进行二次回归。其中roi_pooling也包含在这部分中。
```python
roi_cls_loc, roi_score = self.faster_rcnn.head(
        features,
        sample_roi,
        sample_roi_index)
```
4. 计算分类和回归用的标签，分为两个部分，一部分是rpn的，一部分是head的。

```python
#这是对一阶段回归的分类和标签
#这里使用的是全部anchor和对应的gt来做损失。
gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)

# 返回值 roi 按指定数量和比例采样的正负样本，对应的真值偏移，以及他们的label。
# 这里计算出来的gt是针对roi的偏移和标签，即这是对二阶段回归和分类的标签
sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
```

5. 计算损失，这部分很简单，两个交叉熵用于分类，两个smoothL1用于回归。

首先，解析一下 **rpn** 模块。rpn模块中包括了anchor的生成以及对应偏移量的计算。其在文件 **/model/region_proposal_network.py** 中。

看一下他的forward函数
```python
    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        #每个cell处anchor的数量
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        #计算softmax分类结果
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)

        #提取前景得分
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()

        #对每张图的roi进行筛选采样，选取指定数量的roi，再经过nms，保留指定数量的roi
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        #返回值依次是
        #rpn偏移量，得分，roi(角坐标)，roi对应的图片(在batch中的序号)，anchor
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
```

主要分为以下步骤

1. 建立anchor：通过**_enumerate_shifted_anchor**函数建立，之后将其展开为二维数组返回。这部分相对简单，如下。anchor_based代表一个cell上的多重anchor大小信息，依次加上偏移就得到最终的anchor信息。

```python
def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)

    #注意顺序，这里需要以行优先展开，这样才能和网络预测结果的展开方式对应上
    #因为网络预测结果的大小是[batch,h,w,A,4],因此展开时会先按w展开，再按h展开
    #即先展开横坐标x（列）,再展开纵坐标y（行），因此这里也是同样的方式，先展开x，再展开y
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
```

值得注意的是，这里的x y 展开顺序，**shift_x, shift_y = xp.meshgrid(shift_x, shift_y)**。**一定是先展开x坐标，即矩阵的列坐标，再展开y坐标，即矩阵的行坐标。这样是按行展开的模式，才可以和网络预测的结果展开后（torch都是行优先展开）的结果对应。**

2. 生成网络的预测结果，并展开成二维数组形式。预测结果包括rpn_locs偏移量，以及rpn_scores得分，并通过softmax函数判断前景置信度。
3. 对batch中的每张图片，产生proposal

```python
#对每张图的roi进行筛选采样，选取指定数量的roi，再经过nms，保留指定数量的roi
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
```

源码中，我们可以找到proposal_layer对应的是**ProposalCreator**类，__call__方法代码如下

```python
    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)

        #偏移转换为 ymin xmin ymax xmax
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        # 除去过小的roi
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).

        #选择分数最大N个roi,而不是用阈值判断。
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms] #在nms后保留至多指定数量的roi


        roi = roi[keep.cpu().numpy()]
        return roi
```

其具体内容就是：

   + 将roi的 xywh格式转换为ymin xmin ymax xmax格式，并筛除大小太小的roi；
   + 将rois的置信度排序，选择前N个rois，这个数量由**n_pre_nms**决定，程序中设定的是12000；
   + 将这些rois做nms，之后再保留至多**n_post_nms**个roi，这里程序中设定的是6000；
   + 返回上述经过筛选的rois。

4.返回这些proposal。


了解完rpn模块，再继续往下看，下一步是head模块。这个没有什么复杂的，因为其只是接受了rois，和其对应的indece，在对应的图片的feature_map上做roi_pooling，然后经过一个head网络输出即可。

之后来看一下rpn和 head的target都是如何产生的。rpn的target的生成在程序中是通过**anchor_target_creator()**实现的。其**__call__()**比较简单，核心方法是**_create_label()，_cal_ious()**

```python
    def _create_label(self, inside_index, anchor, bbox):
        #此时的anchor已经是和inside_index一样长度的了，即已经筛除了越界的anchor
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0  #iou小于阈值的设为负样本

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1   #最匹配的anchor设为正样本

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1  #其余大于阈值的anchors也设为正样本

        #正负样本按比例采样
        #具体方法，再上述已经给定的标签基础上，将不采样的数据设为-1 don't care
        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
    # 此时的anchor已经是和inside_index一样长度的了，即已经筛除了越界的anchor
    # ious between the anchors and the gt boxes
    ious = bbox_iou(anchor, bbox)
    argmax_ious = ious.argmax(axis=1) #找到每个anchor iou最大的gt box
    max_ious = ious[np.arange(len(inside_index)), argmax_ious] #得到每个anchor对应的最大iou
    gt_argmax_ious = ious.argmax(axis=0) #得到每个gt最匹配的anchor
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] #gt最匹配anchor的iou
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]  #得到相互最匹配的gt anchor对

    return argmax_ious, max_ious, gt_argmax_ious
```

其功能在上述注释中已经写得比较详细，分配原则如下：
   + 计算anchor和gt_bbox之间的iou，分别得到
     + 每个anchor对应的iou最大的gt_bbox序号，**argmax_ious**
     + 上一条对应的iou，**max_ious**
     + 每个gt_bbox对应的iou最大的anchor序号，**gt_argmax_ious**
     + 上一条对应的iou，**gt_argmax_ious**
   + 为anchor分配class label和target，原则如下
     + 每个gt_bbox至少有一个anchor跟它对应，即和gt_bbox重叠最大的那个。
     + 每个anchor都被划分到与其iou最大的gt_bbox上，iou小于阈值则为负，大于为正
     + 根据上一条的对应关系计算偏差 xywh
     + 根据事先给定的采样数量和正负样本比例，从上述样本中随机抽取指定数量和比例的正负样本。其余样本的label全部设置为-1(don't care)

这就完成了rpn target的计算，之后直接套用损失函数即可。

head的target则使用**ProposalTargetCreator()**类来实现，其原理和上面类似，但是更简单。同样对每个rois都通过iou和一个gt_bbox对应起来，再根据iou判断pos or neg 以及根据对应关系计算偏差即可。

```python
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        ROI被赋予真值，包括gt clasee和boounding box的offset 和scale来匹配gt bounding box。
        pos_ratio * self.n_sample个roi会被采样为正样本。

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.


        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)  # [R,R']
        gt_assignment = iou.argmax(axis=1)  #每个roi仅分配一个真值，即和其roi最大的gtbox关联起来。
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1  #得到每个roi对应的类别

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]  #选择最大iou大于threshold的roi作为正样本
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size)) #如果pos数量没有预定的采样数量多，则修改采样数量
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]  #筛选负样本，其中设置了负样本iou的上限和下线
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image #决定负样本的采样数量
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)               #cat两者的Index
        gt_roi_label = gt_roi_label[keep_index]                    #得到对应的类别
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0 负样本标签更改为背景
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])  #计算target
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))   #对计算出来的偏移量再做一次归一化

        # 返回值 roi 按指定数量和比例采样的正负样本，对应的真值偏移，以及他们的label。
        return sample_roi, gt_roi_loc, gt_roi_label
```

之后做正反传播就可以训练网络了！代码解析接到这里。

