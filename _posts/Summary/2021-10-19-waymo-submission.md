---
layout: post
title: 'Waymo数据集提交格式处理方法'
date: 2021-10-19
author: Poley
cover: '/assets/img/mycat3.jpg'
tags: 总结
---

和KITTI数据集只需要简单的提交TXT数据结果就可以进行评估不同，Waymo数据集的在线提交需要更复杂的包装。这个笔者花了一天时间来实现最终的结果提交。特在此记录，方便以后查阅。

首先，对于程序运行过程中产生的目标结果，先要将其打包成waymo要求的格式，并逐frame的保存成.bin文件。Waymo要求的文件格式如下，需要将单个object包装成 ```metrics_pb2.Object()```格式，并压入一个列表```metrics_pb2.Objects()```中，写入bin文件。

```
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


def _create_pd_file_example():
  """Creates a prediction objects file."""
  objects = metrics_pb2.Objects()

  o = metrics_pb2.Object()
  # The following 3 fields are used to uniquely identify a frame a prediction
  # is predicted at. Make sure you set them to values exactly the same as what
  # we provided in the raw data. Otherwise your prediction is considered as a
  # false negative.
  o.context_name = ('context_name for the prediction. See Frame::context::name '
                    'in  dataset.proto.')
  # The frame timestamp for the prediction. See Frame::timestamp_micros in
  # dataset.proto.
  invalid_ts = -1
  o.frame_timestamp_micros = invalid_ts
  # This is only needed for 2D detection or tracking tasks.
  # Set it to the camera name the prediction is for.
  o.camera_name = dataset_pb2.CameraName.FRONT

  # Populating box and score.
  box = label_pb2.Label.Box()
  box.center_x = 0
  box.center_y = 0
  box.center_z = 0
  box.length = 0
  box.width = 0
  box.height = 0
  box.heading = 0
  o.object.box.CopyFrom(box)
  # This must be within [0.0, 1.0]. It is better to filter those boxes with
  # small scores to speed up metrics computation.
  o.score = 0.5
  # For tracking, this must be set and it must be unique for each tracked
  # sequence.
  o.object.id = 'unique object tracking ID'
  # Use correct type.
  o.object.type = label_pb2.Label.TYPE_PEDESTRIAN

  objects.objects.append(o)

  # Add more objects. Note that a reasonable detector should limit its maximum
  # number of boxes predicted per frame. A reasonable value is around 400. A
  # huge number of boxes can slow down metrics computation.

  # Write objects to a file.
  f = open('/tmp/your_preds.bin', 'wb')
  f.write(objects.SerializeToString())
  f.close()


def main():
  _create_pd_file_example()


if __name__ == '__main__':
  main()
```

具体的代码实现可以参考

https://github.com/caizhongang/waymo_kitti_converter

这个代码实现了KITTI和Waymo数据，标注文件以及预测结果格式的相互转换，对于了解两个数据集的格式差异非常方便。

通过上述Python代码，可以将其全部预测结果打包为一个Bin文件。之后还需要通过waymo官方toolkit的处理，转换为可以提交的格式。

https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md

这部分可以参考上述链接。首先需要下载waymo-open-dataset的源码，并使用Bazel在本地进行编译。

之后使用编译后的文件（自动生成在bazel-bin文件夹下），执行如下操作即可生成最后的提交文件。

```
metrics/tools/create_submission  --input_filenames='metrics/tools/fake_predictions.bin' --output_filename='/tmp/my_model/model' --submission_filename='metrics/tools/submission.txtpb'
```

其中```submission.txtpb```文件包含了测评的基本设置信息以及账号信息等，邮箱需要和自己登陆waymo open dataset网站的邮箱一致。 通过上述方法若干个最终提交文件，并压缩为```tar.gz```文件即可

```
tar cvf /tmp/my_model.tar /tmp/my_model/
gzip /tmp/my_model.tar
```