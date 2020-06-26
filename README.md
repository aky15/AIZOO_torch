
###  目录


​		**–** data_split.sh 划分训练集、验证集、测试集



​		**–**  train_faster_rcnn.py 训练Faster R-CNN模型所用代码

​		**–**  evaluate_faster_rcnn.py 测试Faster R-CNN模型所用代码

​		**–** faster_rcnn_utils 训练、测试Faster R-CNN的依赖



​		**–** train_retinanet.py 训练retinanet 模型所用的代码

​		**–** evaluate_retinanet.py 训练retinanet 模型所用的代码

​		**–** retinanet 训练、测试retinanet的依赖

​		**–** voc2coco.py 进行数据格式的转换，用于retinanet测试



​		**–** train_yolo.py 训练yolov3 模型所用的代码

​		**–** evaluate_yolo.py 测试yolov3 模型所用的代码

​		**–** yolo_utils 训练、测试yolov3的依赖

​		**–** config yolov3的模型定义



### 运行代码所需的软件环境和软件版本

1. 操作系统：CentOS Linux 7

2. python3：推荐使用anaconda版本的python，其中已包含numpy等常用工具包。安装过程请参考https://www.anaconda.com/products/individual

	3. torch 1.3版本:  conda install pytorch=1.3.0 -c pytorch
 	4. torchvision:  conda install torchvision -c pytorch

5. tqdm: 用于查看训练及测试进度 conda install -c conda-forge tqdm 
6. matplotlib: 绘图工具，用于进行可视化 conda install matplotlib

 

### 下载原始训练测试数据和整理数据的命令

​	原始训练测试数据下载：

​	wget -c  https://cloud.tsinghua.edu.cn/d/af356cf803894d65b447/files/?p=%2FAIZOO%2F%E4%BA%BA%E8%84%B8%E5%8F%A3%E7%BD%A9%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86.zip

​	下载并解压后，使用./data_split.sh完成训练、开发、测试集的划分。



### **–** 训练所提交的最优模型的命令

​	我们得到的最优模型为retinanet，其训练命令为

​	`python train_retinanet.py`



### **–** 用上一步训练出来的模型在十张图片上测试的命令

​	`python evaluate_retinanet.py`



### **–** 用提交的最优模型在十张图片上测试的命令，以及预期的结果

   最优模型下载链接

​	https://cloud.tsinghua.edu.cn/f/c207ce51863841249616/?dl=1

### 执行命令

```python
wget -c https://cloud.tsinghua.edu.cn/f/c207ce51863841249616/?dl=1
mv index.html?dl=1 model_final.pt
python voc2coco.py test/Annotations ./test/annotation.json
python evaluate_retinanet.py
```

### 预期结果

face  mAP @ 0.5: 0.8181818181818182

face mask mAP @ 0.5: 1.0

face  mAP @ 0.7: 0.8181818181818182

face mask mAP @ 0.7: 1.0

face  mAP @ 0.9: 0.45454545454545453

face mask mAP @ 0.9: 0.303030303030303

face mean mAP @[.5:.95] : 0.6954545454545454

face mask mean mAP @[.5:.95]: 0.743030303030303

### *refernce*
https://pytorch.org/docs/master/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/yhenon/pytorch-retinanet
