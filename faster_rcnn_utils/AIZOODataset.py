import torch
import os
from PIL import Image
from xml.dom.minidom import parse
import numpy as np
from torchvision.transforms import functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class AIZOODataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")        
        
        # 读取文件，VOC格式的数据集的标注是xml格式的文件
        dom = parse(bbox_xml_path)
        # 获取文档元素对象
        data = dom.documentElement
        # 获取 objects
        objects = data.getElementsByTagName('object')        
        # get bounding box coordinates
        boxes = []
        labels = []
        names = []
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label，face或face_mask
            labels.append(int(name == 'face_mask')+1)  # 背景的label是0，face和face_mask的label分别是1和2
            #names.append(name)
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)        
        #names = torch.as_tensor(names,dtype=torch.int64)
        image_id = torch.tensor([idx])
        #print(img_path)
        #print(boxes)
        #print(boxes.size())
        '''
        if boxes.shape != torch.Size([0]):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor(0)
        '''
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        #target["name"] = names
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)
        return img, target
 
    def __len__(self):
        return len(self.imgs)
