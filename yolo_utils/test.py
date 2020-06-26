from __future__ import division

from .models import *
from .utils import *
from .datasets import *
from .parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def pascal_voc(curve):
   ap = []
   for recthrs in np.arange(0,1.1,0.1):
       tmp = curve[1,curve[0,:]>=recthrs]
       if tmp.size == 0:
           ap.append(0)
       else:
           ap.append(curve[1,curve[0,:]>=recthrs ].max())
   ap = np.array(ap)
   return ap.mean()

   
def evaluate(model, img_path,label_path,list_path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(img_path,label_path,list_path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        #print(outputs)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    curve,precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    pascal_voc_map_face = -1
    pascal_voc_map_facemask = -1
    face_curve = curve[0]
    pascal_voc_map_face = pascal_voc(face_curve)
    np.savetxt("face_curve_{}.txt".format(iou_thres),face_curve)
    if len(curve)>1:
        face_mask_curve = curve[1]
        pascal_voc_map_facemask = pascal_voc(face_mask_curve)
        np.savetxt("facemask_curve_{}.txt".format(iou_thres),face_mask_curve)
    return pascal_voc_map_face,pascal_voc_map_facemask,precision, recall, AP, f1, ap_class


