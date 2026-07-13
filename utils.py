import torch
import numpy as np
from sklearn.metrics import f1_score

def IOU(ref, pred):
    assert ref.shape == pred.shape, f'reference and prediction shapes are not the same {ref.shape}, {pred.shape} respectively'
    #print(f"seg_pred: {ref.shape}, truth: {pred.shape}")
    b, h, w = ref.shape
    intersection = torch.sum(ref*pred)
    union = torch.sum((ref + pred)>=1).long()
    int_over_union = ((intersection+1e-6)/(union+1e-6))/b if b>1 else ((intersection+1e-6)/(union+1e-6))
    return int_over_union

def F1(ref, pred):
    if not isinstance(ref, np.ndarray):
        ref = ref.cpu().numpy()
        pred = pred.cpu().numpy()
    return f1_score(y_true=ref.ravel(), y_pred=pred.ravel())


class Accuracy:
    def __init__(self):

        self.iou = []
        self.f1 = []
        self.counter = 0

    def compute_metrics(self, ref, pred):

        iou = IOU(ref=ref, pred=pred)
        self.iou.append(iou.cpu().numpy())

        f1 = F1(ref=ref, pred=pred)
        self.f1.append(f1)
        
        self.counter+=1

    def reset(self):

        self.iou = []
        self.f1 = []
        self.counter = 0

    def get_metrics(self):

        vals = {}
        if self.iou !=[]:
            vals['iou'] = np.sum(self.iou)/self.counter
        if self.f1 != []:
            vals['f1'] = np.sum(self.f1)/self.counter
        
        return vals
