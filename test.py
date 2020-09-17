# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:54:43 2020

@author: Liuzp
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from poutyne.framework import Model
from poutyne.framework import Experiment   
from poutyne.framework.metrics import EpochMetric
import torch_modulation_recognition as tmr


class TopKAccuracy(EpochMetric):

    def __init__(self, k: int):
        super(TopKAccuracy, self).__init__()
        self.k = k
        self.acc = None
        self.__name__ = "top_{}_accuracy".format(self.k)
        self.register_buffer("_accuracy", None)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """ Compute the metric """
        self._accuracy = tmr.metrics.accuracy_topk(y_pred, y_true, k=self.k)
        return float(self._accuracy.cpu().numpy().squeeze())

    def get_metric(self) -> float:
        """ Return the float version of the computed metric """
        return self._accuracy.numpy()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mrresnet", choices=["vtcnn", "mrresnet"], help="Model to train")
parser.add_argument("--epochs", type=int, default=5, help="Epochs to train for")
parser.add_argument("--batch_size", type=int, default=512, help="Number of samples in each batch (set lower to reduce CUDA memory used")
parser.add_argument("--split", type=float, default=0.9, help="Percentage of data in train set")
args = parser.parse_args()
# Modulation types
MODULATIONS = ["1_QPSK","2_8PSK","3_AM-DSB","4_QAM16","5_GFSK","6_QAM64","7_PAM4",
               "8_CPFSK","9_BPSK","10_WBFM"]
# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Params
N_CLASSES = 10
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SPLIT = args.split
DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
LOG_DIR = os.path.join("logs", args.model)

# Load dataset
dataset = tmr.data.RadioML2016()
sX,sy=dataset.load_data()


# Load model
print("Loading Model")   
if args.model == "vtcnn":
    net= tmr.models.VT_CNN2(
        n_classes=N_CLASSES,
        dropout=DROPOUT,
    )
elif args.model == "mrresnet":
    net = tmr.models.MRResNet(
        n_channels=2,
        n_classes=N_CLASSES,
        n_res_blocks=8,
        n_filters=32
    )
   
# Metrics
top3 = TopKAccuracy(k=3)
top5 = TopKAccuracy(k=5)
metrics = ["acc", top3, top5]

model = Model(
    network=net,
    optimizer="Adam",
    loss_function=nn.CrossEntropyLoss(),
    batch_metrics=metrics
)

    ###################################################
    
checkname=os.path.join(MODEL_DIR+'/'+args.model+'.pth')
model.load_optimizer_state(checkname)#get the checkpoint model to predict
idx=109000
Singledata=sX[idx-1:idx]
idy=sy[idx-1:idx]    
s1=torch.FloatTensor(Singledata).squeeze(0)####important

if args.model=="vtcnn":
    output=model.predict(s1)
    print(MODULATIONS[int(np.argmax(output,axis = 1))])
    print("The true type is {} from vtcnn".format(sy[idx][0]))
    
    
    ##########realize the single sample prediction############
if args.model=="mrresnet":
    
    outmodel=model.load_weights(checkname)## it seems not work

   

    
################################################################################

