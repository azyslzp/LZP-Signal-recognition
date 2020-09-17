import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from poutyne.framework import Model
from poutyne.framework.callbacks import TensorBoardLogger, \
    ModelCheckpoint, lr_scheduler,\
    LRSchedulerCheckpoint,OptimizerCheckpoint
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
parser.add_argument("--epochs", type=int, default=70, help="Epochs to train for")
parser.add_argument("--batch_size", type=int, default=512, help="Number of samples in each batch (set lower to reduce CUDA memory used")
parser.add_argument("--split", type=float, default=0.85, help="Percentage of data in train set")
parser.add_argument("--resume",type=str,default="None",choices=["vtcnn", "mrresnet","None"])
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
   
# Load dataset
dataset = tmr.data.RadioML2016()
#sX,sy=dataset.load_data()
# Split into train/val sets/test_set
total = len(dataset)
lengths1 = [int(len(dataset)*SPLIT)]
lengths1.append(total - lengths1[0])
print("Splitting into {} train and {} val ".format(lengths1[0], lengths1[1]))
train_set, val_set = random_split(dataset, lengths1)

# Setup dataloaders
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Callbacks
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
checkpoint = ModelCheckpoint(
    filename=os.path.join(MODEL_DIR, args.model + ".pth"),
    monitor="val_loss",
    save_best_only=True,
    temporary_filename=os.path.join(MODEL_DIR, args.model + "tmp.pth")
)
writer = SummaryWriter(LOG_DIR)
tb_logger = TensorBoardLogger(writer)
slr=lr_scheduler.StepLR(step_size=40)
callbacks = [checkpoint, tb_logger,slr]

# Metrics
top3 = TopKAccuracy(k=3)
top5 = TopKAccuracy(k=5)
metrics = ["acc", top3, top5]
####define the model
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
model = Model(
    network=net,
    optimizer="Adam",
    loss_function=nn.CrossEntropyLoss(),
    batch_metrics=metrics
)


if args.resume=="vtcnn" or args.resume=="mrresnet":
    checkdir=os.path.join(MODEL_DIR+args.model+'.pth')
    load_model=torch.load(checkdir)

#    exp = Experiment(checkdir,
#                     net,
#                     device=DEVICE,
#                     optimizer='Adam',
#                     loss_function=nn.CrossEntropyLoss(),
#                     task='classif',
#                     batch_metrics=metrics,
#                     monitor_metric ="val_loss",
#                     )
    
#    weights = torch.randn((1),requires_grad=True)
#    optimizer = optim.Adam([weights],lr=0.01)

    
    
#    exp.train(train_dataloader,val_dataloader,
#              lr_schedulers=slr,epochs=EPOCHS)

if args.resume=="None":
    # Load model

    # Train
    model = Model(
        network=net,
        optimizer="Adam",
        loss_function=nn.CrossEntropyLoss(),
        batch_metrics=metrics
    )
    
    
    
    model.cuda()
    model.fit_generator(
        train_dataloader,
        val_dataloader,
        epochs=EPOCHS,
        initial_epoch=1,
        callbacks=callbacks
    )
