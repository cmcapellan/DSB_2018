import os
import numpy as np
import tensorflow as tf
from my_bowl_dataset import BowlDataset

print(tf.__version__)

ROOT_DIR = "/root"
TRAIN_SPLIT = 0.8

stage1_dir = os.path.join(ROOT_DIR,'stage1_train')
stage1_ids = next(os.walk(stage1_dir))[1]
stage1_ids = [os.path.join(stage1_dir, stage1_ids) for stage1_ids in stage1_ids]

train_ids = stage1_ids[:int(len(stage1_ids) * TRAIN_SPLIT)]
val_ids = stage1_ids[int(len(stage1_ids) * TRAIN_SPLIT):]

# Training and validation dataset
dataset_train = BowlDataset()
dataset_train.load_bowl(train_ids)
dataset_train.prepare()

dataset_val = BowlDataset()
dataset_val.load_bowl(val_ids)
dataset_val.prepare()