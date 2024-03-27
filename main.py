import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset

from neurobench.datasets import MSWC
from neurobench.datasets.MSWC_IncrementalLoader import IncrementalFewShot
from neurobench.models import TorchModel

from neurobench.benchmarks import Benchmark
from neurobench.preprocessing import MFCCPreProcessor


from tcn_lib import TCN

squeeze = lambda x: (x[0].squeeze(), x[1])
out2pred = lambda x: torch.argmax(x, dim=-1)
to_device = lambda x: (x[0].to(device), x[1].to(device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda"):
    PIN_MEMORY = True
else:
    PIN_MEMORY = False

fscil_directory = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(fscil_directory, "model_data/")
ROOT = os.path.join(fscil_directory, "./data/")  # data in repo root dir
NUM_WORKERS = 8 if device == torch.device("cuda") else 0
BATCH_SIZE = 64
NUM_REPEATS = 1
PRE_TRAIN = True
NUM_SHOTS = 5
EPOCHS = 50 # if pre-training from scratch

# Define MFCC pre-processing
n_fft = 512
win_length = None
hop_length = 120
n_mels = 20
n_mfcc = 20

encode = MFCCPreProcessor(
    sample_rate=48000,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
        "f_min": 20,
        "f_max": 4000,
    },
    device = device
    )


def test(test_model, mask, Test):
    test_model.eval()
    test_loader = Test

    out_mask = lambda x: x - mask
    with torch.no_grad():
        benchmark = Benchmark(TorchModel(test_model), metric_list=[[], ["classification_accuracy"]], dataloader=test_loader,
                            preprocessors=[to_device, encode, squeeze],
                            postprocessors=[out_mask, out2pred, torch.squeeze])

        pre_train_results = benchmark.run()
        test_accuracy = pre_train_results['classification_accuracy']

    return test_accuracy


def pre_train(model):
    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY)
    test1 = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
    test2 = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    mask = torch.full((200,), float('inf')).to(device)
    mask[torch.arange(0,100, dtype=int)] = 0

    lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}")

        # model.train()

        for _, (data, target) in tqdm(enumerate(pre_train_loader), total=len(base_train_set)//BATCH_SIZE):
            data = data.to(device)
            target = target.to(device)

            # print(data.shape, target.shape)

            # apply transform and model on whole batch directly on device
            data, target = encode((data, target))
            # print(data.shape, target.shape)

            output = model(data.squeeze())
            print(output.shape)
            print(target.shape)
            print(data.shape)
            loss = F.cross_entropy(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_acc = test(model, mask, test1)
            test_acc = test(model, mask, test2)

            print(f"The train accuracy is {train_acc*100}%")
            print(f"The test accuracy is {test_acc*100}%")

        scheduler.step()

    del base_train_set
    del pre_train_loader


if __name__ == '__main__':
    print(fscil_directory)
    if PRE_TRAIN:
        # Using same parameters as provided pre-trained models
        model = TCN(20, 200, [16] * 2, [5] * 1 + [7] * 1).to(device)

        pre_train(model)

        model = TorchModel(model)