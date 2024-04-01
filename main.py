import os
import json

import numpy as np

from tcn_lib import stats
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
print("Device: ",device)

if device == torch.device("cuda"):
    PIN_MEMORY = True
else:
    PIN_MEMORY = False

fscil_directory = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(fscil_directory, "model_data/")
ROOT = os.path.join(fscil_directory, "data/")  # data in repo root dir
NUM_WORKERS = 16 if device == torch.device("cuda") else 0
PREFETCH_FACTOR = 4
BATCH_SIZE = 256
PRE_TRAIN = True
NUM_SHOTS = 5 # How many shots to use for evaluation
EPOCHS = 50 # if pre-training from scratch

# Define MFCC pre-processing
n_fft = 512
win_length = None
hop_length = 240
n_mels = 20
n_mfcc = 20


class NeuroTester:
    def __init__(self, model) -> None:
        self.encode = MFCCPreProcessor(
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
        self.model = model
        self.is_neurobench_used = True
    
    @property
    def neurobench_on(self) -> bool:
        return self.is_neurobench_used
    
    def diy_test(self, mask, test_loader, test_set):
        print("Benchmark")
        self.model.eval()
        """Evaluate accuracy of a model on the given data set."""

        # init
        acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
        n = 0

        for _, (X, y) in tqdm(enumerate(test_loader), total=len(test_set)//BATCH_SIZE):
            # Copy the data to device.
            X, y = X.to(device), y.to(device)
            X, y = self.encode((X, y))
            with torch.no_grad():
                y = y.long()
                acc_sum += torch.sum((torch.argmax(self.model(X.squeeze()), dim=-1) == y))
                n += y.shape[0]  # increases with the number of samples in the batch
        print(acc_sum)
        return acc_sum.item() / n
    
    def neurobench(self, mask, test_loader, test_set):
        self.model.eval()
        out_mask = lambda x: x - mask
        with torch.no_grad():
            benchmark = Benchmark(TorchModel(self.model), metric_list=[[], ["classification_accuracy"]], dataloader=test_loader,
                                preprocessors=[to_device, self.encode, squeeze],
                                postprocessors=[out_mask, out2pred, torch.squeeze])

            pre_train_results = benchmark.run()
            test_accuracy = pre_train_results['classification_accuracy']

        return test_accuracy
    
    def test(self, mask, test_loader, test_set):
        if self.neurobench_on:
            self.neurobench(mask, test_loader, test_set)
        else:
            self.diy_test(mask, test_loader, test_set)

    def load(self, model):
        state_dict = torch.load(os.path.join(MODEL_SAVE_DIR, "mswc_rsnn_proto"),
                            map_location=device)
        model.load_state_dict(state_dict)

        return model

    def save(self, model, filename, optimizer, meta):
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'meta': meta
                }, filename)

    def pre_train(self):
        base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
        pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
        base_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)

        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

        lr = 0.01

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        for epoch in range(EPOCHS):
            print(f"Epoch: {epoch+1}")

            self.model.train()

            for _, (data, target) in tqdm(enumerate(pre_train_loader), total=len(base_train_set)//BATCH_SIZE):
                data = data.to(device)
                target = target.to(device)

                # apply transform and model on whole batch directly on device
                data, target = self.encode((data,target))
                output = self.model(data.squeeze())

                loss = F.cross_entropy(output.squeeze(), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                train_acc = self.test(mask, base_train_loader, base_train_set)
                test_acc = self.test(mask, test_loader, base_test_set)

                print(f"The train accuracy is {train_acc*100}%")
                print(f"The test accuracy is {test_acc*100}%")

            scheduler.step()

        del base_train_set
        del pre_train_loader
    
    def run(self):
        if PRE_TRAIN:
            receptive_field = stats.get_receptive_field_size(16, 24)
            print(receptive_field)
            configuration = stats.get_kernel_size_and_layers(201)  ##configuration = (kernel_size, num_layers)
            print(f"configuration = {configuration}")

            if receptive_field < 201:
                print("Receptive field is too small for the task")
            else:
                self.pre_train()
                model = TorchModel(self.model)
        
        else:
            pass


if __name__ == '__main__':
    print(fscil_directory)
    # Using same parameters as provided pre-trained models
    model = TCN(
        20, 200, [256] * 6, [3] * 6,
        batch_norm=True, weight_norm=True, dropout=0.1, groups=-1, bottleneck=True).to(device)
    bench = NeuroTester(model)
    bench.is_neurobench_used = False
    bench.run()
