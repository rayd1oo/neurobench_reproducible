import os
import json
from datetime import datetime

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
EPOCHS = 11 # if pre-training from scratch
NUM_REPEATS = 1

# Define MFCC pre-processing
n_fft = 512
win_length = None
hop_length = 240
n_mels = 20
n_mfcc = 20


class NeuroTester:
    def __init__(self, model, model_name) -> None:
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
        self.model_name = model_name
        self.model = model
        self.optimizer = None
        self.is_neurobench_used = True
        self.meta = {
            "best_epoch": None,
            "best_train_acc": 0,
            "best_test_acc": 0,
        }
    
    @property
    def neurobench_on(self) -> bool:
        return self.is_neurobench_used
    
    def diy_test(self, mask, loader, dataset) -> float:
        print("Running Benchmark")
        self.model.eval()
        """Evaluate accuracy of a model on the given data set."""

        # init
        acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
        n = 0

        for _, (X, y) in tqdm(enumerate(loader), total=len(dataset)//BATCH_SIZE):
            # Copy the data to device.
            X, y = X.to(device), y.to(device)
            X, y = self.encode((X, y))
            with torch.no_grad():
                y = y.long()
                acc_sum += torch.sum((torch.argmax(self.model(X.squeeze()), dim=-1) == y))
                n += y.shape[0]  # increases with the number of samples in the batch
        return acc_sum.item() / n
    
    def neurobench(self, mask, loader, dataset) -> float:
        self.model.eval()
        out_mask = lambda x: x - mask
        with torch.no_grad():
            benchmark = Benchmark(TorchModel(self.model), metric_list=[[], ["classification_accuracy"]], dataloader=loader,
                                preprocessors=[to_device, self.encode, squeeze],
                                postprocessors=[out_mask, out2pred, torch.squeeze])

            pre_train_results = benchmark.run()
            test_accuracy = pre_train_results['classification_accuracy']

        return test_accuracy
    
    def test(self, mask, loader, dataset) -> float:
        if self.neurobench_on:
            return self.neurobench(mask, loader, dataset)
        else:
            return self.diy_test(mask, loader, dataset)

    def load(self, model) -> None:
        state_dict = torch.load(os.path.join(MODEL_SAVE_DIR, f"model_{self.model_name}"),
                                map_location=device).get("model")
        model.load_state_dict(state_dict)

    def save(self, model, optimizer, filename, meta) -> None:
        if not os.path.exists(MODEL_SAVE_DIR):
            # Create the directory if it doesn't exist
            os.makedirs(MODEL_SAVE_DIR)
        torch.save({
                'model': model.state_dict(),
                'optmizer': optimizer,
                'meta': meta
                }, os.path.join(MODEL_SAVE_DIR, filename))

    def pre_train(self):
        base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
        pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
        base_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)

        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

        lr = 0.01

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 5 == 0:
                train_acc = self.test(mask=mask, loader=base_train_loader, dataset=base_train_set)
                test_acc = self.test(mask=mask, loader=test_loader, dataset=base_test_set)

                if (test_acc > self.meta["best_test_acc"]) and \
                    (abs(test_acc - self.meta["best_test_acc"]) > 1e-6):
                    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                    self.meta["epoch"] = epoch+1
                    self.meta["best_train_acc"] = train_acc
                    self.meta["best_test_acc"] = test_acc
                    self.meta["save_time"] = current_time

                    # filename = "pre_trained_model"
                    filename = f"model_{self.model_name}"

                    self.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        filename=filename,
                        meta=self.meta)
                    print(f"{filename} saved...")

                print(f"The train accuracy is {train_acc*100}%")
                print(f"The test accuracy is {test_acc*100}%")

            scheduler.step()

        del base_train_set
        del pre_train_loader
    

    def incremental_learn(self, model) -> None:
        # Preparation for Prototypical Continual Learning        
        all_evals = []
        all_query = []
        all_act_sparsity = []
        all_syn_ops_dense = []
        all_syn_ops_macs = []
        all_syn_ops_acs = []

        ### Readout Prototypical Conversion ###

        # Loading non-shuffled trainset to get all samples of each class per batch
        base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
        train_loader = DataLoader(base_train_set, batch_size=500, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Set-up new proto readout layer
        output = model.net.output
        proto_out = nn.Linear(512, 200, bias=True).to(device)
        proto_out.weight.data = output.weight.data

        # Compute prototype weights for base classes
        print("Computing prototype weights for base classes")
        for data, target in tqdm(train_loader):
            data, target = self.encode((data.to(device), target.to(device)))
            data = data.squeeze()
            class_id = target[0]

            features = model.net(data, features_out=True)

            mean = torch.sum(features, dim=0)/500
            proto_out.weight.data[class_id] = 2*mean
            proto_out.bias.data[class_id] = -torch.matmul(mean, mean.t())

            del data
            del features
            del mean

        # Replace pre-trained readout with prototypical layer
        model.net.output = proto_out

        del base_train_set
        del train_loader


        ### Evaluation phase ###

        # Get base test set for evaluation
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

        for eval_iter in range(NUM_REPEATS):
            print(f"Evaluation Iteration: 0")

            eval_model = copy.deepcopy(model)

            eval_accs = []
            query_accs = []
            act_sparsity = []
            syn_ops_dense = []
            syn_ops_macs = []
            syn_ops_acs = []

            test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            # Define an arbitrary resampling as an example of pre-processor to feed to the Benchmark object
            eval_model.net.eval()

            # Metrics
            static_metrics = ["footprint", "connection_sparsity"]
            workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

            # Define benchmark object
            benchmark_all_test = Benchmark(eval_model, metric_list=[static_metrics, workload_metrics], dataloader=test_loader, 
                                preprocessors=[to_device, encode, squeeze], postprocessors=[])

            # Define specific post-processing with masking on the base classes
            mask = torch.full((200,), float('inf')).to(device)
            mask[torch.arange(0,100, dtype=int)] = 0
            out_mask = lambda x: x - mask

            # Run session 0 benchmark on base classes
            print(f"Session: 0")

            pre_train_results = benchmark_all_test.run(postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])

            print("Base results:", pre_train_results)

            eval_accs.append(pre_train_results['classification_accuracy'])
            act_sparsity.append(pre_train_results['activation_sparsity'])
            syn_ops_dense.append(pre_train_results['synaptic_operations']['Dense'])
            syn_ops_macs.append(pre_train_results['synaptic_operations']['Effective_MACs'])
            syn_ops_acs.append(pre_train_results['synaptic_operations']['Effective_ACs'])

            print(f"The base accuracy is {eval_accs[-1]*100}%")

            # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
            few_shot_dataloader = IncrementalFewShot(k_shot=NUM_SHOTS, 
                                        root = ROOT,
                                        query_shots=100,
                                        support_query_split=(100,100))

            # Iteration over incremental sessions
            for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
                print(f"Session: {session+1}")

                # Define benchmark object for new classes
                benchmark_new_classes = Benchmark(eval_model, metric_list=[[],["classification_accuracy"]], dataloader=None,
                                    preprocessors=[to_device, self.encode, squeeze], postprocessors=[])

                ### Computing Prototypical Weights and Biases of new classes ###
                data = None

                for X_shot, y_shot in support:
                    if data is None:
                        data = X_shot
                        target = y_shot
                    else:
                        data = torch.cat((data,X_shot), 0)
                        target = torch.cat((target,y_shot), 0)

                data, target = self.encode((data.to(device), target.to(device)))
                data = data.squeeze()

                new_classes = y_shot.tolist()
                Nways = len(y_shot) #Number of ways, should always be 10

                features = eval_model.net(data, features_out=True)

                for index, class_id in enumerate(new_classes):
                    mean = torch.sum(features[[i*Nways+index for i in range(NUM_SHOTS)]], dim=0)/NUM_SHOTS
                    eval_model.net.output.weight.data[class_id] = 2*mean
                    eval_model.net.output.bias.data[class_id] = -torch.matmul(mean, mean.t())

                ### Testing phase ###
                eval_model.net.eval()

                # Define session dataloaders for query and query + base_test samples
                query_loader = DataLoader(query, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

                full_session_test_set = ConcatDataset([base_test_set, query])
                full_session_test_loader = DataLoader(full_session_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

                # Create a mask function to only consider accuracy on classes presented so far
                session_classes = torch.cat((torch.arange(0,100, dtype=int), torch.IntTensor(query_classes))) 
                mask = torch.full((200,), float('inf')).to(device)
                mask[session_classes] = 0
                out_mask = lambda x: x - mask

                # Run benchmark to evaluate accuracy of this specific session
                session_results = benchmark_all_test.run(dataloader = full_session_test_loader, postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])
                print("Session results:", session_results)

                eval_accs.append(session_results['classification_accuracy'])
                act_sparsity.append(session_results['activation_sparsity'])
                syn_ops_dense.append(session_results['synaptic_operations']['Dense'])
                syn_ops_macs.append(session_results['synaptic_operations']['Effective_MACs'])
                syn_ops_acs.append(pre_train_results['synaptic_operations']['Effective_ACs'])
                print(f"Session accuracy: {session_results['classification_accuracy']*100} %")

                # Run benchmark on query classes only
                query_results = benchmark_new_classes.run(dataloader = query_loader, postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])
                print(f"Accuracy on new classes: {query_results['classification_accuracy']*100} %")
                query_accs.append(query_results['classification_accuracy'])

            all_evals.append(eval_accs)
            all_query.append(query_accs)
            all_act_sparsity.append(act_sparsity)
            all_syn_ops_dense.append(syn_ops_dense)
            all_syn_ops_macs.append(syn_ops_macs)
            all_syn_ops_acs.append(syn_ops_acs)

            mean_accuracy = np.mean(eval_accs)
            print(f"The total mean accuracy is {mean_accuracy*100}%")

            # Print all data
            print(f"Eval Accs: {eval_accs}")
            print(f"Query Accs: {query_accs}")
            print(f"Act Sparsity: {act_sparsity}")
            print(f"Syn Ops Dense: {syn_ops_dense}")
            print(f"Syn Ops MACs: {syn_ops_macs}")

    def run(self) -> None:
        if PRE_TRAIN:
            receptive_field = stats.get_receptive_field_size(16, 24)
            print(receptive_field)
            configuration = stats.get_kernel_size_and_layers(201)  ##configuration = (kernel_size, num_layers)
            print(f"configuration = {configuration}")

            if receptive_field < 201:
                print("Receptive field is too small for the task")
                return
            else:
                self.pre_train()
            
        # incremental learn
        self.load(self.model)
        self.incremental_learn(TorchModel(self.model))
    

if __name__ == '__main__':
    print(fscil_directory)
    # Using same parameters as provided pre-trained models
    model = TCN(
        20, 200, [256] * 4, [9] * 4,
        batch_norm=True, weight_norm=True, dropout=0.1, groups=-1, bottleneck=True).to(device)
    bench = NeuroTester(model, model.__class__.__name__)
    # bench.is_neurobench_used = False
    # PRE_TRAIN = False
    bench.run()
