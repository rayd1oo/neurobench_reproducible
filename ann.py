import torch
# import the dataloader
from torch.utils.data import DataLoader

# import the dataset, preprocessors and postprocessors you want to use
from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SPreProcessor
from neurobench.postprocessing import choose_max_count

# import the NeuroBench wrapper to wrap the snnTorch model
from neurobench.models import SNNTorchModel
# import the benchmark class
from neurobench.benchmarks import Benchmark

from torch import nn
import snntorch as snn
from snntorch import surrogate

beta = 0.9
spike_grad = surrogate.fast_sigmoid()
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(20, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 35),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
)


test_set = SpeechCommands(path="data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net.load_state_dict(torch.load("neurobench/examples/gsc/model_data/s2s_gsc_snntorch"))

# Wrap our net in the SNNTorchModel wrapper
model = SNNTorchModel(net)

preprocessors = [S2SPreProcessor()]
postprocessors = [choose_max_count]
static_metrics = ["footprint", "connection_sparsity"]
workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

benchmark = Benchmark(model, test_set_loader,
                      preprocessors, postprocessors, [static_metrics, workload_metrics])

results = benchmark.run()
print(results)
