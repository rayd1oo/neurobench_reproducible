from tcn_lib import TCN
import torch
import neurobench
from neurobench.datasets import MSWC_dataset

def main():
    # print("hello world")
    mswc_dataset_training = MSWC_dataset.MSWC(root = "MSWC/", 
        subset = "base", procedure = "training", language = "English" ,download = True)
    mswc_dataset_training.download()
    print()

if __name__ == "__main__":
    main()