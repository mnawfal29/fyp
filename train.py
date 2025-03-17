import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import argparse
import gc
import torch

from transformers import CLIPProcessor

from CoLeLib.datasets import CIFAR100FSCIL, CUB200FSCIL, MiniImageNetFSCIL
from CoLeLib.training.strategies import CLIPPE

warnings.filterwarnings("ignore")

datasets = dict(
    cub200 = dict(dataset = CUB200FSCIL, train_mb_size_base_class = 4, train_epochs_base_class = 6),
    cifar100 = dict(dataset = CIFAR100FSCIL, train_mb_size_base_class = 32, train_epochs_base_class = 8),
    miniimagenet = dict(dataset = MiniImageNetFSCIL, train_mb_size_base_class = 32, train_epochs_base_class = 5)
)

parser = argparse.ArgumentParser()
parser.add_argument("--L_g", type=int, default=2, help="Number of general prompts to be used")
parser.add_argument("--L_s", type=int, default=2, help="Number of shared prompts to be used")
parser.add_argument("--D_g", type=int, default=4, help="Number of layers in which the general prompts will be processed")
parser.add_argument("--D_s", type=int, default=8, help="Number of layers in which the shared prompts will be processed")
parser.add_argument("--text_deep_replace_method", type=str, default="replace", choices=["replace", "accumalate", "accumulate_same"], help="Method to replace the text prompts in the encoders. Options: replace, accumulate, accumulate_same")
parser.add_argument("--vision_deep_replace_method", type=str, default="accumulate", choices=["replace", "accumulate", "accumulate_same"], help="Method to replace the vision prompts in the encoders. Options: replace, accumulate, accumulate_same")
parser.add_argument("--dataset_name", type=str, default="cifar100", choices=["cifar100", "cub200", "miniimagenet"], help="Name of the dataset to be used. Options: cifar100, cub200, miniimagenet")
parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to be executed", required=False)
parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to be used in the runs", required=False)
# parser.add_argument("--ablation", type=str, choices=["no_accumulation", "no_regularization", "no_vision_prompts"])

args = parser.parse_args()
n_runs = args.n_runs
dataset_name = args.dataset_name

few_shot_examples = [5]

assert n_runs > 0, "Number of runs must be greater than 0."
# NOTE: Seeds used in the paper for 5 runs are: seeds = [42, 13, 50, 24, 69]
seeds = [42]
# ablation = args.ablation

img_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").feature_extractor

if __name__ == "__main__":
    for run in range(n_runs):
        exp_name = f"model_{dataset_name}_L_g_{args.L_g}_L_s_{args.L_s}_D_g_{args.D_g}_D_s_{args.D_s}_run_{run}"
        
        Dataset = datasets[dataset_name]["dataset"]
        train_mb_size_base_class = datasets[dataset_name]["train_mb_size_base_class"]
        train_epochs_base_class = datasets[dataset_name]["train_epochs_base_class"]

        strategy = CLIPPE(
            device='cuda',
            seed=seeds[run],
            L_g=args.L_g,
            L_s=args.L_s,
            D_g=args.D_g,
            D_s=args.D_s,
            text_deep_replace_method=args.text_deep_replace_method,
            vision_deep_replace_method=args.vision_deep_replace_method,
            regularization_method='balance',
            train_mb_size_base_class=train_mb_size_base_class,
            train_epochs_base_class=train_epochs_base_class,
            lr=0.00325,
            use_scheduler=True,
            json_file_name=exp_name + ".json",
            eval_mb_size=64
        )
        print(strategy.model)

        experiences = Dataset(transforms=img_preprocess)

        strategy.train(experiences)

        torch.cuda.empty_cache()
        gc.collect()

        del strategy
        del experiences
