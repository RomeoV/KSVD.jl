from ksvdmodel import KSVD

import numpy as np

import os
import torch
from pathlib import Path

import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes

# Parameters
MODEL_NAME = "gemma-2-2b"
HOOK_LAYER = 12
HOOK_NAME = f"blocks.{HOOK_LAYER}.hook_resid_post"
DEVICE = "cuda:0"
dtype = torch.bfloat16

# Load the dictionary and create the ksvd model
FILE_NAMES = [
        # these are outputs of `ksvd_gemma.jl`
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-4096-20.npy',
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-4096-40.npy',
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-4096-80.npy',
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-4096-20.npy'
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-4096-40.npy'
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-4096-80.npy'
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-16384-20.npy',
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-16384-40.npy',
        'gemma_2b_2ishM-result-baseline-2621440-65536-1-16384-80.npy',
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-16384-20.npy'
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-16384-40.npy'
        'gemma_2b_2ishM-result-matryoshka-2621440-65536-1-16384-80.npy'
        ]

RANDOM_SEED = 42

run_all_evals_custom_saes.MODEL_CONFIGS = {
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
    },
}

# Select your eval types here.
eval_types = [
    "absorption",
    "autointerp",
    "core",
    "ravel",
    "scr",
    "sparse_probing",
    # "tpp",
    # "unlearning",
]

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "ravel": "eval_results/ravel",
    "scr": "eval_results/scr",
    "sparse_probing": "eval_results/sparse_probing",
    # "tpp": "eval_results/tpp",
    # "unlearning": "eval_results/unlearning",
}

saes = []
for file_name in FILE_NAMES:
    L0 = int(file_name.split("-")[-1].split(".")[0])
    D = np.load(file_name)
    ksvd = KSVD(D, L0, MODEL_NAME, HOOK_LAYER, DEVICE, dtype)
    saes.append((Path(file_name).stem, ksvd))

d_model = run_all_evals_custom_saes.MODEL_CONFIGS[MODEL_NAME]["d_model"]
llm_batch_size = run_all_evals_custom_saes.MODEL_CONFIGS[MODEL_NAME]["batch_size"]
llm_dtype = run_all_evals_custom_saes.MODEL_CONFIGS[MODEL_NAME]["dtype"]


try:
    with open("openai_api_key.txt") as f:
        api_key = f.read().strip()
except FileNotFoundError:
    raise Exception("Please create openai_api_key.txt with your API key")

run_all_evals_custom_saes.run_evals(
    MODEL_NAME,
    saes,
    llm_batch_size,
    llm_dtype,
    DEVICE,
    eval_types=eval_types,
    force_rerun=False,
    save_activations=True,
    api_key=api_key,
)
