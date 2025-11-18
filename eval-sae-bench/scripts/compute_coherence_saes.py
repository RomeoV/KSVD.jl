import torch
import h5py
import numpy as np
from sae_bench.custom_saes.batch_topk_sae import load_dictionary_learning_batch_topk_sae
from sae_bench.custom_saes.batch_topk_sae import load_dictionary_learning_matryoshka_batch_topk_sae

base_repo_path = "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108"
hf_model_name = "google/gemma-2-2b"
device = torch.device("cpu")
dtype = torch.float32

nnz_map = {
    20: 0,
    40: 1,
    80: 2
}

output_filename = "coherence_values_saes.h5"

with h5py.File(output_filename, "w") as hf:
    for model_type_prefix in ["MatryoshkaBatchTopK", "BatchTopK"]:
        for nnz_val, trainer_idx in nnz_map.items():
            model_path_middle = f"{model_type_prefix}_gemma-2-2b__0108/resid_post_layer_12/trainer_{trainer_idx}/ae.pt"
            dataset_name = f"{model_type_prefix.lower().replace('batchtopk', '')}_nnz_{nnz_val}"
            if "Matryoshka" not in model_type_prefix: # to make names "matryoshka_..." and "batch_..."
                 dataset_name = f"batch_nnz_{nnz_val}"


            print(f"Processing: {model_type_prefix}, nnz={nnz_val}")

            if model_type_prefix == "MatryoshkaBatchTopK":
                model = load_dictionary_learning_matryoshka_batch_topk_sae(
                    base_repo_path, model_path_middle, hf_model_name, device, dtype
                )
            else: # BatchTopK
                model = load_dictionary_learning_batch_topk_sae(
                    base_repo_path, model_path_middle, hf_model_name, device, dtype
                )

            D = model.W_dec
            # Ensure D is on the correct device and dtype if not already
            D = D.to(device=device, dtype=dtype)

            # Calculation: (D @ D.T - I).abs().max(dim=1)
            # D.shape[0] is the dictionary size (number of atoms)
            identity_matrix = torch.eye(D.shape[0], device=device, dtype=dtype)
            values = (D @ D.T - identity_matrix).abs().max(dim=1).values.detach().cpu().numpy()

            hf.create_dataset(dataset_name, data=values)
            print(f"Saved data to dataset: {dataset_name}")

print(f"All data saved to {output_filename}")
