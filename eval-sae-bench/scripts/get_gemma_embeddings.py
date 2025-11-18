from nnsight import LanguageModel

from dictionary_learning.utils import hf_dataset_to_generator, get_submodule
from dictionary_learning import ActivationBuffer
import numpy as np
import h5py
import torch

DEVICE = "cuda:0"
MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 12
generator = hf_dataset_to_generator(DATASET_NAME)
config = {'torch_dtype': torch.bfloat16}
model = LanguageModel(MODEL_NAME, dispatch=True, device_map=DEVICE, torch_dtype=torch.bfloat16)

submodule = get_submodule(model, LAYER)
activation_dim = model.config.hidden_size

context_length = 128
llm_batch_size = 4
sae_batch_size = 2048
num_contexts_per_sae_batch = sae_batch_size // context_length

num_inputs_in_buffer = num_contexts_per_sae_batch * 20

activation_buffer = ActivationBuffer(
    generator,
    model,
    submodule,
    n_ctxs=int(1e4),
    ctx_len=context_length,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
    io="out",
    d_submodule=activation_dim,
    device=DEVICE,
)

# Set target number of samples
target_samples = 5_000_000

# Get first batch to determine shape
first_batch = next(iter(activation_buffer))
# print(first_batch.shape)
batch_size, activation_dim = first_batch.shape

# Create HDF5 file with resizable dataset
# This file will be ~50GB
with h5py.File('gemma_2b_5M_layer12_largebuf.h5', 'w') as f:
    # Create resizable dataset
    dset = f.create_dataset(
        'layer_activations',
        shape=(0, activation_dim),  # Initial shape
        maxshape=(target_samples, activation_dim),  # Maximum shape
        dtype=np.float32,
        chunks=True  # Enable chunking for efficient resizing
    )
    
    current_samples = 0
    
    # Write first batch
    batch_numpy = first_batch.cpu().float().numpy()
    dset.resize(batch_numpy.shape[0], axis=0)
    dset[:] = batch_numpy
    current_samples += batch_numpy.shape[0]
    
    # Process remaining batches
    for batch in activation_buffer:
        batch_numpy = batch.cpu().float().numpy()
        
        # Resize dataset to accommodate new batch
        new_size = current_samples + batch_numpy.shape[0]
        if new_size > target_samples:
            # Trim final batch if needed
            batch_numpy = batch_numpy[:target_samples-current_samples]
            new_size = target_samples
            
        dset.resize(new_size, axis=0)
        dset[current_samples:new_size] = batch_numpy
        current_samples = new_size
        
        print(f"Collected {current_samples} samples")
        
        if current_samples >= target_samples:
            break
