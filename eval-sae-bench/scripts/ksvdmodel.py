import juliacall
from juliacall import Main as jl
jl.seval('using KSVD')

import numpy as np
import torch
import torch.nn as nn
import time
import os
import pickle
from collections import OrderedDict

import sae_bench.custom_saes.base_sae as base_sae

def jlmat32(M):
    return juliacall.convert(jl.Matrix[jl.Float32], M)

def hashtensor(t):
    return hash(t.to("cpu", dtype=torch.float32).numpy().tobytes())

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def __getitem__(self, key):
        if key not in self.cache:
            raise KeyError(key)
        # Move accessed item to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        # If key exists, update and move to end
        if key in self.cache:
            self.cache.move_to_end(key)
        # Insert new item and check capacity
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove first item (least recently used)
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)


class KSVD(base_sae.BaseSAE):
    def __init__(self,
                 D,
                 l0,
                 model_name,
                 hook_layer,
                 device,
                 dtype,
                 sub_sparse_code = False
    ):
        super().__init__(D.shape[0], D.shape[1], model_name, hook_layer, device, dtype)
        self.D = torch.tensor(D).to(torch.float32)
        self.Djl = jlmat32(self.D.cpu())
        self.Ddev = torch.tensor(D).to(device)
        self.DtD = (self.Ddev.T @ self.Ddev).cpu()
        self.DtDjl = jlmat32(self.DtD)
        self.l0 = l0
        self.W_dec = nn.Parameter(torch.from_numpy(D).T)
        self.sparse_coding_method = jl.KSVD.ParallelMatchingPursuit(max_nnz=self.l0)
        self.cache = LRUCache(550)
        self.last_cache_save_time = 0  # Track when cache was last saved
        self.sub_sparse_code = sub_sparse_code

        # Define cache filename based on parameters
        self.cache_file = os.path.expanduser(f"~/.cache/ksvd_cache_{hashtensor(self.D)}_{self.l0}.pkl")

    def encode(self, x, use_cache=True):
        # Check if the tensor has been computed before
        key = hashtensor(x)
        if key in self.cache and use_cache:
            print(f"Cache lookup for tensor: {key}")
            result = self.cache[key].to_dense().to(self.device, dtype=self.dtype)
        else:
            print(f"Compute for tensor: {key}")
            self.cache[key] = self._encode(x)
            result = self.cache[key].to_dense().to(self.device, dtype=self.dtype)

        # Ensure result is at least 2D
        if result.dim() == 1:
            result = result.unsqueeze(0)

        return result

    def _encode(self, x):
        # x is either (b, l, d) or (l, d)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        x = x.reshape(-1, hidden_dim).T.to(torch.float32)
        DtY = (self.Ddev.T @ x.to(self.device)).cpu()

        if self.sub_sparse_code:
            print("Using sparse coding matryoshka")
            Xjl = np.array(jl.KSVD.sparse_coding_matryoshka(self.sparse_coding_method, jlmat32(x.cpu()), self.Djl,
                         DtD=self.DtDjl, DtY=jlmat32(DtY)))
        else:
            Xjl = np.array(jl.sparse_coding(self.sparse_coding_method, jlmat32(x.cpu()), self.Djl,
                         DtD=self.DtDjl, DtY=jlmat32(DtY)))
        res = torch.tensor(Xjl)

        # res is num_dicts x (l * b)
        res = res.T
        res = res.reshape(*orig_shape[:-1], -1)
        return res.squeeze().to_sparse()

    def decode(self, x):
        D_torch = self.Ddev.to(self.device, dtype=self.dtype)
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).T.to(self.dtype)
        recon = D_torch @ x.to(self.device)
        recon = recon.reshape(self.D.shape[0], *orig_shape[:-1])
        dims = list(range(len(orig_shape)))  # Get all dimension indices
        dims = dims[1:] + [0]  # Move first index to end
        recon = recon.permute(*dims)
        return recon

    def forward(self, x):
        x = self.encode(x)
        recon = self.decode(x)
        return recon
