import os
import json

import pandas as pd
import zarr
import numpy as np
from tqdm import tqdm

def write_zarr(output_dir: str, indices: dict[str, list[int]] | list[int], values: dict[str, list[list[float]]], attributes: dict[str, list[str]] | None = None, fill_value: float = 0.):
    if isinstance(indices, list):
        indices = {"": indices}
    
    if isinstance(values, list):
        values = {"": values}
    
    root = zarr.open_group(output_dir, mode="a")

    for key in indices:
        idx_array = np.array(indices[key])
        value_array = np.array(values[key])
        assert value_array.ndim == 2
        max_idx = idx_array.max().item() + 1
        shape = (max_idx, value_array.shape[1])
        dataset = root.create_dataset(key, shape=shape, dtype=value_array.dtype, chunks=(min(10000, max_idx), value_array.shape[1]), fill_value=fill_value, overwrite=True)
        dataset[idx_array, :] = value_array
        
        if attributes is not None:
            dataset.attrs.update(attributes)