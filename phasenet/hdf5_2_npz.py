import os
import numpy as np
import h5py

def save_hdf5_group_to_npz(hdf_group, output_folder, prefix=''):
    for key in hdf_group.keys():
        item = hdf_group[key]
        item_path = f"{prefix}{key}"
        if isinstance(item, h5py.Dataset):
            data = np.array(item)
            npz_file = os.path.join(output_folder, f"{item_path.replace('/', '_')}.npz")
            np.savez_compressed(npz_file, data=data)
        elif isinstance(item, h5py.Group):
            save_hdf5_group_to_npz(item, output_folder, prefix=f"{item_path}/")

def hdf5_to_npz_folder(hdf5_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with h5py.File(hdf5_path, 'r') as hdf:
        save_hdf5_group_to_npz(hdf, output_folder)

# Cách dùng:
hdf5_to_npz_folder(r"D:\ResUNet_pick_phase\PhaseNet-main\data\chunk2.hdf5", 'big_train')
