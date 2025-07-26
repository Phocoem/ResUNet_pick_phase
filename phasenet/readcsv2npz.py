import numpy as np
import pandas as pd
import os

# Load file CSV
csv = pd.read_csv(r"D:\PhaseNet_UNet\data\chunk2.csv")

# Duyệt từng dòng và cập nhật file .npz tương ứng
for _, row in csv.iterrows():
    fname = "data_" + row["trace_name"] + ".npz" # hoặc "fname", tên file .npz
    p_idx = int(row["p_arrival_sample"])
    s_idx = int(row["s_arrival_sample"])

    # Load file .npz
    npz_path = os.path.join(r"D:\PhaseNet_UNet\data\big_train", fname)
    data = np.load(npz_path)

    # Ghi lại với p_idx, s_idx thêm vào
    np.savez(npz_path, data=data["data"], p_idx=p_idx, s_idx=s_idx)
