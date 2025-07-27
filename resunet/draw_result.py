import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Đọc picks.csv
df = pd.read_csv(r"D:\PhaseNet-main\PhaseNet-main\results\picks.csv")

# Lọc ra danh sách file_name duy nhất
unique_files = df["file_name"].unique()

# Đường dẫn thư mục chứa file npz
data_dir = r"D:\PhaseNet-main\PhaseNet-main\data\npz_final"

# Lặp qua từng file (duy nhất)
for file_name in unique_files:
    npz_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(npz_path):
        print(f"❌ File không tồn tại: {npz_path}")
        continue

    # Đọc waveform
    data = np.load(npz_path)
    waveform = data["data"]

    # Lấy pick tốt nhất cho mỗi phase
    picks = df[df["file_name"] == file_name]
    best_p = picks[picks["phase_type"] == "P"].sort_values("phase_score", ascending=False).head(1)
    best_s = picks[picks["phase_type"] == "S"].sort_values("phase_score", ascending=False).head(1)
    p_idx = best_p["phase_index"].values
    s_idx = best_s["phase_index"].values

    # Vẽ
    t = np.arange(waveform.shape[0]) / 100  # Giả định dt=0.01s
    plt.figure(figsize=(15, 6))
    for i, ch in enumerate(['Z', 'N', 'E']):
        plt.plot(t, waveform[:, i] + i * 10, label=f'Channel {ch}')
    
    for idx in p_idx:
        plt.axvline(x=idx / 100, color='blue', linestyle='--', label='P pick')
    for idx in s_idx:
        plt.axvline(x=idx / 100, color='red', linestyle='--', label='S pick')

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform and Picks for {file_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
