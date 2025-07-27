import os
import numpy as np
import pandas as pd

def check_npz_folder(folder_path, output_csv):
    report = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            try:
                npz = np.load(file_path, allow_pickle=True)
                data = npz.get('data')
                p_idx = npz.get('p_idx')
                s_idx = npz.get('s_idx')

                # Kiểm tra hợp lệ
                valid_shape = isinstance(data, np.ndarray) and data.shape == (12000, 3)
                valid_p = isinstance(p_idx, (int, np.integer)) and 0 <= p_idx < 12000
                valid_s = isinstance(s_idx, (int, np.integer)) and 0 <= s_idx < 12000

                report.append({
                    'file': filename,
                    'data_shape': str(data.shape) if data is not None else None,
                    'p_idx': int(p_idx) if p_idx is not None else None,
                    's_idx': int(s_idx) if s_idx is not None else None,
                    'valid_shape': valid_shape,
                    'valid_p_idx': valid_p,
                    'valid_s_idx': valid_s,
                })
            except Exception as e:
                report.append({
                    'file': filename,
                    'data_shape': None,
                    'p_idx': None,
                    's_idx': None,
                    'valid_shape': False,
                    'valid_p_idx': False,
                    'valid_s_idx': False,
                    'error': str(e)
                })

    df = pd.DataFrame(report)
    df.to_csv(output_csv, index=False)
    print(f"[✔] Saved report to {output_csv}")
    return df


# Ví dụ dùng
if __name__ == "__main__":
    folder = r"D:\ResUNet_pick_phase\PhaseNet-main\data\train"  # ← Đặt đúng đường dẫn
    output_csv = r"D:\ResUNet_pick_phase\PhaseNet-main\data\read.csv"
    check_npz_folder(folder, output_csv)
