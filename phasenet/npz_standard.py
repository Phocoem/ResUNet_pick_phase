import os
import numpy as np

input_folder = r"D:\PhaseNet_UNet\data\big_train"        # 📁 Đường dẫn thư mục chứa .npz
output_folder = r"D:\PhaseNet_UNet\data\big_train"    # 📁 Nơi lưu file căn chỉnh
os.makedirs(output_folder, exist_ok=True)

target_p_idx = 3000  # Mục tiêu đưa P về vị trí này

for filename in os.listdir(input_folder):
    if not filename.endswith(".npz"):
        continue

    path = os.path.join(input_folder, filename)
    try:
        data = np.load(path)
        X = data["data"]
        p_idx = int(data["p_idx"])
        s_idx = int(data["s_idx"])
        dt = 0.01

        shift = target_p_idx - p_idx
        X_shifted = np.roll(X, shift, axis=0)

        itp = np.array([float(target_p_idx)], dtype=np.float32)
        its = np.array([float(s_idx + shift)], dtype=np.float32)

        out_path = os.path.join(output_folder, filename)
        np.savez(out_path, data=X_shifted, dt=dt,
                 p_idx=np.array([target_p_idx]),
                 s_idx=np.array([s_idx + shift]),
                 itp=itp, its=its)

        print(f"✅ {filename} → dịch P về {target_p_idx} (shift={shift})")
    except Exception as e:
        print(f"❌ {filename} → lỗi: {e}")
