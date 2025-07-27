import os
import numpy as np

def pad_waveform_center(input_path, output_path, target_length=12000):
    """
    Chuyển waveform từ (3, N) → (target_length, 3), đưa waveform gốc vào giữa.
    Cập nhật lại p_idx, s_idx sau khi pad.
    """
    data = np.load(input_path)

    # Lấy waveform và transpose (3, N) → (N, 3)
    waveform = data['data']
    if waveform.shape[0] == 3:
        waveform = waveform.T
    elif waveform.shape[1] != 3:
        raise ValueError(f"{input_path}: waveform không đúng định dạng 3 kênh.")
    N = waveform.shape[0]

    # Tính padding để đưa waveform vào giữa
    pad_left = (target_length - N) // 2
    pad_right = target_length - N - pad_left

    padded = np.pad(waveform, ((pad_left, pad_right), (0, 0)), mode='constant')

    # Cập nhật chỉ số phase
    p_idx_new = int(data['p_idx']) + pad_left if 'p_idx' in data else -1
    s_idx_new = int(data['s_idx']) + pad_left if 's_idx' in data else -1

    # Ghi file mới
    np.savez_compressed(output_path,
                        data=padded,
                        p_idx=p_idx_new,
                        s_idx=s_idx_new)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.endswith(".npz"):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            try:
                pad_waveform_center(in_path, out_path)
                print(f"✅ {fname} → OK")
            except Exception as e:
                print(f"❌ {fname} → lỗi: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    input_dir = r"D:\PhaseNet_UNet\data\big_train"     # thư mục chứa file gốc
    output_dir = r"D:\PhaseNet_UNet\data\big_train"  # thư mục kết quả
    process_folder(input_dir, output_dir)
