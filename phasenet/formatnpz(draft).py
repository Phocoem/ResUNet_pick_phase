import os
import numpy as np

def convert_npz_to_format(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.npz'):
            filepath = os.path.join(input_folder, filename)
            with np.load(filepath) as loaded_data:
                waveform = loaded_data['data'].astype(np.float32)

                # Đảm bảo waveform có shape (n_samples, 3)
                if waveform.ndim == 1:
                    waveform = waveform[:, np.newaxis]
                if waveform.shape[1] != 3:
                    waveform = np.repeat(waveform, 3, axis=1)

                # Lấy station_id từ tên file, bỏ .npz
                station_id = filename.replace('.npz', '')

                # Tạo các trường rỗng như yêu cầu
                np.savez_compressed(
                    os.path.join(output_folder, filename),
                    data=waveform,
                    station_id=station_id,
                    p_idx=np.array([], dtype=np.int32),
                    itp=np.array([[]], dtype=np.int32),
                    s_idx=np.array([], dtype=np.int32),
                    its=np.array([[]], dtype=np.int32)
                )

# Cách sử dụng:
convert_npz_to_format(r"D:\PhaseNet_UNet\data\output_folder", r"D:\PhaseNet_UNet\data\test1")
