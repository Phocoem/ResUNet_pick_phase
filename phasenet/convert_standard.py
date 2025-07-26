import os
import numpy as np

def convert_fur_format(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.endswith(".npz"):
            continue
        try:
            path = os.path.join(input_folder, fname)
            data = np.load(path)

            # Lấy dữ liệu & nhãn
            fur_data = data["data"].squeeze().T  # (6000, 3)
            p_idx = int(data["itp"][0]) if "itp" in data and data["itp"].size > 0 else -1
            s_idx = int(data["its"][0]) if "its" in data and data["its"].size > 0 else -1
            fname_base = fname.replace(".npz", "")

            converted = {
                "data": fur_data,
                "dt": 0.01,
                "p_idx": p_idx,
                "s_idx": s_idx,
                "snr": np.array([0.0, 0.0, 0.0]),
                "p_time": 0.0,
                "s_time": 0.0,
                "p_remark": "",
                "p_weight": 0.0,
                "s_remark": "",
                "s_weight": 0.0,
                "first_motion": "",
                "distance_km": 0.0,
                "azimuth": 0.0,
                "emergence_angle": 0.0,
                "network": "XX",
                "station": fname_base.split("_")[0],
                "location_code": "",
                "station_latitude": 0.0,
                "station_longitude": 0.0,
                "station_elevation_m": 0.0,
                "event_latitude": 0.0,
                "event_longitude": 0.0,
                "event_depth_km": 0.0,
                "event_time": 0.0,
                "event_magnitude": 0.0,
                "event_magnitude_type": "",
                "unit": "counts",
                "channels": "ZNE",
                "event_index": 0
            }

            out_path = os.path.join(output_folder, fname)
            np.savez(out_path, **converted)
            print(f"✅ Converted: {fname}")

        except Exception as e:
            print(f"❌ Failed: {fname} - {e}")

# Ví dụ sử dụng
convert_fur_format(r"D:\PhaseNet_UNet\data\big_train", r"D:\PhaseNet_UNet\data\big_train")
