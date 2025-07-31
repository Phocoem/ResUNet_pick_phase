import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.core import Trace
from obspy.signal.trigger import ar_pick

# === CẤU HÌNH ===
folder_path = r"D:\PhaseNet_UNet\data\test"  # ← Đổi đường dẫn nếu cần
sampling_rate = 100
dt = 1.0 / sampling_rate
plot_result = False  # Đặt True nếu muốn vẽ waveform
output_csv = os.path.join(folder_path, "ar_picker_results.csv")

# === Hàm gọi AR Picker ===
def run_ar_picker(trace, dt):
    try:
        ar_p, ar_s = ar_pick(
            trace.data, trace.data, trace.data,
            dt, 2.0, 10.0,
            lta_p=1.0, sta_p=0.1,
            lta_s=1.5, sta_s=0.2,
            m_p=2, m_s=2,
            l_p=0.5, l_s=0.5,
            s_pick=True
        )
        print(f"DEBUG: ar_p = {ar_p}, ar_s = {ar_s}")

        return ar_p, ar_s
    except Exception as e:
        print("❌ AR Picker failed:", repr(e))
        return None, None

# === Xử lý tất cả file .npz trong thư mục ===
results = []
npz_files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]

for fname in npz_files:
    fpath = os.path.join(folder_path, fname)
    print(f"\n📄 Đang xử lý: {fname}")
    try:
        data = np.load(fpath)

        if "data" not in data:
            print("⚠️ Không có trường 'data', bỏ qua")
            continue
        if "p_idx" not in data or "s_idx" not in data:
            print("⚠️ Không có p_idx hoặc s_idx, bỏ qua")
            continue

        waveform = data["data"]
        if waveform.shape[0] == 3:
            waveform = waveform.T

        trace = Trace(data=waveform[:, 0])
        trace.stats.sampling_rate = sampling_rate
        t = np.arange(waveform.shape[0]) * dt

        gt_p = float(np.atleast_1d(data["p_idx"])[0]) * dt
        gt_s = float(np.atleast_1d(data["s_idx"])[0]) * dt

        ar_p, ar_s = run_ar_picker(trace, dt)

        if ar_p:
            print(f"→ Ground truth P: {gt_p:.2f}s | AR P: {ar_p:.2f}s")
        else:
            print(f"→ AR P: Không xác định (giá trị: {ar_p})")


        if ar_s:
            print(f"→ Ground truth P: {gt_s:.2f}s | AR P: {ar_s:.2f}s")
        else:
            print(f"→ AR P: Không xác định (giá trị: {ar_s})")


        results.append({
            "file": fname,
            "gt_p_time": gt_p,
            "gt_s_time": gt_s,
            "ar_p_time": ar_p,
            "ar_s_time": ar_s
        })

        # === Vẽ waveform nếu bật ===
        if plot_result:
            plt.figure(figsize=(10, 3))
            plt.plot(t, trace.data, label="Z channel")
            plt.axvline(gt_p, color="gray", linestyle="--", label="GT P")
            plt.axvline(gt_s, color="gray", linestyle=":", label="GT S")
            if ar_p: plt.axvline(ar_p, color="blue", linestyle="--", label="AR P")
            if ar_s: plt.axvline(ar_s, color="blue", linestyle=":", label="AR S")
            plt.title(f"AR Picker vs Ground Truth - {fname}")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {fname}: {repr(e)}")

# === Ghi kết quả ra file CSV ===
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu kết quả vào: {output_csv}")
else:
    print("\n⚠️ Không có kết quả nào được ghi.")
