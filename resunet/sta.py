import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# ==== HÀM ĐÁNH GIÁ ====
def evaluate_pick_performance(pred_idx, true_idx, dt, tolerance=0.1):
    matched = []
    unmatched_pred = pred_idx.copy()
    unmatched_true = true_idx.copy()
    errors = []

    for gt in true_idx:
        closest = min(pred_idx, key=lambda x: abs(x - gt), default=None)
        if closest is not None and abs(closest - gt) * dt <= tolerance:
            matched.append((gt, closest))
            if closest in unmatched_pred: unmatched_pred.remove(closest)
            if gt in unmatched_true: unmatched_true.remove(gt)
            errors.append(abs(closest - gt) * dt)

    TP = len(matched)
    FP = len(unmatched_pred)
    FN = len(unmatched_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = np.mean(errors) if errors else None
    mad = np.median(np.abs(errors)) if errors else None

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1_score, 4),
        "MAE": round(mae, 4) if mae is not None else None,
        "MAD": round(mad, 4) if mad is not None else None,
    }

def to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (int, float)):
        return [int(x)]
    elif isinstance(x, list):
        return x
    elif x is None:
        return []
    else:
        raise ValueError(f"Không hỗ trợ kiểu dữ liệu: {type(x)}")


# ==== CẤU HÌNH ====
input_folder = r"D:\ResUNet_pick_phase\PhaseNet-main\data\test"
output_folder = r"D:\ResUNet_pick_phase\PhaseNet-main\sta_lta_output"
channel_index = 2
sta, lta = 1.0, 10.0
threshold_on = 2.5
threshold_off = 1.0
dt = 0.01
tolerance = 0.1

os.makedirs(output_folder, exist_ok=True)
metrics_all = []

# ==== LẶP QUA FILE ====
for file_name in os.listdir(input_folder):
    if file_name.endswith(".npz"):
        file_path = os.path.join(input_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        print(f"🔍 Đang xử lý: {file_name}")

        # Load dữ liệu
        data = np.load(file_path)
        waveform = data["data"]
        trace = waveform[:, channel_index]
        sr = 1 / dt
        t = np.arange(len(trace)) * dt

        # STA/LTA
        sta_samples = int(sta * sr)
        lta_samples = int(lta * sr)
        cft = classic_sta_lta(trace, sta_samples, lta_samples)
        triggers = trigger_onset(cft, threshold_on, threshold_off)
        trigger_indices = [onset[0] for onset in triggers]
        trigger_times = [i * dt for i in trigger_indices]

        # Vẽ waveform và trigger
        plt.figure(figsize=(12, 4))
        plt.plot(t, trace, label="Waveform", linewidth=0.5)
        for x in trigger_times:
            plt.axvline(x, color='r', linestyle='--', alpha=0.6)
        plt.title(f"STA/LTA Trigger: {file_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        fig_path = os.path.join(output_folder, f"{base_name}_plot.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()

        # Lưu trigger CSV
        df = pd.DataFrame({
            "trigger_index": trigger_indices,
            "trigger_time_sec": trigger_times
        })
        csv_path = os.path.join(output_folder, f"{base_name}_trigger.csv")
        df.to_csv(csv_path, index=False)

        # ==== ĐÁNH GIÁ VỚI P/S TRUE ====
        try:
            itp = to_list(data.get("itp") or data.get("p_idx"))
            its = to_list(data.get("its") or data.get("s_idx"))
        except Exception as e:
            print(f"⚠️ Không thể đọc ground truth từ {file_name}: {e}")
            continue

        metrics_p = evaluate_pick_performance(trigger_indices, to_list(itp), dt, tolerance)
        metrics_s = evaluate_pick_performance(trigger_indices, to_list(its), dt, tolerance)


        # Ghi kết quả
        metrics_all.append({
            "file": file_name,
            "P_TP": metrics_p["TP"],
            "P_FP": metrics_p["FP"],
            "P_FN": metrics_p["FN"],
            "P_Precision": metrics_p["Precision"],
            "P_Recall": metrics_p["Recall"],
            "P_F1": metrics_p["F1"],
            "P_MAE": metrics_p["MAE"],
            "P_MAD": metrics_p["MAD"],
            "S_TP": metrics_s["TP"],
            "S_FP": metrics_s["FP"],
            "S_FN": metrics_s["FN"],
            "S_Precision": metrics_s["Precision"],
            "S_Recall": metrics_s["Recall"],
            "S_F1": metrics_s["F1"],
            "S_MAE": metrics_s["MAE"],
            "S_MAD": metrics_s["MAD"],
        })

        print(f"✅ Đã xử lý: {file_name}")

# ==== LƯU FILE ĐÁNH GIÁ ====
metrics_df = pd.DataFrame(metrics_all)
metrics_df.to_csv(os.path.join(output_folder, "sta_lta_evaluation.csv"), index=False)
# ==== TỔNG KẾT TOÀN CỤC ====
def safe_mean(values):
    valid = [v for v in values if pd.notnull(v)]
    return round(np.mean(valid), 4) if len(valid) > 0 else None


# Tính tổng
total = {
    "P_TP": metrics_df["P_TP"].sum(),
    "P_FP": metrics_df["P_FP"].sum(),
    "P_FN": metrics_df["P_FN"].sum(),
    "S_TP": metrics_df["S_TP"].sum(),
    "S_FP": metrics_df["S_FP"].sum(),
    "S_FN": metrics_df["S_FN"].sum(),
}

# Precision/Recall/F1 tổng cho P và S
def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return round(precision, 4), round(recall, 4), round(f1, 4)

p_prec, p_rec, p_f1 = compute_prf(total["P_TP"], total["P_FP"], total["P_FN"])
s_prec, s_rec, s_f1 = compute_prf(total["S_TP"], total["S_FP"], total["S_FN"])

# Tính MAE & MAD trung bình trên các file có giá trị
p_mae_avg = safe_mean(metrics_df["P_MAE"])
p_mad_avg = safe_mean(metrics_df["P_MAD"])
s_mae_avg = safe_mean(metrics_df["S_MAE"])
s_mad_avg = safe_mean(metrics_df["S_MAD"])

# In ra bảng tổng kết
summary = {
    "P_Total_TP": total["P_TP"],
    "P_Total_FP": total["P_FP"],
    "P_Total_FN": total["P_FN"],
    "P_Precision": p_prec,
    "P_Recall": p_rec,
    "P_F1": p_f1,
    "P_MAE_Avg": p_mae_avg,
    "P_MAD_Avg": p_mad_avg,
    "S_Total_TP": total["S_TP"],
    "S_Total_FP": total["S_FP"],
    "S_Total_FN": total["S_FN"],
    "S_Precision": s_prec,
    "S_Recall": s_rec,
    "S_F1": s_f1,
    "S_MAE_Avg": s_mae_avg,
    "S_MAD_Avg": s_mad_avg,
}

# Ghi ra file tổng kết
summary_path = os.path.join(output_folder, "sta_lta_summary.txt")
with open(summary_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("\n📊 Tổng kết toàn bộ file:")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n🎉 ĐÃ HOÀN TẤT – KẾT QUẢ LƯU TRONG: sta_lta_evaluation.csv")
