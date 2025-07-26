import os
import csv

def write_mseed_filenames_to_csv(folder_path, output_csv):
    # Lấy danh sách file .mseed trong thư mục
    mseed_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    # Ghi vào file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fname'])  # Tiêu đề cột
        for filename in mseed_files:
            writer.writerow([filename])

    print(f"✅ Đã ghi {len(mseed_files)} tên file vào '{output_csv}'")

# Ví dụ sử dụng:
write_mseed_filenames_to_csv(r"D:\ResUNet_pick_phase\PhaseNet-main\data\test_2", r"D:\ResUNet_pick_phase\PhaseNet-main\data\test2.csv")
