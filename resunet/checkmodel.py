import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import UNet
from model import ModelConfig

def predict_obs_folder(npz_dir, checkpoint_path):
    tf.reset_default_graph()  # 💥 Reset toàn bộ graph trước khi tạo model
    config = ModelConfig(X_shape=[3000, 1, 3], dt=0.01)
    model = UNet(config)
    model.build(mode='test')

    sess = tf.Session()
    model.restore(sess, checkpoint_path)

    total_p_err, total_s_err = 0, 0
    count_p, count_s = 0, 0

    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])

    for fname in files:
        try:
            path = os.path.join(npz_dir, fname)
            npz = np.load(path)
            data = npz["data"].astype(np.float32)
            data = (data - data.mean(axis=0)) / data.std(axis=0)
            X = np.expand_dims(data[:3000], axis=(0, 1))  # [1, 3000, 1, 3]

            pred = sess.run(model.prob, feed_dict={
                model.X: X,
                model.drop_rate: 0.0,
                model.is_training: False
            })

            pred_p = np.argmax(pred[0, :, 0, 0])
            pred_s = np.argmax(pred[0, :, 0, 1])
            true_p = int(npz["p_idx"]) if "p_idx" in npz else None
            true_s = int(npz["s_idx"]) if "s_idx" in npz else None

            print(f"\n{fname}")
            print(f"Pred P: {pred_p}", end="")
            if true_p is not None:
                print(f" | True P: {true_p} | Error: {abs(pred_p - true_p)}")
                total_p_err += abs(pred_p - true_p)
                count_p += 1
            else:
                print(" | No true P")

            print(f"Pred S: {pred_s}", end="")
            if true_s is not None:
                print(f" | True S: {true_s} | Error: {abs(pred_s - true_s)}")
                total_s_err += abs(pred_s - true_s)
                count_s += 1
            else:
                print(" | No true S")

        except Exception as e:
            print(f"Error reading {fname}: {e}")

    print("\n===== FINAL METRICS =====")
    if count_p: print("P MAE:", total_p_err / count_p)
    if count_s: print("S MAE:", total_s_err / count_s)
    sess.close()


# ⚙️ Chạy tại đây
if __name__ == "__main__":
    tf.disable_eager_execution()
    predict_obs_folder(
        npz_dir=r"D:\PhaseNet-main\PhaseNet-main\test_data\OO_data",
        checkpoint_path=r"D:\PhaseNet-main\PhaseNet-main\model\190703-214543\model_95.ckpt"
    )
