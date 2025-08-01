import argparse
import logging
import multiprocessing
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from data_reader import DataReader_mseed_array, DataReader_pred
from model import ModelConfig, UNet
from metrics import compute_class_metrics, compute_mae_mad, plot_residual_hist

from postprocess import (
    extract_amplitude,
    extract_picks,
    save_picks,
    save_picks_json,
    save_prob_h5,
)
from tqdm import tqdm
from visulization import plot_waveform

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--model_dir", help="Checkpoint directory (default: None)")
    parser.add_argument("--data_dir", default="", help="Input file directory")
    parser.add_argument("--data_list", default="", help="Input csv file")
    parser.add_argument("--hdf5_file", default="", help="Input hdf5 file")
    parser.add_argument("--hdf5_group", default="data", help="data group name in hdf5 file")
    parser.add_argument("--result_dir", default="results", help="Output directory")
    parser.add_argument("--result_fname", default="picks", help="Output file")
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--mpd", default=50, type=float, help="Minimum peak distance")
    parser.add_argument("--amplitude", action="store_true", help="if return amplitude value")
    parser.add_argument("--format", default="numpy", help="input format")
    parser.add_argument("--s3_url", default="localhost:9000", help="s3 url")
    parser.add_argument("--stations", default="", help="seismic station info")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    parser.add_argument("--pre_sec", default=1, type=float, help="Window length before pick")
    parser.add_argument("--post_sec", default=4, type=float, help="Window length after pick")

    parser.add_argument("--highpass_filter", default=0.0, type=float, help="Highpass filter")
    parser.add_argument("--response_xml", default=None, type=str, help="response xml file")
    parser.add_argument("--sampling_rate", default=100, type=float, help="sampling rate")
    args = parser.parse_args()

    return args


def pred_fn(args, data_reader, figure_dir=None, prob_dir=None, log_dir=None):
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if (args.plot_figure == True) and (figure_dir is None):
        figure_dir = os.path.join(log_dir, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
    if (args.save_prob == True) and (prob_dir is None):
        prob_dir = os.path.join(log_dir, "probs")
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)
    if args.save_prob:
        h5 = h5py.File(os.path.join(args.result_dir, "result.h5"), "w", libver="latest")
        prob_h5 = h5.create_group("/prob")
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))

    with tf.compat.v1.name_scope("Input_Batch"):
        if args.format == "mseed_array":
            batch_size = 1
        else:
            batch_size = args.batch_size
        dataset = data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    config = ModelConfig(X_shape=data_reader.X_shape)
    with open(os.path.join(log_dir, "config.log"), "w") as fp:
        fp.write("\n".join("%s: %s" % item for item in vars(config).items()))

    model = UNet(config=config, input_batch=batch, mode="pred")
    # model = UNet(config=config, mode="pred")
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        latest_check_point = tf.train.latest_checkpoint(args.model_dir)
        logging.info(f"restoring model {latest_check_point}")
        saver.restore(sess, latest_check_point)

        picks = []
        amps = [] if args.amplitude else None
        if args.plot_figure:
            multiprocessing.set_start_method("spawn")
            pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for _ in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            if args.amplitude:
                pred_batch, X_batch, amp_batch, fname_batch, t0_batch, station_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3], batch[4]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )
            #    X_batch, amp_batch, fname_batch, t0_batch = sess.run([batch[0], batch[1], batch[2], batch[3]])
            else:
                pred_batch, X_batch, fname_batch, t0_batch, station_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )
            #    X_batch, fname_batch, t0_batch = sess.run([model.preds, batch[0], batch[1], batch[2]])
            # pred_batch = []
            # for i in range(0, len(X_batch), 1):
            #     pred_batch.append(sess.run(model.preds, feed_dict={model.X: X_batch[i:i+1], model.drop_rate: 0, model.is_training: False}))
            # pred_batch = np.vstack(pred_batch)

            waveforms = None
            if args.amplitude:
                waveforms = amp_batch

            picks_ = extract_picks(
                preds=pred_batch,
                file_names=fname_batch,
                station_ids=station_batch,
                begin_times=t0_batch,
                config=args,
                waveforms=waveforms,
                use_amplitude=args.amplitude,
                dt=1.0 / args.sampling_rate,
            )

            picks.extend(picks_)

            # ## save pick per file
            # if (len(fname_batch) == 1) & (len(picks_) > 0):
            #     df = pd.DataFrame(picks_)
            #     df = df[df["phase_index"] > 10]
            #     if not os.path.exists(os.path.join(args.result_dir, "picks")):
            #         os.makedirs(os.path.join(args.result_dir, "picks"))
            #     df = df[
            #         [
            #             "station_id",
            #             "begin_time",
            #             "phase_index",
            #             "phase_time",
            #             "phase_score",
            #             "phase_type",
            #             "phase_amplitude",
            #             "dt",
            #         ]
            #     ]
            #     df.to_csv(
            #         os.path.join(
            #             args.result_dir, "picks", fname_batch[0].decode().split("/")[-1].rstrip(".mseed") + ".csv"
            #         ),
            #         index=False,
            #     )

            if args.plot_figure:
                if not (isinstance(fname_batch, np.ndarray) or isinstance(fname_batch, list)):
                    fname_batch = [fname_batch.decode().rstrip(".mseed") + "_" + x.decode() for x in station_batch]
                else:
                    fname_batch = [x.decode() for x in fname_batch]
                pool.starmap(
                    partial(
                        plot_waveform,
                        figure_dir=figure_dir,
                    ),
                    # zip(X_batch, pred_batch, [x.decode() for x in fname_batch]),
                    zip(X_batch, pred_batch, fname_batch),
                )

            if args.save_prob:
                # save_prob(pred_batch, fname_batch, prob_dir=prob_dir)
                if not (isinstance(fname_batch, np.ndarray) or isinstance(fname_batch, list)):
                    fname_batch = [fname_batch.decode().rstrip(".mseed") + "_" + x.decode() for x in station_batch]
                else:
                    fname_batch = [x.decode() for x in fname_batch]
                save_prob_h5(pred_batch, fname_batch, prob_h5)

        if len(picks) > 0:
            # save_picks(picks, args.result_dir, amps=amps, fname=args.result_fname+".csv")
            # save_picks_json(picks, args.result_dir, dt=data_reader.dt, amps=amps, fname=args.result_fname+".json")
            df = pd.DataFrame(picks)
            # df["fname"] = df["file_name"]
            # df["id"] = df["station_id"]
            # df["timestamp"] = df["phase_time"]
            # df["prob"] = df["phase_prob"]
            # df["type"] = df["phase_type"]

            base_columns = [
                "station_id",
                "begin_time",
                "phase_index",
                "phase_time",
                "phase_score",
                "phase_type",
                "file_name",
            ]
            if args.amplitude:
                base_columns.append("phase_amplitude")
                base_columns.append("phase_amp")
                df["phase_amp"] = df["phase_amplitude"]

            df = df[base_columns]
            df.to_csv(os.path.join(args.result_dir, args.result_fname + ".csv"), index=False)
            # === GROUND TRUTH LOADING ===
            gt_p, gt_s, pred_p, pred_s = [], [], [], []

            for fname in df["file_name"].unique():
                fpath = os.path.join(args.data_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    data = np.load(fpath, allow_pickle=True)
                    gt_p.append(int(data["p_idx"]))
                    gt_s.append(int(data["s_idx"]))
                except Exception as e:
                    print(f"Skip {fname}: {e}")
                    continue

                picks_f = df[df["file_name"] == fname]
                pred_p_idx = picks_f[picks_f["phase_type"] == "P"].sort_values("phase_score", ascending=False)["phase_index"].values
                pred_s_idx = picks_f[picks_f["phase_type"] == "S"].sort_values("phase_score", ascending=False)["phase_index"].values


                pred_p.append(int(pred_p_idx[0]) if len(pred_p_idx) > 0 else -1)
                pred_s.append(int(pred_s_idx[0]) if len(pred_s_idx) > 0 else -1)

                # === METRICS CALC ===
            res_p = [p - g for p, g in zip(pred_p, gt_p) if p > 0 and g > 0]
            res_s = [s - g for s, g in zip(pred_s, gt_s) if s > 0 and g > 0]

            # Nhị phân hóa GT và Pred cho phase P/S
            binary_gt_p = [1 if g > 0 else 0 for g in gt_p]
            binary_pred_p = [1 if p > 0 else 0 for p in pred_p]
            binary_gt_s = [1 if g > 0 else 0 for g in gt_s]
            binary_pred_s = [1 if s > 0 else 0 for s in pred_s]

            # Precision / Recall / F1
            metrics_p = compute_class_metrics(np.array(binary_gt_p), np.array(binary_pred_p))
            metrics_s = compute_class_metrics(np.array(binary_gt_s), np.array(binary_pred_s))

            # MAE / MAD
            mae_p = compute_mae_mad(gt_p, pred_p)
            mae_s = compute_mae_mad(gt_s, pred_s)


            # === SAVE TO FILE ===
            with open(os.path.join(args.result_dir, "eval_metrics.txt"), "w") as f:
                f.write(f"Precision_P: {metrics_p['precision']:.4f}, Recall_P: {metrics_p['recall']:.4f}, F1_P: {metrics_p['f1']:.4f}\n")
                f.write(f"Precision_S: {metrics_s['precision']:.4f}, Recall_S: {metrics_s['recall']:.4f}, F1_S: {metrics_s['f1']:.4f}\n")
                f.write(f"MAE_P: {mae_p['mae']:.2f}, MAD_P: {mae_p['mad']:.2f}\n")
                f.write(f"MAE_S: {mae_s['mae']:.2f}, MAD_S: {mae_s['mad']:.2f}\n")

            # === PLOT ===
            plot_residual_hist(res_p, "Residuals for P", os.path.join(args.result_dir, "residual_p.png"))
            plot_residual_hist(res_s, "Residuals for S", os.path.join(args.result_dir, "residual_s.png"))

            print(
                f"Done with {len(df[df['phase_type'] == 'P'])} P-picks and {len(df[df['phase_type'] == 'S'])} S-picks"
            )
        else:
            print(f"Done with 0 P-picks and 0 S-picks")
    return 0


def main(args):
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    with tf.compat.v1.name_scope("create_inputs"):
        if args.format == "mseed_array":
            data_reader = DataReader_mseed_array(
                data_dir=args.data_dir,
                data_list=args.data_list,
                stations=args.stations,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
            )
        else:
            data_reader = DataReader_pred(
                format=args.format,
                data_dir=args.data_dir,
                data_list=args.data_list,
                hdf5_file=args.hdf5_file,
                hdf5_group=args.hdf5_group,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
                response_xml=args.response_xml,
                sampling_rate=args.sampling_rate,
            )

        pred_fn(args, data_reader, log_dir=args.result_dir)

    return


if __name__ == "__main__":
    args = read_args()
    main(args)
