"""
STEP 4.3 / 4.4 - Train and evaluate the BiLSTM and the CNN-LSTM.

Run from the project root:

    python -m src.models.train_dl --quick        # 2 stations, 1 epoch
    python -m src.models.train_dl                # the real run

Trained on 2023, scored on 2024. The 2025 test split is never loaded.

On the GPU this uses mixed precision (autocast + GradScaler): the forward
pass runs in 16-bit, the weights stay in 32-bit. If the card runs out of
memory the batch size is halved and the epoch starts again.

Outputs
    models/dl/<model>.pt                         best weights
    models/dl/scaler.joblib                      the input scaler
    results/val/predictions/<model>_<task>_h<h>.parquet
    results/val/plots/<model>_training_curve.png
    results/val/metrics_<task>.csv               rows added
"""

import argparse
import json
import os
import time

# This must be set BEFORE torch is imported. It gives cuBLAS a fixed
# workspace so its matrix multiplications are deterministic; without it
# torch warns on every batch that determinism cannot be guaranteed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import load_config, project_path
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models import common as C
from src.models import sequences as S
from src.models.baselines import metric_row
from src.models.dl_models import MODELS, MultiTaskLoss, count_parameters

CFG = load_config()
DL = CFG["deep_learning"]
HORIZONS = CFG["horizons"]
N_CLASSES = len(CFG["target"]["class_names"])


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def pick_device():
    """config says "auto": use the GPU when there is one, otherwise the CPU."""
    want = DL["device"]
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        print("  config asks for cuda but no GPU is visible - using the CPU")
        want = "cpu"
    return torch.device(want)


def describe_device(device):
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | device {device}")
    if device.type == "cuda":
        i = device.index or 0
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU: {p.name}, {p.total_memory / 1024 ** 3:.1f} GB VRAM, "
              f"compute {p.major}.{p.minor}")


def set_all_seeds(seed):
    """
    Everything that can be fixed, is fixed. GPU LSTM kernels are still not
    bit-for-bit reproducible between runs - cuDNN picks non-deterministic
    algorithms for recurrent layers - so two runs can differ in the last
    decimal. That is noted in the report.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def build_targets(data, do_scale):
    """
    reg: the DO CHANGE from t to t+h, divided by the training standard
         deviation of DO. NaN where the target is missing.
    cls: the class 0/1/2, or -1 where the target is missing (-1 is the
         "ignore me" marker the loss looks for).
    """
    n = len(data.meta)
    reg = np.full((n, len(HORIZONS)), np.nan, dtype="float32")
    cls = np.full((n, len(HORIZONS)), -1, dtype="int64")
    now = data.meta["do_now"].to_numpy(dtype="float32")
    for j, h in enumerate(HORIZONS):
        y = data.meta[f"y_reg_{h}"].astype("float64").to_numpy()
        reg[:, j] = ((y - now) / do_scale).astype("float32")
        c = data.meta[f"y_cls_{h}"].astype("float64").to_numpy()
        ok = ~np.isnan(c)
        cls[ok, j] = c[ok].astype("int64")
    return reg, cls


def class_weights_from_train(cls_target):
    """
    w = n / (n_classes * count), the same rule as class_weight="balanced",
    computed per horizon on the TRAINING targets only.
    """
    w = np.ones((len(HORIZONS), N_CLASSES), dtype="float32")
    for j in range(len(HORIZONS)):
        col = cls_target[:, j]
        col = col[col >= 0]
        if not len(col):
            continue
        for c in range(N_CLASSES):
            count = (col == c).sum()
            w[j, c] = len(col) / (N_CLASSES * count) if count else 1.0
    return w


# ---------------------------------------------------------------------------
# One pass over a split
# ---------------------------------------------------------------------------
def to_device(x, station, device, pin):
    xt = torch.from_numpy(x)
    st = torch.from_numpy(station)
    if pin:
        xt, st = xt.pin_memory(), st.pin_memory()
    return (xt.to(device, non_blocking=True), st.to(device, non_blocking=True))


def run_epoch(model, data, reg_t, cls_t, loss_fn, device, batch_size,
              optimizer=None, scaler=None, use_amp=False, shuffle=False):
    """
    One pass. With an optimizer it trains; without one it only evaluates and
    also returns the predictions.
    """
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(data))
    if shuffle:
        np.random.shuffle(order)

    pin = device.type == "cuda" and DL["pin_memory"]
    total, n_batches = 0.0, 0
    reg_pred = np.empty((len(data), len(HORIZONS)), dtype="float32")
    cls_prob = np.empty((len(data), len(HORIZONS), N_CLASSES), dtype="float32")

    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        x, station = data.batch(ids)
        xt, st = to_device(x, station, device, pin)
        rt = torch.from_numpy(reg_t[ids]).to(device, non_blocking=True)
        ct = torch.from_numpy(cls_t[ids]).to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                reg_out, cls_out = model(xt, st)
                loss, _, _ = loss_fn(reg_out.float(), cls_out.float(), rt, ct)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)      # unscale BEFORE clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               DL["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               DL["grad_clip"])
                optimizer.step()
        else:
            reg_pred[ids] = reg_out.float().cpu().numpy()
            cls_prob[ids] = torch.softmax(cls_out.float(), dim=2).cpu().numpy()

        total += float(loss.item())
        n_batches += 1

    return total / max(n_batches, 1), reg_pred, cls_prob


def predictions_in_original_units(data, reg_pred, do_scale):
    """Undo the "predict the change, scaled" trick: DO(t) + change."""
    return data.meta["do_now"].to_numpy()[:, None] + reg_pred * do_scale


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------
def train_model(name, train_data, val_data, reg_tr, cls_tr, reg_va, cls_va,
                do_scale, device, args):
    n_stations = int(max(max(train_data.station_ids.values()),
                         max(val_data.station_ids.values())) + 1)
    model = MODELS[name](S.N_CHANNELS, n_stations).to(device)
    loss_fn = MultiTaskLoss(class_weights_from_train(cls_tr)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=DL["lr_factor"], patience=DL["lr_patience"])
    use_amp = device.type == "cuda" and DL["mixed_precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    batch_size = args.batch_size or DL["batch_size"]

    print(f"\n=== {name} ===")
    print(f"  {count_parameters(model):,} parameters | "
          f"{S.N_CHANNELS} channels x {S.SEQ_LEN} hours | "
          f"batch {batch_size} | mixed precision {use_amp}")
    print(f"  train windows {len(train_data):,} | val windows {len(val_data):,}")

    max_epochs = 1 if args.quick else DL["max_epochs"]
    history, best = [], {"val_loss": np.inf, "epoch": 0, "state": None}
    bad_epochs = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        while True:                       # retry smaller if the GPU runs out
            try:
                train_loss, _, _ = run_epoch(
                    model, train_data, reg_tr, cls_tr, loss_fn, device,
                    batch_size, optimizer, scaler, use_amp, shuffle=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size = max(32, batch_size // 2)
                print(f"    out of GPU memory - retrying with batch "
                      f"{batch_size}")

        val_loss, reg_pred, cls_prob = run_epoch(
            model, val_data, reg_va, cls_va, loss_fn, device, batch_size)
        epoch_time = time.perf_counter() - t0

        # a quick look at the 24 h horizon, the headline number
        j24 = HORIZONS.index(24)
        do_pred = predictions_in_original_units(val_data, reg_pred, do_scale)
        rm = regression_metrics(val_data.meta["y_reg_24"], do_pred[:, j24])
        cm = classification_metrics(val_data.meta["y_cls_24"],
                                    cls_prob[:, j24, :].argmax(1))
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if device.type == "cuda" else 0.0)

        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_rmse_24": rm["rmse"],
                        "val_f1_24": cm["f1_macro"], "seconds": epoch_time,
                        "peak_gpu_gb": peak,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"  epoch {epoch:>2}  train {train_loss:.4f}  val {val_loss:.4f}"
              f"  RMSE24 {rm['rmse']:.4f}  macroF1_24 {cm['f1_macro']:.4f}"
              f"  {epoch_time / 60:.1f} min  peak {peak:.2f} GB")

        if epoch == 1 and epoch_time > DL["max_minutes_per_epoch"] * 60:
            print(f"\n  STOP: one epoch took {epoch_time / 60:.1f} minutes, "
                  f"more than the {DL['max_minutes_per_epoch']} minute limit "
                  f"in config.yaml.")
            print("  Tell the user before continuing.")
            raise SystemExit(2)

        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= DL["early_stopping_patience"]:
                print(f"  early stopping: no improvement for "
                      f"{bad_epochs} epochs")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, pd.DataFrame(history), best, batch_size


# ---------------------------------------------------------------------------
# Evaluation and saving
# ---------------------------------------------------------------------------
def evaluate_and_save(name, model, val_data, reg_va, cls_va, do_scale,
                      device, loss_fn, batch_size, hist, best, quick):
    _, reg_pred, cls_prob = run_epoch(model, val_data, reg_va, cls_va,
                                      loss_fn, device, batch_size)
    do_pred = predictions_in_original_units(val_data, reg_pred, do_scale)
    meta = val_data.meta
    persist = meta["do_now"].to_numpy()

    rows = {"regression": [], "classification": []}
    total_seconds = float(hist["seconds"].sum())
    extra = {"epochs": int(hist["epoch"].max()),
             "best_epoch": int(best["epoch"]),
             "seconds_per_epoch": round(float(hist["seconds"].mean()), 1),
             "peak_gpu_gb": round(float(hist["peak_gpu_gb"].max()), 2)}

    for j, h in enumerate(HORIZONS):
        y_reg = meta[f"y_reg_{h}"]
        m = regression_metrics(y_reg, do_pred[:, j], persist)
        if not quick:
            C.save_predictions(name, "regression", h, meta, do_pred[:, j])
        row = metric_row(name, "regression", h, m, total_seconds)
        row.update(extra)
        rows["regression"].append(row)

        proba = cls_prob[:, j, :]
        pred_cls = proba.argmax(1)
        m = classification_metrics(meta[f"y_cls_{h}"], pred_cls, proba)
        if not quick:
            C.save_predictions(name, "classification", h, meta, pred_cls, proba)
        row = metric_row(name, "classification", h, m, total_seconds)
        row.update(extra)
        rows["classification"].append(row)

    return rows, extra


def plot_curve(name, hist):
    folder = project_path(CFG["models"]["dirs"]["val_plots"])
    os.makedirs(folder, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], "-o", label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], "-o", label="validation")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_rmse_24"], "-o", color="tab:red")
    axes[1].set_ylabel("validation RMSE at 24 h (mg/L)")
    axes[2].plot(hist["epoch"], hist["val_f1_24"], "-o", color="tab:green")
    axes[2].set_ylabel("validation macro F1 at 24 h")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    fig.suptitle(f"{name}: training on 2023, watched on 2024")
    fig.tight_layout()
    path = os.path.join(folder, f"{name}_training_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--quick", action="store_true",
                    help="2 stations, 1 epoch: a smoke test of the whole path")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    set_all_seeds(CFG["seed"])
    device = pick_device()
    describe_device(device)

    stations = None
    if args.quick:
        all_st = pd.read_parquet(project_path(CFG["paths"]["stations"]),
                                 columns=None) if False else None
        import pandas as _pd
        all_st = _pd.read_csv(project_path(CFG["paths"]["stations"]),
                              dtype={"site_no": str})["site_no"].tolist()
        stations = sorted(all_st)[:2]
        print(f"QUICK MODE: stations {stations}, 1 epoch, nothing saved")

    print("\nBuilding sequences...")
    t0 = time.perf_counter()
    train_data = S.prepare("train", stations=stations)
    val_data = S.prepare("val", scaler=train_data.scaler,
                         medians=train_data.medians, stations=stations)
    if not args.quick:
        print(f"  scaler saved to {S.save_scaler(train_data.scaler, train_data.medians)}")
    do_scale = float(train_data.scaler.scale_[S.CHANNELS.index("do_mgl")])
    print(f"  {len(train_data):,} train and {len(val_data):,} val windows, "
          f"{S.N_CHANNELS} channels, built in {time.perf_counter() - t0:.0f}s")
    print(f"  DO standard deviation on train: {do_scale:.3f} mg/L "
          f"(the regression head predicts the change in these units)")

    reg_tr, cls_tr = build_targets(train_data, do_scale)
    reg_va, cls_va = build_targets(val_data, do_scale)

    all_rows = {"regression": [], "classification": []}
    summary = {}
    for name in args.models:
        model, hist, best, batch_size = train_model(
            name, train_data, val_data, reg_tr, cls_tr, reg_va, cls_va,
            do_scale, device, args)
        loss_fn = MultiTaskLoss(class_weights_from_train(cls_tr)).to(device)
        rows, extra = evaluate_and_save(name, model, val_data, reg_va, cls_va,
                                        do_scale, device, loss_fn, batch_size,
                                        hist, best, args.quick)
        summary[name] = extra
        for task in all_rows:
            all_rows[task] += rows[task]

        if not args.quick:
            folder = project_path(DL["dirs"]["dl_models"])
            os.makedirs(folder, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "channels": S.CHANNELS, "seq_len": S.SEQ_LEN,
                        "do_scale": do_scale, "config": CFG["models"][name]},
                       os.path.join(folder, f"{name}.pt"))
            hist.to_csv(os.path.join(folder, f"{name}_history.csv"), index=False)
            print(f"  saved {folder}\\{name}.pt and {plot_curve(name, hist)}")

        j = HORIZONS.index(24)
        r = [x for x in rows["regression"] if x["horizon"] == 24][0]
        c = [x for x in rows["classification"] if x["horizon"] == 24][0]
        print(f"  24 h: RMSE {r['rmse']:.4f} (skill "
              f"{r['skill_vs_persistence']:+.3f}) | macro F1 "
              f"{c['f1_macro']:.4f} | Low recall {c['recall_Low']:.3f}")

    if not args.quick:
        for task in all_rows:
            print(f"\nWrote {C.update_metrics_csv(task, all_rows[task])}")
        with open(os.path.join(project_path(DL["dirs"]["dl_models"]),
                               "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nNext: python -m src.evaluation.step4_report")
    else:
        print("\nQuick mode finished: the whole path works.")


if __name__ == "__main__":
    main()
