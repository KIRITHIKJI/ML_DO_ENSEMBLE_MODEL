"""
STEP 4.3 / 4.4, extended in STEP 5 - train and evaluate the deep models.

Run from the project root:

    python -m src.models.train_dl --quick                  # smoke test
    python -m src.models.train_dl                          # step-4 settings
    python -m src.models.train_dl --variant reg_hybrid     # one step-5 variant
    python -m src.models.train_dl --final                  # 3 seeds, averaged

Trained on 2023, scored on 2024. The 2025 test split is never loaded.

A "variant" is a set of overrides from `dl_variants:` in config.yaml - more
dropout, AdamW, the engineered-feature input, a longer window, a smaller
hidden layer. `step4` is the empty variant, i.e. exactly what step 4 ran.

On the GPU this uses mixed precision (autocast + GradScaler): the forward
pass runs in 16-bit, the weights stay in 32-bit. If the card runs out of
memory the batch size is halved and the epoch starts again.

Outputs
    models/dl/<model>.pt or <model>_seed<k>.pt
    models/dl/scaler.joblib
    results/val/predictions/<model>_<task>_h<h>.parquet
    results/val/plots/<model>_training_curve.png
    results/val/metrics_<task>.csv                rows added or replaced
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
SEARCH = CFG["dl_search"]
HORIZONS = CFG["horizons"]
N_CLASSES = len(CFG["target"]["class_names"])


# ---------------------------------------------------------------------------
# Settings and setup
# ---------------------------------------------------------------------------
def variant_settings(variant):
    """
    The deep-learning settings with one variant's overrides applied on top.
    `step4` has no overrides, so it reproduces step 4 exactly.
    """
    if variant not in CFG["dl_variants"]:
        raise SystemExit(f"unknown variant {variant!r}; "
                         f"config has {list(CFG['dl_variants'])}")
    settings = dict(DL)
    settings.update(CFG["dl_variants"][variant] or {})
    settings["variant"] = variant
    return settings


def architecture_overrides(settings, model_name):
    """The part of a variant that changes the network itself."""
    out = {}
    for key in ["dropout", "hidden_size", "feature_hidden"]:
        if key in (CFG["dl_variants"][settings["variant"]] or {}):
            out[key] = settings[key]
    return out


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
        p = torch.cuda.get_device_properties(device.index or 0)
        print(f"  GPU: {p.name}, {p.total_memory / 1024 ** 3:.1f} GB VRAM, "
              f"compute {p.major}.{p.minor}")


def set_all_seeds(seed):
    """
    Everything that can be fixed, is fixed. GPU LSTM kernels are still not
    bit-for-bit reproducible between runs - cuDNN picks non-deterministic
    algorithms for recurrent layers - so two runs can differ in the last
    decimal. That is noted in the report, and is why step 5 averages three
    seeds instead of trusting one.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_data(settings, stations=None, save=True):
    """
    Sequences for train and validation under these settings. Returns
    (train_data, val_data, do_scale).
    """
    seq_len = settings["sequence_length"]
    with_features = settings["use_engineered_features"]
    t0 = time.perf_counter()
    train_data = S.prepare("train", stations=stations, seq_len=seq_len,
                           with_features=with_features)
    val_data = S.prepare("val", scaler=train_data.scaler,
                         medians=train_data.medians, stations=stations,
                         seq_len=seq_len, with_features=with_features,
                         feature_scaler=train_data.feature_scaler,
                         feature_medians=train_data.feature_medians)
    if save:
        S.save_scaler(train_data.scaler, train_data.medians, seq_len,
                      train_data.feature_scaler, train_data.feature_medians,
                      train_data.feature_names)
    do_scale = float(train_data.scaler.scale_[S.CHANNELS.index("do_mgl")])
    n_feat = 0 if train_data.features is None else train_data.features.shape[1]
    print(f"  {len(train_data):,} train and {len(val_data):,} val windows | "
          f"{seq_len} h x {S.N_CHANNELS} channels"
          + (f" + {n_feat} engineered features" if n_feat else "")
          + f" | built in {time.perf_counter() - t0:.0f}s")
    return train_data, val_data, do_scale


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
        x, station, feats = data.batch(ids)
        xt, st = torch.from_numpy(x), torch.from_numpy(station)
        ft = None if feats is None else torch.from_numpy(feats)
        if pin:
            xt, st = xt.pin_memory(), st.pin_memory()
            ft = None if ft is None else ft.pin_memory()
        xt = xt.to(device, non_blocking=True)
        st = st.to(device, non_blocking=True)
        ft = None if ft is None else ft.to(device, non_blocking=True)
        rt = torch.from_numpy(reg_t[ids]).to(device, non_blocking=True)
        ct = torch.from_numpy(cls_t[ids]).to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                reg_out, cls_out = model(xt, st, ft)
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
def make_optimizer(model, settings):
    """Adam, or AdamW with weight decay when the variant asks for it."""
    lr = settings["learning_rate"]
    if settings.get("optimizer", "adam").lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=settings.get("weight_decay", 0.0))
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_model(name, settings, train_data, val_data, reg_tr, cls_tr,
                reg_va, cls_va, do_scale, device, seed, max_epochs=None,
                batch_size=None, verbose=True):
    """Train one model under one set of settings and one seed."""
    set_all_seeds(seed)
    n_stations = int(max(max(train_data.station_ids.values()),
                         max(val_data.station_ids.values())) + 1)
    n_features = (0 if train_data.features is None
                  else train_data.features.shape[1])
    arch = architecture_overrides(settings, name)
    model = MODELS[name](S.N_CHANNELS, n_stations, cfg=arch,
                         n_features=n_features).to(device)
    loss_fn = MultiTaskLoss(class_weights_from_train(cls_tr)).to(device)
    optimizer = make_optimizer(model, settings)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=DL["lr_factor"], patience=DL["lr_patience"])
    use_amp = device.type == "cuda" and DL["mixed_precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    batch_size = batch_size or settings["batch_size"]
    max_epochs = max_epochs or settings["max_epochs"]
    patience = settings["early_stopping_patience"]

    if verbose:
        print(f"  {count_parameters(model):,} parameters | optimizer "
              f"{settings.get('optimizer', 'adam')} lr {settings['learning_rate']}"
              f" | dropout {arch.get('dropout', 'default')}"
              f" | batch {batch_size} | patience {patience}")

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
                print(f"    out of GPU memory - retrying with batch {batch_size}")

        val_loss, reg_pred, cls_prob = run_epoch(
            model, val_data, reg_va, cls_va, loss_fn, device, batch_size)
        epoch_time = time.perf_counter() - t0

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
                        "val_f1_24": cm["f1_macro"],
                        "val_recall_low_24": cm["recall_Low"],
                        "seconds": epoch_time, "peak_gpu_gb": peak,
                        "lr": optimizer.param_groups[0]["lr"]})
        if verbose:
            print(f"  epoch {epoch:>2}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  RMSE24 {rm['rmse']:.4f}  "
                  f"macroF1_24 {cm['f1_macro']:.4f}  {epoch_time / 60:.1f} min"
                  f"  peak {peak:.2f} GB")

        if epoch == 1 and epoch_time > DL["max_minutes_per_epoch"] * 60:
            print(f"\n  STOP: one epoch took {epoch_time / 60:.1f} minutes, "
                  f"more than the {DL['max_minutes_per_epoch']} minute limit.")
            raise SystemExit(2)

        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stopping: no improvement for "
                          f"{bad_epochs} epochs")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, pd.DataFrame(history), best, batch_size, loss_fn


# ---------------------------------------------------------------------------
# Prediction and scoring
# ---------------------------------------------------------------------------
def predict(model, data, reg_t, cls_t, loss_fn, device, batch_size, do_scale):
    """Validation predictions in the original units, plus class probabilities."""
    val_loss, reg_pred, cls_prob = run_epoch(model, data, reg_t, cls_t,
                                             loss_fn, device, batch_size)
    return val_loss, predictions_in_original_units(data, reg_pred, do_scale), cls_prob


def score(meta, do_pred, cls_prob, name, total_seconds, extra, save=True):
    """Metrics rows for every horizon, and optionally the prediction files."""
    rows = {"regression": [], "classification": []}
    persist = meta["do_now"].to_numpy()
    for j, h in enumerate(HORIZONS):
        m = regression_metrics(meta[f"y_reg_{h}"], do_pred[:, j], persist)
        if save:
            C.save_predictions(name, "regression", h, meta, do_pred[:, j])
        row = metric_row(name, "regression", h, m, total_seconds)
        row.update(extra)
        rows["regression"].append(row)

        proba = cls_prob[:, j, :]
        m = classification_metrics(meta[f"y_cls_{h}"], proba.argmax(1), proba)
        if save:
            C.save_predictions(name, "classification", h, meta,
                               proba.argmax(1), proba)
        row = metric_row(name, "classification", h, m, total_seconds)
        row.update(extra)
        rows["classification"].append(row)
    return rows


def plot_curve(name, hist, suffix=""):
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
    fig.suptitle(f"{name}{suffix}: training on 2023, watched on 2024")
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
    ap.add_argument("--variant", default="step4",
                    help="a key of dl_variants: in config.yaml")
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--quick", action="store_true",
                    help="2 stations, 1 epoch: a smoke test of the whole path")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    device = pick_device()
    describe_device(device)
    settings = variant_settings(args.variant)

    stations = None
    if args.quick:
        stations = sorted(pd.read_csv(project_path(CFG["paths"]["stations"]),
                                      dtype={"site_no": str})["site_no"])[:2]
        print(f"QUICK MODE: stations {stations}, 1 epoch, nothing saved")

    print(f"\nVariant: {args.variant} | seed {args.seed}")
    train_data, val_data, do_scale = build_data(settings, stations,
                                                save=not args.quick)
    reg_tr, cls_tr = build_targets(train_data, do_scale)
    reg_va, cls_va = build_targets(val_data, do_scale)

    all_rows = {"regression": [], "classification": []}
    summary = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        model, hist, best, batch_size, loss_fn = train_model(
            name, settings, train_data, val_data, reg_tr, cls_tr, reg_va,
            cls_va, do_scale, device, args.seed,
            max_epochs=1 if args.quick else None, batch_size=args.batch_size)
        _, do_pred, cls_prob = predict(model, val_data, reg_va, cls_va,
                                       loss_fn, device, batch_size, do_scale)
        extra = {"epochs": int(hist["epoch"].max()),
                 "best_epoch": int(best["epoch"]),
                 "seconds_per_epoch": round(float(hist["seconds"].mean()), 1),
                 "peak_gpu_gb": round(float(hist["peak_gpu_gb"].max()), 2),
                 "variant": args.variant, "seeds": str(args.seed)}
        rows = score(val_data.meta, do_pred, cls_prob, name,
                     float(hist["seconds"].sum()), extra, save=not args.quick)
        summary[name] = extra
        for task in all_rows:
            all_rows[task] += rows[task]

        if not args.quick:
            folder = project_path(DL["dirs"]["dl_models"])
            os.makedirs(folder, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "channels": S.CHANNELS,
                        "seq_len": settings["sequence_length"],
                        "do_scale": do_scale, "variant": args.variant,
                        "arch": architecture_overrides(settings, name)},
                       os.path.join(folder, f"{name}.pt"))
            hist.to_csv(os.path.join(folder, f"{name}_history.csv"), index=False)
            print(f"  saved {plot_curve(name, hist)}")

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
    else:
        print("\nQuick mode finished: the whole path works.")


if __name__ == "__main__":
    main()
