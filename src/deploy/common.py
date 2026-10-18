"""
STEP 12.0 - the plumbing every deployment script shares.

Loads config_deploy.yaml alongside config.yaml, resolves the deployment
paths, works out the model version id, picks the device, and times the
stages. Nothing here is specific to one script.

    from src.deploy import common as D
    D.CFG            # config.yaml, unchanged
    D.DEP            # the `deploy:` block of config_deploy.yaml
    D.model_version() -> "final-1a2b3c4d5e6f"
"""

import hashlib
import os
import sys
import time

import yaml

from src.config import ROOT, load_config, project_path

DEPLOY_CONFIG_PATH = os.path.join(ROOT, "config_deploy.yaml")


def load_deploy_config(path=DEPLOY_CONFIG_PATH):
    """
    Both configs, together. config.yaml is the frozen science and is read
    through the project's own loader; config_deploy.yaml is the deployment
    layer. They are kept separate on purpose: nothing in the deployment can
    silently change a setting the paper depends on.
    """
    with open(path, "r", encoding="utf-8") as f:
        dep = yaml.safe_load(f)
    return {"base": load_config(), "deploy": dep["deploy"]}


_BOTH = load_deploy_config()
CFG = _BOTH["base"]
DEP = _BOTH["deploy"]

HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
LOW_CLASS = CLASS_NAMES.index("Low")
LOW = CFG["target"]["thresholds"]["low"]
DO = CFG["target"]["column"]
MEMBERS = CFG["ensemble"]["members"]
SEED = CFG["seed"]
FINAL_SYSTEM = CFG["final_system"]

TEST26 = DEP["test_2026"]
DATA = DEP["data"]
FETCH = DEP["fetch"]
RUN = DEP["run"]
MON = DEP["monitor"]
RETRAIN = DEP["retrain"]
COMPUTE = DEP["compute"]
BOOT = DEP["bootstrap"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def dpath(*parts, **fmt):
    """
    A path from config_deploy.yaml, absolute, with `{mode}` style
    placeholders filled in and the parent directory created.
    """
    parts = [p.format(**fmt) if fmt else p for p in parts]
    path = project_path(*parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def ensure_dir(*parts):
    path = project_path(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def deployed_folder():
    """The folder the deployed members live in (models/final)."""
    from src.models.predict_service import STAGE_DIRS
    return project_path(STAGE_DIRS[DEP["models"]["stage"]])


# ---------------------------------------------------------------------------
# The model version id
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_version(folder=None):
    """
    The id that goes into every forecast row.

    It is the SHA-256 over the per-file hashes of the deployed model folder,
    so it changes if and only if a model file changes - which is exactly
    when a forecast stops being comparable with an earlier one. Files the
    deployment itself writes (the model card, the timing log) are excluded,
    so documenting the version cannot change it.
    """
    folder = folder or deployed_folder()
    skip = {"MODEL_CARD.md", "training_times.json"}
    if not os.path.isdir(folder):
        return f"{DEP['models']['version_prefix']}-missing"
    digest = hashlib.sha256()
    for name in sorted(os.listdir(folder)):
        if name in skip or name.startswith("."):
            continue
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            digest.update(name.encode("utf-8"))
            digest.update(sha256_file(full).encode("ascii"))
    n = int(DEP["models"]["version_digest_chars"])
    return f"{DEP['models']['version_prefix']}-{digest.hexdigest()[:n]}"


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def report_device(what="inference"):
    """
    Print what we are actually running on, and say so out loud when CUDA is
    missing. The project's own rule: a silent fall back to the CPU for the
    deep models is a bug, not a convenience.
    """
    try:
        import torch
    except ImportError:
        print(f"  [{what}] torch is not installed - deep members unavailable")
        return "cpu", False
    want = COMPUTE["deep_device"]
    available = bool(torch.cuda.is_available())
    if available and want in ("auto", "cuda"):
        name = torch.cuda.get_device_name(0)
        if COMPUTE.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        print(f"  [{what}] CUDA available: {name} (torch {torch.__version__}, "
              f"cudnn.benchmark="
              f"{torch.backends.cudnn.benchmark})")
        return "cuda", True
    print(f"  [{what}] CUDA NOT available (torch {torch.__version__}) - "
          f"RUNNING ON CPU. Install the CUDA build: see README.md "
          f"('pip install torch==2.2.2 --index-url "
          f"https://download.pytorch.org/whl/cu121').")
    return "cpu", False


def require_cuda_or_explain():
    """
    Used by the entry points that do real deep-model work. Returns True when
    the GPU is there; otherwise prints the README instructions and returns
    False so the caller can stop instead of quietly using the CPU.
    """
    try:
        import torch
    except ImportError:
        print("torch is not installed. See README.md -> Install.")
        return False
    if torch.cuda.is_available():
        return True
    print(
        "CUDA is not available, so the deep models would run on the CPU.\n"
        "This project's models were trained with the CUDA build of torch.\n"
        "To install it (from README.md):\n\n"
        "    pip uninstall -y torch\n"
        "    pip install torch==2.2.2 --index-url "
        "https://download.pytorch.org/whl/cu121\n"
        "    python -c \"import torch; print(torch.cuda.is_available())\"\n\n"
        "cu121 is correct for any driver from CUDA 12.1 upwards.\n"
        "Re-run with --allow-cpu to proceed on the CPU anyway.")
    return False


def tree_n_jobs():
    return int(COMPUTE["tree_n_jobs"])


def feature_workers():
    """How many processes to build features with."""
    n = int(COMPUTE["feature_workers"])
    return os.cpu_count() or 1 if n in (-1, 0) else n


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Stages:
    """
    How long each stage took, printed at the end of a run.

        st = Stages()
        with st("fetch"):
            ...
        st.report()
    """

    def __init__(self):
        self.times = {}
        self._name = None
        self._t0 = None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times[self._name] = (self.times.get(self._name, 0.0)
                                  + time.perf_counter() - self._t0)
        return False

    def add(self, name, seconds):
        self.times[name] = self.times.get(name, 0.0) + float(seconds)

    def report(self, title="Stage timings"):
        if not self.times:
            return
        width = max(len(k) for k in self.times)
        total = sum(self.times.values())
        print(f"\n{title}")
        for k, v in self.times.items():
            print(f"  {k:<{width}}  {v:>8.1f}s  ({v / 60:>5.2f} min)")
        print(f"  {'TOTAL':<{width}}  {total:>8.1f}s  ({total / 60:>5.2f} min)")


def log(*args):
    print(*args, flush=True)


def shortpath(path):
    """
    A path relative to the project root for printing - or the path itself
    when it is somewhere else entirely. On Windows `os.path.relpath` raises
    when the two are on different drives, and a run must never die on a log
    line.
    """
    try:
        return os.path.relpath(path, project_path("."))
    except ValueError:
        return str(path)


def stations_table():
    """The 30 deployed stations - the same stations.csv the paper used."""
    import pandas as pd
    return pd.read_csv(project_path(CFG["paths"]["stations"]),
                       dtype={"site_no": str})


def station_codes():
    """
    station -> integer code, exactly as build_features assigned them: the
    position in the sorted list of stations in the hourly panel. The deep
    models' embedding rows depend on this, so it must not be recomputed from
    a different list.
    """
    import pandas as pd
    panel_stations = sorted(stations_table()["site_no"].astype(str))
    return {s: i for i, s in enumerate(panel_stations)}
