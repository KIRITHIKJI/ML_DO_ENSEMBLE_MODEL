"""
Pack everything the two presentation notebooks need into ML_MAIN_V2_colab.zip,
ready to upload to Google Drive and open in Colab.

Run from the project root, AFTER notebook 2 has been run once with RUN_MODE = "train":

    python make_colab_zip.py

Upload the zip to the top folder of Google Drive ("My Drive"), then open the
notebooks in Colab. Nothing in the project is changed.
"""
import os
import zipfile

INCLUDE = [
    "config.yaml", "requirements.txt", "src",
    "data/processed", "data/interim/hourly.parquet", "data/raw/stations.csv",
    "results/test", "results/live_2026/walk_forward.csv",
    "models/ensemble/warning_thresholds.json", "models/ensemble/ablation_thresholds.json",
    "models/notebook_run", "notebooks",
]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "plots"}
COMPRESS = {".py", ".csv", ".json", ".yaml", ".md", ".ipynb", ".txt"}   # parquet / joblib / pt are already compressed

root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, "ML_MAIN_V2_colab.zip")
if not os.path.exists(os.path.join(root, "models", "notebook_run", "training_log.joblib")):
    print("NOTE: models/notebook_run has no finished training run yet - notebook 2 will only work in Colab "
          "with RUN_MODE = 'train'.")

n, size = 0, 0
with zipfile.ZipFile(out, "w") as z:
    for item in INCLUDE:
        path = os.path.join(root, item)
        if not os.path.exists(path):
            print("  missing, skipped:", item)
            continue
        files = [path] if os.path.isfile(path) else [
            os.path.join(d, f) for d, dirs, fs in os.walk(path)
            for f in fs if not (set(os.path.relpath(d, root).replace("\\", "/").split("/")) & SKIP_DIRS)]
        for f in files:
            rel = os.path.relpath(f, root).replace("\\", "/")
            how = zipfile.ZIP_DEFLATED if os.path.splitext(f)[1] in COMPRESS else zipfile.ZIP_STORED
            z.write(f, "ML_MAIN_V2/" + rel, compress_type=how)
            n += 1
            size += os.path.getsize(f)
print(f"Wrote {out}: {n} files, {size / 1e6:,.0f} MB before zipping, {os.path.getsize(out) / 1e6:,.0f} MB zip")
print("Next: upload it to the top folder of Google Drive and open the notebooks in Colab.")
