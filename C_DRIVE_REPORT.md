# C: Drive Space Report

Scanned and cleaned 2026-11-01. Phase 2 is complete.

| | |
|---|---|
| **C: free before deleting** | **34.05 GB** of 231.81 GB |
| **C: free after deleting** | **59.43 GB** |
| **Gained** | **25.38 GB** |

C: read 31.96 GB free during the scan and 34.05 GB just before deletion started. That change came from other activity on the machine.

## Phase 2 results

| Item | Removed (GB) | Method |
|---|---:|---|
| `NVIDIA\DXCache` | 13.52 | Deleted the files, kept the folder |
| pip cache | 6.14 | `pip cache purge` |
| npm `_cacache` | 4.61 | `npm cache clean --force` |
| Brave cache (added under "other unwanted") | 0.36 | Deleted the files; Brave was not running |
| Cached updater installers for Antigravity, Notion and LearnSquare (added) | 0.29 | Deleted `installer.exe`; each app downloads it again when it updates |
| `.cache\chroma`, `huggingface`, `whisper`, `torch` | 0.40 | Deleted the files |
| `%TEMP%`, files older than 24 hours | 0.02 | Deleted the files |
| WhatsApp crash dump (added) | < 0.01 | Deleted the file |

**Locked and skipped:** `%TEMP%\iu-14D2N.tmp\_unins.tmp`.

**Left in `%TEMP%`:** 0.21 GB of files under 24 hours old, plus the `claude` folder, because running programs may be using them.

**Environment variables set for your user:** `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `KERAS_HOME` and `XDG_CACHE_HOME`, all pointing to `A:\ml_cache`. Apps that are already open won't see them until restarted.

**Checked afterwards:** torch `2.2.2+cu121` imports with CUDA available, along with numpy, pandas, scikit-learn, xgboost, lightgbm, shap and streamlit. Nothing was installed.

The tables below show the Phase 1 scan as it stood before deletion.

## Locations

Sorted biggest first. "Extra" means the location was not on your scan list but turned up as a large folder.

| Location | Size (GB) | What it is | Safe to delete? |
|---|---:|---|---|
| `%LOCALAPPDATA%\Claude-Data` (extra) | 16.84 | Claude app data: settings, sign-in, Cowork tasks and session history. Its README says deleting it loses data permanently. | KEEP (use Claude > Settings > This computer > System > Storage to shrink it) |
| `%LOCALAPPDATA%\NVIDIA\DXCache` (extra) | 13.52 | NVIDIA DirectX shader cache from games and apps. Not the GPU driver. Rebuilds by itself; games may stutter briefly on first launch. | SAFE (optional, needs your explicit OK) |
| `%LOCALAPPDATA%\Google\Chrome\User Data` (extra) | 7.85 | Chrome profiles, plus a 3.98 GB on-device AI model that Chrome downloads again by itself. | KEEP |
| `...\Programs\Python\Python310` | 6.15 | Base Python 3.10 and 6.01 GB of global packages. **Both projects run from here.** | KEEP |
| `%LOCALAPPDATA%\pip\Cache` | 6.14 | Downloaded wheels, including two torch wheels of about 2.3 GB each. Installed packages are not affected. | SAFE |
| `%LOCALAPPDATA%\npm-cache\_cacache` | 4.61 | npm download cache. | SAFE |
| `%USERPROFILE%\.gemini` (extra) | 3.61 | Antigravity IDE data and browser profile, including a 0.89 GB `antigravity-backup`. | KEEP |
| `%USERPROFILE%\.vscode\extensions` (extra) | 3.40 | VS Code extensions. | KEEP (protected: VS Code) |
| `%LOCALAPPDATA%\ms-playwright` (extra) | 1.37 | Playwright browsers. Revision 1223 is pinned by Python 3.10's playwright; 1248 belongs to another tool. Restoring either needs an install. | KEEP |
| `%USERPROFILE%\.cache\puppeteer` | 0.57 | Puppeteer's Chrome builds. Restoring them needs an install. | KEEP |
| `C:\Windows\Temp` | 0.40 | System temp files. Inside a Windows folder, and this account has no admin rights to clear it. The real size may be larger. | KEEP (protected: Windows) |
| `%LOCALAPPDATA%\npm-cache\_npx` | 0.40 | Installed npx tools. Node processes are running, and removing these forces a reinstall. | KEEP |
| `%LOCALAPPDATA%\Python\pythoncore-3.14-64` | 0.39 | Base Python 3.14 and 0.26 GB of packages. | KEEP |
| `%TEMP%` | 0.22 | User temp files. Locked files will be skipped, and the `claude` subfolder is excluded because this session uses it. | SAFE |
| `%USERPROFILE%\.cache\chroma` | 0.16 | ChromaDB's embedding model. Downloads again on demand. | SAFE |
| `%USERPROFILE%\.cache\huggingface` | 0.12 | all-MiniLM-L6-v2 model and empty dataset stubs. Downloads again on demand. | SAFE |
| `%APPDATA%\npm` | 0.10 | Globally installed npm tools. | KEEP |
| `%USERPROFILE%\.cache\whisper` | 0.07 | Whisper `tiny.pt` model. Downloads again on demand. | SAFE |
| `%USERPROFILE%\.cache\torch` | 0.04 | ResNet-18 checkpoint. Downloads again on demand. | SAFE |
| `%APPDATA%\Python\Python314` | 0.02 | Python 3.14 user packages (Flask and others). | KEEP |
| `%USERPROFILE%\.matplotlib`, `.ipython`, `.cache\tooling`, `.cache\chrome-devtools-mcp` | < 0.01 | Small config and cache files. Nothing to gain. | KEEP |

**Total SAFE:** 11.37 GB from your list, plus 13.52 GB if you approve `DXCache`.

## Checked and not found

- Conda, Anaconda and Miniconda: not installed
- venv or `.venv` folders under `C:\Users\KIRITHIKJI`: none
- CUDA Toolkit in `C:\Program Files\NVIDIA GPU Computing Toolkit`: folder does not exist
- `%USERPROFILE%\.keras`, `%APPDATA%\jupyter`, Yarn cache, `catboost_info`, LightGBM and XGBoost caches: none
- Dataset or model files over 50 MB in Downloads: none (the whole folder is 0.02 GB)
- `node_modules` from your projects: none on C:. All 21 top-level folders belong to editor extensions, app tools or npm itself.

## Why no global packages are marked SAFE

`A:\ML_MAIN` and `A:\ML_MAIN_V2` have no virtual environment, so both import directly from Python 3.10's global packages. You also said not to reinstall anything, so uninstalling these could not be undone. torch alone is 4.26 GB, and it is the `2.2.2+cu121` GPU build the project was trained on.

Neither project imports some of the other packages, but they add up to under 0.5 GB, and other projects on A: (such as `CODES`, `PROJECT` and `WORKSPACE`) may use them. `pip uninstall` is therefore not part of Phase 2.

## Top 20 packages: Python 3.10

| Package | GB | | Package | GB |
|---|---:|---|---|---:|
| torch | 4.264 | | pandas | 0.049 |
| xgboost | 0.129 | | sympy | 0.046 |
| scipy | 0.108 | | pymupdf | 0.042 |
| playwright | 0.104 | | onnxruntime | 0.039 |
| llvmlite | 0.100 | | sklearn | 0.037 |
| googleapiclient | 0.093 | | numpy.libs | 0.036 |
| pyarrow | 0.073 | | babel | 0.029 |
| transformers | 0.070 | | debugpy | 0.029 |
| kubernetes | 0.067 | | numpy | 0.025 |
| plotly | 0.067 | | google | 0.023 |

Python 3.14's largest packages are scipy (0.107), numpy (0.031) and matplotlib (0.030).

## Notes

- **Scope:** the user profile totals 86.2 GB of the 199.6 GB used on C:. Windows, Program Files and installed apps were not scanned.
- **pip cache:** after the purge, any future torch reinstall downloads about 2.3 GB again.
- **npm:** your Phase 2 variables don't cover npm, so its cache will refill on C:.
- **Fixed cache paths:** ChromaDB and Puppeteer keep using `.cache` on C: even after the variables are set.
