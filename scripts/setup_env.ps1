# Environment bootstrap for the airline_S6E10 Kaggle campaign (single env, CPython 3.11).
# Idempotent. Run:  powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) { py -3.11 -m venv $venv }
& $py -m pip install --upgrade pip wheel setuptools
# Blackwell (RTX 5070 Ti, sm_120) needs cu128 or newer.
& $py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
& $py -m pip install "numpy<3" "pandas<3" polars scipy scikit-learn lightgbm xgboost catboost optuna pyarrow numba joblib tqdm kaggle matplotlib statsmodels
& $py -m pip install "pytabkit[models,hpo]"
& $py -m pip freeze | Out-File -Encoding utf8 (Join-Path $root "requirements-lock.txt")
& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"