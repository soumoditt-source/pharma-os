$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/preflight.py --keep-server --open-browser
