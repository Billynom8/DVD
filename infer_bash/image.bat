@echo off
cd /d "%~dp0.."
set CKPT=ckpt
:: Update this path to your dataset location
set IMAGE_BASE_DATA_DIR=C:\AI2\DVD\dataset\eval\depth

uv run python test_script/test_from_trained_all_img.py --ckpt %CKPT% --base_data_dir %IMAGE_BASE_DATA_DIR%
