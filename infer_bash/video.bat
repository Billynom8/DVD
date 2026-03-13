@echo off
cd /d "%~dp0.."
set CKPT=ckpt
set FRAME_NUM=300
set NUM=10
:: Update this path to your dataset location
set VIDEO_BASE_DATA_DIR=C:\AI2\DVD\dataset

uv run python test_script/test_from_trained_all_vid.py --ckpt %CKPT% --frame %FRAME_NUM% --num %NUM% --base_data_dir %VIDEO_BASE_DATA_DIR%
