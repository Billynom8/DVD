@echo off
cd /d "%~dp0.."
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Set CUDA_VISIBLE_DEVICES if necessary, e.g., set CUDA_VISIBLE_DEVICES=1
uv run python utils/depth2normal.py ^
    --data_path %PATH_TO_VKITTI_DATA% ^
    --batch_size 10 ^
    --scenes 01 02 06 18 20
