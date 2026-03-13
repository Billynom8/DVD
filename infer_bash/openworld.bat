@echo off
cd /d "%~dp0.."

set CKPT=ckpt

:: You could increase the resolution here but expect slower inference speed.
set HEIGHT=480
set WIDTH=640

:: You might increase these two number to ensure more stable scale variation but expect slower inference speed
set WINDOW_SIZE=41
set OVERLAP=11

:: set INPUT_VIDEO=demo/drone.mp4
:: python test_script/test_single_video.py --ckpt %CKPT% --input_video %INPUT_VIDEO% --height %HEIGHT% --width %WIDTH% --window_size %WINDOW_SIZE% --overlap %OVERLAP%

set INPUT_VIDEO=demo/AWO_clip2_V1-1574.mp4

uv run python test_script/test_single_video.py --ckpt %CKPT% --input_video %INPUT_VIDEO% --height %HEIGHT% --width %WIDTH% --window_size %WINDOW_SIZE% --overlap %OVERLAP%
pause