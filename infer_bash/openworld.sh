CKPT='ckpt/DVD'
INPUT_VIDEO='demo/raw_run_cropped_src.mp4'
HEIGHT=720
WIDTH=960
WINDOW_SIZE=45 

python test_script/test_single_video.py --ckpt $CKPT --input_video $INPUT_VIDEO  --height $HEIGHT --width $WIDTH --window_size $WINDOW_SIZE