export CUDA_VISIBLE_DEVICES=2,3,4,5
python3 main.py --lr 0.001 --pretrained -a mobilenet_v2  --block_size 4  --initial_threshold 0.9 --final_threshold 0.1 --initial_warmup 1 --final_warmup 4 /datasets/imagenet


