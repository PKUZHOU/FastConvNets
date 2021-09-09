export CUDA_VISIBLE_DEVICES=0,1
python3 main.py --lr 0.001 --pretrained -a mobilenet_v2 --sparsity 0.5 --block_size 4 /datasets/imagenet


