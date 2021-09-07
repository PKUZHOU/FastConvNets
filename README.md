# Reproduce Fast ConvNets @CVPR 2020

## Requirements

- `pip install -r requirements.txt`
- `download imagenets if doing training or evaluation`

## Files

- `main.py`: borrow from https://github.com/pytorch/examples/blob/master/imagenet, now supports evaluation and dense training. Sparse training is yet to implement.
- `mobilenet_v2.py`: define the sparse mobilenet model.
- `run.sh`: an example to show how to run main.py.

## Training 

- TODO

## Evaluation

- As shown in `run.sh`,  `python3 main.py --gpu 0 -e --pretrained -a mobilenet_v2 --sparsity 0.9 --block_size 2 /datasets/imagenet`, -e for evaluation, --pretrained for loading the dense model trained on imagenet and directly convert it to sparse model.

## Generate sparse models

- `python3 mobilenet_v2.py`
- modify the `sparsity` and `block_size` in the file to genenrate different models.