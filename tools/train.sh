CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ../experiments/pretrain/mae_cifar10_mse.yaml
#CUDA_VISIBLE_DEVICES=1 python3 train.py --cfg ../experiments/pretrain/simclr_cifar100_infonce+sinkhorn.yaml 