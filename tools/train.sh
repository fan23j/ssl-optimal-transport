#python3  train.py --cfg ../experiments/pretrain/swav_cifar10_base.yaml --local_rank 0
#CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ../experiments/finetune/classify_anything_cifar10_single.yaml
#CUDA_VISIBLE_DEVICES=1 python3 train.py --cfg ../experiments/pretrain/simclr_cifar100_gw_supervision.yaml
#CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ../experiments/pretrain/mae_cifar10_gw_supervision.yaml
CUDA_VISIBLE_DEVICES=1 python3 train.py --cfg ../experiments/finetune/coco_multi.yaml