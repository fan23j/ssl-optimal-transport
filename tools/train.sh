

#CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg ../experiments/finetune/classify_anything_mixed_ot_coco_cifar.yaml
CUDA_VISIBLE_DEVICES=1 python3 train.py --cfg ../experiments/finetune/classify_anything_mixed_ot_nuswide_cifar.yaml