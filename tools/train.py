from __future__ import absolute_import, division, print_function

import argparse
import os
import csv

import torch
import torch.distributed as dist
import torch.utils.data

import _init_paths
from config import cfg, update_config
from datasets.dataset_factory import get_dataset
from model.model import create_model, save_model
from train.train_factory import train_factory


def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    return args


def main(cfg, local_rank):
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    print("Creating model...")
    model = create_model(cfg.MODEL.NAME, cfg)

    # create weights output directory
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID)):
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID))

    num_gpus = torch.cuda.device_count()

    # check if distribute training is applicable
    if cfg.TRAIN.DISTRIBUTE:
        device = torch.device("cuda:%d" % local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=num_gpus, rank=local_rank
        )
    else:
        device = torch.device("cuda")

    # set up optimizer
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD
        )
    elif cfg.TRAIN.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM
        )
    else:
        NotImplementedError

    # load in pretrained weights if applicable
    start_epoch = 0
    if cfg.MODEL.INIT_WEIGHTS:
        print("load pretrained model from {}".format(cfg.MODEL.PRETRAINED))
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED), strict=False)

    # set up trainer code from train_factory
    Trainer = train_factory[cfg.TASK]
    trainer = Trainer(cfg, model, optimizer)

    if cfg.TRAIN.MASTER_BATCH_SIZE == -1:
        master_batch_size = cfg.TRAIN.BATCH_SIZE // len(cfg.GPUS)
    else:
        master_batch_size = cfg.TRAIN.MASTER_BATCH_SIZE
    rest_batch_size = cfg.TRAIN.BATCH_SIZE - master_batch_size
    chunk_sizes = [cfg.TRAIN.MASTER_BATCH_SIZE]
    for i in range(len(cfg.GPUS) - 1):
        slave_chunk_size = rest_batch_size // (len(cfg.GPUS) - 1)
        if i < rest_batch_size % (len(cfg.GPUS) - 1):
            slave_chunk_size += 1
        chunk_sizes.append(slave_chunk_size)
    trainer.set_device(device)

    # load in dataset
    print("Setting up data...")
    val_dataset, train_dataset = get_dataset(cfg)

    # Create validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # Create training dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_gpus, rank=local_rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE // num_gpus
        if cfg.TRAIN.DISTRIBUTE
        else cfg.TRAIN.BATCH_SIZE,
        shuffle=not cfg.TRAIN.DISTRIBUTE,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler if cfg.TRAIN.DISTRIBUTE else None,
    )

    print("Starting training...")
    log_file = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, "log.csv")
    header_prepared = False

    for epoch in range(start_epoch + 1, cfg.TRAIN.EPOCHS + 1):
        mark = epoch if cfg.TRAIN.SAVE_ALL_MODEL else "last"
        train_sampler.set_epoch(epoch)

        # run training script
        log_dict_train = trainer.train(epoch, train_loader)

        # run validation script
        if cfg.TRAIN.VAL_INTERVALS > 0 and epoch % cfg.TRAIN.VAL_INTERVALS == 0:
            save_model(
                os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, "model_{}.pth".format(mark)),
                epoch,
                model,
            )
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader)
        else:
            save_model(
                os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, "model_last.pth"), epoch, model
            )
            if epoch % 25 == 0:
                save_model(
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.EXP_ID, "model_epoch_{}.pth".format(epoch)
                    ),
                    epoch,
                    model,
                )
            log_dict_val = {}  # create an empty log_dict_val for non-validation epochs

        # prepare data row for CSV
        data_row = {"epoch": epoch, **log_dict_train, **log_dict_val}

        # write to CSV
        with open(log_file, mode="a") as file:
            writer = csv.DictWriter(file, fieldnames=data_row.keys())
            if not header_prepared:
                writer.writeheader()
                header_prepared = True
            writer.writerow(data_row)


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args.cfg)
    local_rank = args.local_rank
    main(cfg, local_rank)
