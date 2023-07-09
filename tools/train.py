from __future__ import absolute_import, division, print_function

import argparse
import os
import apex

import torch
import torch.distributed as dist
import torch.utils.data

import _init_paths
from logger import Logger
from config import cfg, update_config
from datasets.dataset_factory import get_dataset
from model.model import create_model, save_model, load_model
from model.backbones.swav_resnet import resnet50
from train.train_factory import train_factory
from train.scheduler_factory import OptimizerSchedulerFactory


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

    # model = resnet50(
    #     normalize=True,
    #     hidden_mlp=2048,
    #     output_dim=128,
    #     nmb_prototypes=100,
    # )

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

    # set up logger
    logger = Logger(cfg)

    # load in dataset
    print("Setting up data...")
    train_dataset, val_dataset = get_dataset(cfg)

    # Create validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=cfg.TEST.DROP_LAST,
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

    # set up optimizer and scheduler
    scheduler_factory = OptimizerSchedulerFactory(cfg, model, train_loader)
    optimizer, lr_scheduler = scheduler_factory.create()

    # load in pretrained weights if applicable
    start_epoch = 0
    if cfg.MODEL.INIT_WEIGHTS:
        print("load pretrained model from {}".format(cfg.MODEL.PRETRAINED))
        model, optimizer, lr_scheduler, start_epoch = load_model(
            cfg,
            model,
            optimizer,
            lr_scheduler,
        )

    model.to("cuda")
    # init mixed precision
    if cfg.TRAIN.USE_MIXED_PRECISION:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        print("Initializing mixed precision done.")

    # set up trainer code from train_factory
    Trainer = train_factory[cfg.TASK]
    trainer = Trainer(cfg, model, optimizer=optimizer, lr_scheduler=lr_scheduler)

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

    print("Starting training...")
    best = {
        "ACC@1": 0.0,
        "ACC@5": 0.0,
    }

    for epoch in range(start_epoch + 1, cfg.TRAIN.EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        # run training script
        log_dict_train = trainer.train(epoch, train_loader)
        logger.write("epoch: {} |".format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary("train_{}".format(k), v, epoch)
            logger.write("{} {:8f} | ".format(k, v))
        # run validation script
        if cfg.TRAIN.VAL_INTERVALS > 0 and epoch % cfg.TRAIN.VAL_INTERVALS == 0:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader)

            for k, v in log_dict_val.items():
                if k == "imgs":
                    logger.write_image(k, v, epoch)
                elif k == "figure":
                    logger.add_figure(k, v.gcf(), epoch)
                else:
                    logger.scalar_summary("val_{}".format(k), v, epoch)
                    logger.write("{} {:8f} | ".format(k, v))

            if log_dict_val["ACC@1"] > best["ACC@1"]:
                best["ACC@1"] = log_dict_val["ACC@1"]
                best["ACC@5"] = log_dict_val["ACC@5"]
                save_model(
                    os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, "model_best.pth"),
                    epoch,
                    model,
                    optimizer,
                    lr_scheduler,
                )

        save_model(
            os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, "model_last.pth"),
            epoch,
            model,
            optimizer,
            lr_scheduler,
        )
        if epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
            save_model(
                os.path.join(
                    cfg.OUTPUT_DIR, cfg.EXP_ID, "model_epoch_{}.pth".format(epoch)
                ),
                epoch,
                model,
                optimizer,
                lr_scheduler,
            )
            log_dict_val = {}  # create an empty log_dict_val for non-validation epochs
        logger.write("\n")

    if cfg.TRAIN.VAL_INTERVALS > 0:
        print("Best ACC@1: ", best["ACC@1"])
        print("Best ACC@5: ", best["ACC@5"])
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args.cfg)
    local_rank = args.local_rank
    main(cfg, local_rank)
