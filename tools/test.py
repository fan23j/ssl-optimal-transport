from __future__ import absolute_import, division, print_function

import argparse
import os

import torch
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
        print("load pretrained model from {}".format(cfg.TEST.MODEL_PATH))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_PATH), strict=False)

    # set up trainer code from train_factory
    Trainer = train_factory[cfg.TEST.TASK]
    trainer = Trainer(cfg, model, optimizer=optimizer, lr_scheduler=None)

    trainer.set_device(device)

    # load in dataset
    print("Setting up data...")
    _, val_dataset = get_dataset(cfg)

    # Create validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print("Starting testing...")

    with torch.no_grad():
        results = trainer.test(val_loader)

    out_path = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID, cfg.TEST.OUTPUT_FILE)

    # Write the predictions and actual labels to a file
    # with open(out_path, "w") as f:
    #     for label, prediction in results:
    #         f.write(f"{label}, {prediction}\n")

    # print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args.cfg)
    local_rank = args.local_rank
    main(cfg, local_rank)
