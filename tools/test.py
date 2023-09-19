import argparse
import torch

import _init_paths
from logger import Logger
from config import cfg, update_config
from datasets.dataset_factory import get_dataset
from model.model import create_model, load_model
from train.train_factory import train_factory
from train.scheduler_factory import OptimizerSchedulerFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    return args


def main(cfg, local_rank):
    print("Creating model...")
    model = create_model(cfg.MODEL.NAME, cfg)
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

    # set up optimizer and scheduler
    scheduler_factory = OptimizerSchedulerFactory(cfg, model, val_loader)
    optimizer, lr_scheduler = scheduler_factory.create()

    # load in pretrained weights
    print("Creating model...")
    model = create_model(cfg.MODEL.NAME, cfg)
    if cfg.MODEL.PRETRAINED != "":
        print("load pretrained model from {}".format(cfg.MODEL.PRETRAINED))
        model, optimizer, lr_scheduler, start_epoch = load_model(
            cfg,
            model,
            optimizer,
            lr_scheduler,
        )
    model.to(device)

    # set up trainer code from train_factory
    Trainer = train_factory[cfg.TASK]
    trainer = Trainer(cfg, model, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dataset=train_dataset, val_dataset=val_dataset)

    trainer.set_device(device)

    print("Starting testing...")
    with torch.no_grad():
        log_dict_val = trainer.val(0, val_loader)

#     for k, v in log_dict_val.items():
#         if k == "imgs":
#             logger.write_image(k, v, 0)
#         elif k == "figure":
#             logger.add_figure(k, v.gcf(), 0)
#         else:
#             logger.write("{} {:8f} | ".format(k, v))

#     logger.write("\n")
#     logger.close()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args.cfg)
    local_rank = args.local_rank
    exps = [[1.0, False],[0.9,False,],[0.5,False],[0.3,False],[0.9,True],[0.5,True],[0.3,True]]
    main(cfg, local_rank)
    # for exp in exps:
    #     cfg.defrost() 
    #     cfg.DATASET.LT_IMBALANCE_RATIO = exp[0]
    #     cfg.DATASET.LT_REVESER = exp[1]
    #     cfg.freeze()
    #     main(cfg, local_rank)
    #     print(exp)
