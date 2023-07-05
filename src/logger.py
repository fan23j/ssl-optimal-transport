from __future__ import absolute_import, division, print_function
import tensorboardX

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import sys
import time

import torch

USE_TENSORBOARD = True
print("Using tensorboardX")


class Logger(object):
    def __init__(self, cfg):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID)):
            try:
                os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID))
            except:
                pass
        time_str = time.strftime("%Y-%m-%d-%H-%M")

        file_name = os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID), "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> torch version: {}\n".format(torch.__version__))
            opt_file.write(
                "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
            )
            opt_file.write("==> Cmd:\n")
            opt_file.write(str(sys.argv))
            opt_file.write("\n==> Opt:\n")

        log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID) + "/logs_{}".format(time_str)
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            try:
                os.makedirs(os.path.dirname(log_dir))
            except:
                pass
            try:
                os.makedirs(log_dir)
            except:
                pass
        self.log = open(log_dir + "/log.txt", "w")
        try:
            os.system(
                "cp {}/opt.txt {}/".format(
                    os.path.join(cfg.OUTPUT_DIR, cfg.EXP_ID), log_dir
                )
            )
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime("%Y-%m-%d-%H-%M")
            self.log.write("{}: {}".format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if "\n" in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)

    def write_image(self, k, v, epoch):
        if USE_TENSORBOARD:
            v = v.unsqueeze(0)
            self.writer.add_images(k, v, global_step=epoch)

    def add_figure(self, k, v, epoch):
        if USE_TENSORBOARD:
            self.writer.add_figure(k, v, global_step=epoch)
