from typing import Optional

import hydra
import logging
import os
from csv import reader
from time import localtime, strftime
from datetime import datetime
import torch
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torchvision import transforms
import pandas as pd

from data import CLEVRDataModule
from method import SlotAttentionMethod
from models.slot_attention import SlotAttentionModel
from utils import ImageLogCallback
from utils import rescale


# def main(params: Optional[SlotAttentionParams] = None):
class _Workplace(object):
    def __init__(self, cfg):

        assert cfg.num_slots > 1, "Must have at least 2 slots."

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

        dates = [1029,1030,1031,1101,1102,1103]

        if cfg.is_verbose:
            print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({cfg.num_slots - 1}) objects.")
            if cfg.num_train_images:
                print(f"INFO: restricting the train dataset size to `num_train_images`: {cfg.num_train_images}")
            if cfg.num_val_images:
                print(f"INFO: restricting the validation dataset size to `num_val_images`: {cfg.num_val_images}")

        # open the csv file to get the seed and the dataset mixing weights
        df = pd.read_csv(cfg.data_mix_csv)
        self.data_weights = df.iloc[cfg.data_mix_idx, 1:-1]
        seed = df.loc[cfg.data_mix_idx, "seed"]

        self.result_csv = cfg.result_csv
        self.data_mix_idx = cfg.data_mix_idx

        seed_everything(seed)

        clevr_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(240),
                transforms.Lambda(rescale),  # rescale between -1 and 1
                transforms.Resize(tuple(cfg.resolution)),
            ]
        )

        if os.path.exists(os.path.join(cfg.test_root, "obj_test", "CLEVR_test_cases.csv")):
            with open(os.path.join(cfg.test_root, "obj_test", "CLEVR_test_cases.csv"), "r") as f:
                csv_reader = reader(f)
                self.obj_algebra_test_cases = list(csv_reader)
        else:
            self.obj_algebra_test_cases = None
            print(os.path.join(cfg.test_root, "obj_test", "CLEVR_test_cases.csv")+" does not exist.")

        if os.path.exists(os.path.join(cfg.test_root, "attr_test", "CLEVR_test_cases.csv")):
            with open(os.path.join(cfg.test_root, "attr_test", "CLEVR_test_cases.csv"), "r") as f:
                csv_reader = reader(f)
                self.attr_algebra_test_cases = list(csv_reader)
        else:
            self.attr_algebra_test_cases = None
            print(os.path.join(cfg.test_root, "attr_test", "CLEVR_test_cases.csv")+" does not exist.")
        if os.path.exists(os.path.join(cfg.val_root, "CLEVR_val_list.csv")):
            with open(os.path.join(cfg.val_root, "CLEVR_val_list.csv"), "r") as f:
                csv_reader = reader(f)
                self.val_list = list(csv_reader)
        else:
            self.val_list = None
            print(os.path.join(cfg.val_root, "CLEVR_val_list.csv")+" does not exist.")

        clevr_datamodule = CLEVRDataModule(
            data_root=cfg.data_root,
            val_root=cfg.val_root,
            test_root=cfg.test_root,
            max_n_objects=cfg.num_slots - 1,
            train_batch_size=cfg.batch_size,
            val_batch_size=cfg.val_batch_size,
            test_batch_size=cfg.test_batch_size,
            clevr_transforms=clevr_transforms,
            num_train_images=cfg.num_train_images,
            num_val_images=cfg.num_val_images,
            num_test_images=cfg.num_test_images,
            num_workers=cfg.num_workers,
            val_list = self.val_list,
            data_weights = self.data_weights,
            obj_algebra_test_cases = self.obj_algebra_test_cases,
            attr_algebra_test_cases = self.attr_algebra_test_cases,
        )

        print(f"Training set size (images must have {cfg.num_slots - 1} objects):", len(clevr_datamodule.train_dataset))

        model = SlotAttentionModel(
            resolution=cfg.resolution,
            num_slots=cfg.num_slots,
            num_iterations=cfg.num_iterations,
            empty_cache=cfg.empty_cache,
        )

        checkpoint_path = "/checkpoint/siruixie/runs/objectness/hydra_train_dmix/data_mix_idx="+str(cfg.data_mix_idx)+",lr=0.0002,num_iterations=4,sweep_name=dmix/objectness-test-clevr6"
        max_birthtime = None
        last_dir_name = None
        for dir_name in os.listdir(checkpoint_path):
            dir_birthtime = os.stat(os.path.join(checkpoint_path, dir_name)).st_mtime
            dir_date = datetime.fromtimestamp(dir_birthtime).strftime('%m%d')
            print(dir_date)
            if int(dir_date) in dates:
                if max_birthtime is None:
                    max_birthtime = dir_birthtime
                    last_dir_name = dir_name
                elif max_birthtime>dir_birthtime:
                    continue
                else:
                    max_birthtime = dir_birthtime
                    last_dir_name = dir_name

        # The following code is for loading a saved checkpoint
        if not last_dir_name is None:
            last_dir = os.path.join(checkpoint_path, dir_name, "checkpoints")
            checkpoint_name = os.listdir(last_dir)[0]
            print("Loading checkpoint from "+last_dir_name, ", "+checkpoint_name+" exists.")
            ckpt = torch.load(os.path.join(last_dir, checkpoint_name))
            state_dict = ckpt['state_dict']
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '')] = state_dict.pop(key)
            model.load_state_dict(state_dict)
        else:
            print("No checkpoint exists for "+str(cfg.data_mix_idx))
            exit(0)

        self.method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, params=cfg)

        logger_name = "slot-attn/test"+str(cfg.data_mix_idx)#+str(cfg.lr) + "-it-"+str(cfg.num_iterations)+ "-s-" + str(seed)#"-dup-"+str(cfg.dup_threshold)
        logger = pl_loggers.WandbLogger(project="objectness-test-new", name=logger_name)
        # Use this line for Tensorboard logger
        # logger = pl_loggers.TensorBoardLogger("./logs/"+logger_name+strftime("-%Y%m%d%H%M%S", localtime()))

        self.trainer = Trainer(
            logger=logger if cfg.is_logger_enabled else False,
            accelerator="ddp" if cfg.gpus > 1 else None,
            num_sanity_val_steps=cfg.num_sanity_val_steps,
            gpus=cfg.gpus,
            max_epochs=0,#cfg.max_epochs,
            check_val_every_n_epoch=cfg.eval_every_n_epoch,
            log_every_n_steps=50,
            callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if cfg.is_logger_enabled else [],
        )

    def run_training(self):
        self.trainer.fit(self.method)
        # Here we get the metrics from the final epoch
        print("-----------------")
        print(self.trainer.logged_metrics)

@hydra.main(config_path='hydra_cfg', config_name='experiment')
def main(cfg):
    logging.info(cfg)

    logging.info("Base directory: %s", os.getcwd())

    workplace = _Workplace(cfg)

    workplace.run_training()


if __name__ == "__main__":
    main()