from typing import Optional

import hydra
import logging
import os
from csv import reader
from time import localtime, strftime
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torchvision import transforms
import pandas as pd

from slot_attention.data import CLEVRDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
# from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale


# def main(params: Optional[SlotAttentionParams] = None):
class _Workplace(object):
    def __init__(self, cfg):

        assert cfg.num_slots > 1, "Must have at least 2 slots."

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

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
        # The following code is for loading a saved checkpoint
        # ckpt = torch.load("path_to_checkpoint")
        # state_dict = ckpt['state_dict']
        # for key in list(state_dict.keys()):
        #     state_dict[key.replace('model.', '')] = state_dict.pop(key)
        # model.load_state_dict(state_dict)

        self.method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, params=cfg)

        logger_name = "slot-attn/mix2gpu-lr-"+str(cfg.lr) + "-it-"+str(cfg.num_iterations)+ "-s-" + str(seed)#"-dup-"+str(cfg.dup_threshold)
        logger = pl_loggers.WandbLogger(project="objectness-test-clevr6", name=logger_name)
        # Use this line for Tensorboard logger
        # logger = pl_loggers.TensorBoardLogger("./logs/"+logger_name+strftime("-%Y%m%d%H%M%S", localtime()))

        self.trainer = Trainer(
            logger=logger if cfg.is_logger_enabled else False,
            accelerator="ddp" if cfg.gpus > 1 else None,
            num_sanity_val_steps=cfg.num_sanity_val_steps,
            gpus=cfg.gpus,
            max_epochs=cfg.max_epochs,
            check_val_every_n_epoch=cfg.eval_every_n_epoch,
            log_every_n_steps=50,
            callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if cfg.is_logger_enabled else [],
        )
        # to log the metric from the sanity check
        self.trainer.current_epoch = -1

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
