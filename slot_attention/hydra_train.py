from typing import Optional

import os
from time import localtime, strftime
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

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

        clevr_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(rescale),  # rescale between -1 and 1
                transforms.Resize(cfg.resolution),
            ]
        )

        clevr_datamodule = CLEVRDataModule(
            data_root=cfg.data_root,
            max_n_objects=cfg.num_slots - 1,
            train_batch_size=cfg.batch_size,
            val_batch_size=cfg.val_batch_size,
            clevr_transforms=clevr_transforms,
            num_train_images=cfg.num_train_images,
            num_val_images=cfg.num_val_images,
            num_workers=cfg.num_workers,
        )

        print(f"Training set size (images must have {cfg.num_slots - 1} objects):", len(clevr_datamodule.train_dataset))

        model = SlotAttentionModel(
            resolution=cfg.resolution,
            num_slots=cfg.num_slots,
            num_iterations=cfg.num_iterations,
            empty_cache=cfg.empty_cache,
        )

        method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, params=cfg)

        logger_name = "slot-attention-clevr6"
        logger = pl_loggers.TensorBoardLogger("./logs/"+logger_name+strftime("-%Y%m%d%H%M%S", localtime()))

        trainer = Trainer(
            logger=logger if cfg.is_logger_enabled else False,
            accelerator="ddp" if cfg.gpus > 1 else None,
            num_sanity_val_steps=cfg.num_sanity_val_steps,
            gpus=cfg.gpus,
            max_epochs=cfg.max_epochs,
            log_every_n_steps=50,
            callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if cfg.is_logger_enabled else [],
        )

    def run_training(self):
        trainer.fit(method)

@hydra.main(config_path='hydra_cfg/experiment.yaml')
def main(cfg):
    logging.info(cfg.pretty())

    logging.info("Base directory: %s", os.getcwd())

    workplace = _Workplace(cfg)

    workplace.run_training()


if __name__ == "__main__":
    main()