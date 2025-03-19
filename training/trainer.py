import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import WandbLogger

from utils.tokenizer import gpt2_tokenizer
from data.tinystoriesInstruct import TinyStoriesDataModule, ChunkedDataset
from models.gpt2 import GPT2
from utils.config import config

L.seed_everything(0, workers=True)

import os 
os.makedirs("/scratch/anish.joishy/", exist_ok=True)


def train(model, data_module):
    wandb_logger = WandbLogger(name="rename", project="sequence-compression-2")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/scratch/anish.joishy/",
        filename=config.model.name + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    print("initializing trainer")
    trainer = L.Trainer(
        logger=wandb_logger,
        devices=4,
        accelerator="auto",
        max_epochs=config.training.epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        val_check_interval=5000,
        accumulate_grad_batches=16,
    )

    # print("Finding Learning Rate")
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, data_module, attr_name="learning_rate")

    # new_lr = lr_finder.suggestion()
    # model.hparams.learning_rate = new_lr

    print("Traing Started")
    trainer.fit(model, data_module)
    return trainer
