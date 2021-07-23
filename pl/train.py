from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from configs.run000 import Cfg
from dataset import MyDataModule
from lightning_module import MyLightningModule

if __name__ == "__main__":

    fold = 0
    cfg = Cfg()
    seed_everything(777)
    # logger は csv logger と wandb logger 両方使ってみる
    logger = CSVLogger(save_dir=str(cfg.OUTPUT_PATH), name=f"fold_{fold}")
    wandb_logger = WandbLogger(name=f"{cfg.RUN_NAME}_{fold}", project=cfg.PROJECT_NAME)

    # 学習済重みを保存するために必要
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.OUTPUT_PATH),
        filename=f"{cfg.RUN_NAME}_fold_{fold}",
        save_weights_only=True,
        save_top_k=None,
        monitor=None,
    )
    trainer = Trainer(
        max_epochs=cfg.NUM_EPOCHS,
        gpus=cfg.NUM_GPUS,
        callbacks=[checkpoint_callback],
        logger=[logger, wandb_logger],
    )

    # Lightning module and start training
    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    trainer.fit(model, datamodule=datamodule)
