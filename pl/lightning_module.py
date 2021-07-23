from collections import OrderedDict

import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error

from loss import RMSELoss
from models.roberta import MyModel


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = MyModel(cfg.MODEL_PATH)
        self.criterion = RMSELoss()

    def forward(self, ids, attention_mask):
        output = self.backbone(ids, attention_mask=attention_mask)
        return output

    def training_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask)
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask)
        loss = self.criterion(output, targets)
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": output.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

        score = mean_squared_error(targets, preds, squared=False)
        d["v_score"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
