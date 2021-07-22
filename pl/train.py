import gc

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

gc.enable()
NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 248
NUM_CLASS = 1
MODEL_PATH = "../input/clrp_roberta_base"
TOKENIZER_PATH = "../input/clrp_roberta_base"
TEXT_COL = "excerpt"
TARGET_COL = "target"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pl.seed_everything(777)


class TextDataset(Dataset):
    def __init__(self, df, is_train=True):
        super().__init__()

        self.df = df
        self.is_train = is_train
        self.text = df[TEXT_COL].tolist()

        if self.is_train:
            self.target = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            return_attention_mask=True,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.encoded["input_ids"][index])
        attention_mask = torch.tensor(self.encoded["attention_mask"][index])

        if self.is_train:
            target = self.target[index]
            return {
                "ids": input_ids,
                "attention_mask": attention_mask,
                "targets": target,
            }
        else:
            return {"ids": input_ids, "attention_mask": attention_mask}


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(MODEL_PATH)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )

        self.roberta = AutoModel.from_pretrained(MODEL_PATH, config=config)
        self.attention = nn.Sequential(
            nn.Linear(768, 512), nn.Tanh(), nn.Linear(512, 1), nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(nn.Linear(768, 1))

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat.flatten(), y))


class PLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = LitModel()
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
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask)
        loss = self.criterion(output, targets)
        self.log("val_loss", loss)
        return {"oof": output, "targets": targets, "loss": loss}

    def test_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        output = self.backbone(ids, attention_mask=attention_mask)
        return {"preds": output}

    def test_epoch_end(self, outputs):
        preds = np.concatenate(
            [x["preds"].detach().cpu().numpy() for x in outputs], axis=0
        )
        np.save("test_preds", preds)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":

    train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    # Remove incomplete entries if any.
    train_df.drop(
        train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
        inplace=True,
    )
    train_df.reset_index(drop=True, inplace=True)
    test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv(
        "../input/commonlitreadabilityprize/sample_submission.csv"
    )

    test_dataset = TextDataset(test_df, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    gc.collect()

    y_preds = []
    oof_train = np.zeros((len(train_df), NUM_CLASS))
    cv = KFold(n_splits=NUM_FOLDS, random_state=777, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(cv.split(train_df)):
        print(f"\nFold {fold}/{NUM_FOLDS}")
        model_path = f"model_{fold}.pth"

        train_dataset = TextDataset(train_df.loc[train_indices])
        val_dataset = TextDataset(train_df.loc[val_indices])

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=4,
        )

        model = PLModel()
        model = model.to(DEVICE)
        trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(test_dataloaders=test_loader)
