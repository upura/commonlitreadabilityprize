import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        df,
        text_col: str,
        target_col: str,
        tokenizer_name: str,
        max_len: int,
        is_train: bool = True,
    ):
        super().__init__()

        self.df = df
        self.is_train = is_train
        self.text = df[text_col].tolist()

        if self.is_train:
            self.target = torch.tensor(df[target_col].values, dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding="max_length",
            max_length=max_len,
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


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.test_df = None
        self.train_df = None
        self.valid_df = None
        self.cfg = cfg

    def get_test_df(self):
        df = pd.read_csv(self.cfg.TEST_DF_PATH)
        return df

    def split_train_valid_df(self):
        if int(self.cfg.debug):
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH, nrows=100)
        else:
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH)

        # Remove incomplete entries if any.
        df.drop(
            df[(df.target == 0) & (df.standard_error == 0)].index,
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        cv = KFold(n_splits=self.cfg.NUM_FOLDS, shuffle=True, random_state=42)
        for n, (train_index, val_index) in enumerate(cv.split(df)):
            df.loc[val_index, "fold"] = int(n)
        df["fold"] = df["fold"].astype(int)

        train_df = df[df["fold"] != self.cfg.fold].reset_index(drop=True)
        valid_df = df[df["fold"] == self.cfg.fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        self.test_df = self.get_test_df()
        train_df, valid_df = self.split_train_valid_df()
        self.train_df = train_df
        self.valid_df = valid_df

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_ds(self, phase):
        assert phase in {"train", "valid", "test"}
        ds = TextDataset(
            df=self.get_dataframe(phase=phase),
            text_col=self.cfg.TEXT_COL,
            target_col=self.cfg.TARGET_COL,
            tokenizer_name=self.cfg.TOKENIZER_PATH,
            max_len=self.cfg.MAX_LEN,
            is_train=(phase != "test"),
        )
        return ds

    def get_loader(self, phase):
        dataset = self.get_ds(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True if phase == "train" else False,
            num_workers=4,
            drop_last=True if phase == "train" else False,
        )

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")
