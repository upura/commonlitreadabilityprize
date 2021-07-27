import gc
import math
import multiprocessing
import os
import random
import time

import more_itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer

gc.enable()
NUM_FOLDS = 5
NUM_EPOCHS = 5
BATCH_SIZE = 8
MAX_LEN = 248
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1.0, 1)]
ROBERTA_PATH = "../input/electra/large-discriminator"
TOKENIZER_PATH = "../input/electra/large-discriminator"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


class LitDataset(Dataset):
    def __init__(self, df, inference_only=False):
        super().__init__()

        self.df = df
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()
        # self.text = [text.replace("\n", " ") for text in self.text]

        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.encoded["input_ids"][index])
        attention_mask = torch.tensor(self.encoded["attention_mask"][index])

        if self.inference_only:
            return (input_ids, attention_mask)
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = AutoConfig.from_pretrained(ROBERTA_PATH)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )
        self.config.update({"num_labels": 1})

        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=self.config)

        self.linear = nn.Linear(6144, 1)
        self.attention = nn.Sequential(
            nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 1), nn.Softmax(dim=1)
        )

        reinit_layers = 5

        if reinit_layers > 0:
            print(f"Reinitializing Last {reinit_layers} Layers ...")
            encoder_temp = self.roberta
            for layer in encoder_temp.encoder.layer[-reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(
                            mean=0.0, std=self.config.initializer_range
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(
                            mean=0.0, std=self.config.initializer_range
                        )
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
            print("Done.!")

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output_state = torch.stack(roberta_output.hidden_states)
        cat_over_last_layers = torch.cat(
            (output_state[-1], output_state[-2], output_state[-3]), -1
        )
        cls_pooling = cat_over_last_layers[:, 0]
        weights = self.attention(cat_over_last_layers)
        context_vectors = torch.sum(weights * cat_over_last_layers, dim=1)
        return self.linear(torch.cat([context_vectors, cls_pooling], -1))


def eval_mse(model, data_loader):
    """Evaluates the mean squared error of the |model| on |data_loader|"""
    model.eval()
    mse_sum = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input_ids, attention_mask)

            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()

    return mse_sum / len(data_loader.dataset)


def predict(model, data_loader):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))
    index = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            pred = model(input_ids, attention_mask)

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result


def train(
    model,
    model_path,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    num_epochs=NUM_EPOCHS,
):
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]

    start = time.time()

    for epoch in range(num_epochs):
        val_rmse = None

        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            model.train()

            pred = model(input_ids, attention_mask)

            mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)

            mse.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if step >= last_eval_step + eval_period:
                # Evaluate the model on val_loader.
                elapsed_seconds = time.time() - start
                num_steps = step - last_eval_step
                print(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step

                val_rmse = math.sqrt(eval_mse(model, val_loader))

                print(
                    f"Epoch: {epoch} batch_num: {batch_num}",
                    f"val_rmse: {val_rmse:0.4}",
                )

                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break

                if not best_val_rmse or val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    print(f"New best_val_rmse: {best_val_rmse:0.4}")
                else:
                    print(
                        f"Still best_val_rmse: {best_val_rmse:0.4}",
                        f"(from epoch {best_epoch})",
                    )

                start = time.time()

            step += 1

    return best_val_rmse


def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(23, -1, -1):
        layer_params = {
            "params": [
                p for n, p in model.named_parameters() if f"encoder.layer.{layer}." in n
            ],
            "lr": lr,
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        "params": [
            p
            for n, p in model.named_parameters()
            if "linear" in n
            or "pooling" in n
            or "attention.0" in n
            or "attention.2" in n
        ],
        "lr": classifier_lr,
    }
    parameters.append(classifier_params)
    return AdamW(parameters)


class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(SmartBatchingDataset, self).__init__()
        self._data = (
            (f"{tokenizer.bos_token} " + df.excerpt + f" {tokenizer.eos_token}")
            .apply(tokenizer.tokenize)
            .apply(tokenizer.convert_tokens_to_ids)
            .to_list()
        )
        self._targets = None
        if "target" in df.columns:
            self._targets = df.target.tolist()
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_len, pad_id):
        self.sampler = SmartBatchingSampler(
            data_source=self._data, batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            targets=self._targets, max_length=max_len, pad_token_id=pad_id
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=(multiprocessing.cpu_count() - 1),
            pin_memory=True,
        )
        return dataloader


class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds


class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id,
        )

        if self._targets is not None:
            output = input_ids, attention_mask, torch.tensor(targets)
        else:
            output = input_ids, attention_mask
        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks


class ContinuousStratifiedKFold(StratifiedKFold):
    def split(self, x, y, groups=None):
        num_bins = int(np.floor(1 + np.log2(len(y))))
        bins = pd.cut(y, bins=num_bins, labels=False)
        return super().split(x, bins, groups)


if __name__ == "__main__":
    train_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/train.csv")

    # Remove incomplete entries if any.
    train_df.drop(
        train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
        inplace=True,
    )
    train_df.reset_index(drop=True, inplace=True)

    test_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv(
        "/kaggle/input/commonlitreadabilityprize/sample_submission.csv"
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    test_dataset = LitDataset(test_df, inference_only=True)
    all_predictions = np.zeros((5, len(test_df)))

    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    for index in range(5):
        model_path = (
            f"../input/commonlit-concat-attention-pooling-electra/model_{index + 1}.pth"
        )
        print(f"\nUsing {model_path}")

        model = LitModel()
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)

        all_predictions[index] = predict(model, test_loader)

        del model
        gc.collect()
    predictions = all_predictions.mean(axis=0)
    np.save("shigeria_pred9", predictions.reshape(-1, 1))
