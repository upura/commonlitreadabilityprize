import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)

gc.enable()
BATCH_SIZE = 32
MAX_LEN = 248
NUM_MODELS = 5
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1.0, 1)]
ROBERTA_PATH = "/kaggle/input/roberta-base"
TOKENIZER_PATH = "/kaggle/input/roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace("\n", "")
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    curr_sent = {}
    padding_length = max_len - len(tok["input_ids"])
    curr_sent["input_ids"] = tok["input_ids"] + ([0] * padding_length)
    curr_sent["token_type_ids"] = tok["token_type_ids"] + ([0] * padding_length)
    curr_sent["attention_mask"] = tok["attention_mask"] + ([0] * padding_length)
    return curr_sent


class DatasetRetriever(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt, label = self.excerpts[item], self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len, self.is_test
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "token_type_ids": torch.tensor(
                    features["token_type_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    features["attention_mask"], dtype=torch.long
                ),
                "label": torch.tensor(label, dtype=torch.double),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len, self.is_test
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "token_type_ids": torch.tensor(
                    features["token_type_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    features["attention_mask"], dtype=torch.long
                ),
            }


class CommonLitModel(nn.Module):
    def __init__(
        self, model_name, config, multisample_dropout=False, output_hidden_states=False
    ):
        super(CommonLitModel, self).__init__()
        self.config = config
        self.roberta = RobertaModel.from_pretrained(
            model_name, output_hidden_states=output_hidden_states
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))

        logits /= len(self.dropouts)

        # calculate loss
        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


def make_model(model_name, num_labels=1):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    config.update({"num_labels": num_labels})
    model = CommonLitModel(model_name, config=config)
    return model, tokenizer


def make_loader(
    data,
    tokenizer,
    max_len,
    batch_size,
):

    test_dataset = DatasetRetriever(data, tokenizer, max_len, is_test=True)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
    )

    return test_loader


class Evaluator:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def evaluate(self, data_loader, tokenizer):
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids = (
                    batch_data["input_ids"],
                    batch_data["attention_mask"],
                    batch_data["token_type_ids"],
                )
                input_ids, attention_mask, token_type_ids = (
                    input_ids.cuda(),
                    attention_mask.cuda(),
                    token_type_ids.cuda(),
                )

                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                logits = outputs[0].detach().cpu().numpy().squeeze().tolist()
                preds += logits
        return preds


def config(fold, model_name, load_model_path):
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)

    max_len = 250
    batch_size = 8

    model, tokenizer = make_model(model_name=model_name, num_labels=1)
    model.load_state_dict(torch.load(f"{load_model_path}/model{fold}.bin"))
    test_loader = make_loader(test, tokenizer, max_len=max_len, batch_size=batch_size)

    if torch.cuda.device_count() >= 1:
        print(
            "Model pushed to {} GPU(s), type {}.".format(
                torch.cuda.device_count(), torch.cuda.get_device_name(0)
            )
        )
        model = model.cuda()
    else:
        raise ValueError("CPU training is not supported")

    # scaler = torch.cuda.amp.GradScaler()
    scaler = None
    return (model, tokenizer, test_loader, scaler)


def run(fold=0, model_name=None, load_model_path=None):
    model, tokenizer, test_loader, scaler = config(fold, model_name, load_model_path)
    evaluator = Evaluator(model, scaler)

    test_time_list = []

    torch.cuda.synchronize()
    tic1 = time.time()

    preds = evaluator.evaluate(test_loader, tokenizer)

    torch.cuda.synchronize()
    tic2 = time.time()
    test_time_list.append(tic2 - tic1)

    del model, tokenizer, test_loader, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return preds


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
            return_attention_mask=True,
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

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )

        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)

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


if __name__ == "__main__":
    test_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv(
        "/kaggle/input/commonlitreadabilityprize/sample_submission.csv"
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    test_dataset = LitDataset(test_df, inference_only=True)

    all_predictions = np.zeros((NUM_MODELS, len(test_df)))
    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    for model_index in range(NUM_MODELS):
        model_path = f"../input/commonlit-roberta-0467/model_{model_index + 1}.pth"
        print(f"\nUsing {model_path}")

        model = LitModel()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        all_predictions[model_index] = predict(model, test_loader)

        del model
        gc.collect()
    model1_predictions = all_predictions.mean(axis=0)

    test = test_df
    pred_df1 = pd.DataFrame()
    pred_df2 = pd.DataFrame()
    pred_df3 = pd.DataFrame()
    for fold in tqdm(range(5)):
        pred_df1[f"fold{fold}"] = run(
            fold, "../input/roberta-base/", "../input/commonlit-roberta-base-i/"
        )
        pred_df2[f"fold{fold+5}"] = run(
            fold, "../input/robertalarge/", "../input/roberta-large-itptfit/"
        )
        pred_df3[f"fold{fold+10}"] = run(
            fold, "../input/robertalarge/", "../input/commonlit-roberta-large-ii/"
        )

    pred_df1 = np.array(pred_df1)
    pred_df2 = np.array(pred_df2)
    pred_df3 = np.array(pred_df3)
    model2_predictions = (
        (pred_df2.mean(axis=1) * 0.5)
        + (pred_df1.mean(axis=1) * 0.3)
        + (pred_df3.mean(axis=1) * 0.2)
    )

    np.save("andretugan_pred1", model1_predictions.reshape(-1))
    np.save("andretugan_pred2", model2_predictions.reshape(-1))
