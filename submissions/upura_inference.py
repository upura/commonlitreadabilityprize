import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

gc.enable()
NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 248
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1.0, 1)]
ROBERTA_PATH = "../input/clrp-roberta-base/clrp_roberta_base"
TOKENIZER_PATH = "../input/clrp-roberta-base/clrp_roberta_base"
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

    test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv(
        "../input/commonlitreadabilityprize/sample_submission.csv"
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    gc.collect()

    SEED = 1000

    test_dataset = LitDataset(test_df, inference_only=True)
    all_predictions = np.zeros((NUM_FOLDS, len(test_df)))

    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    for index in range(NUM_FOLDS):
        model_path = f"../input/commonlit-exp000/model_{index + 1}.pth"
        print(f"\nUsing {model_path}")

        model = LitModel()
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)

        all_predictions[index] = predict(model, test_loader)

        del model
        gc.collect()

    upura_pred = all_predictions.mean(axis=0)
    np.save("upura_pred", upura_pred)
