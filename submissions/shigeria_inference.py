import gc
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ARDRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.append("../input/sentence-transformers/sentence-transformers-master")
from sentence_transformers import SentenceTransformer, models

gc.enable()
warnings.filterwarnings("ignore")
model_path = "../input/clrp-roberta-base/clrp_roberta_base"
word_embedding_model = models.Transformer(model_path, max_seq_length=275)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
BATCH_SIZE = 32
MAX_LEN = 248
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1.0, 1)]
ROBERTA_PATH = "/kaggle/input/roberta-base"
TOKENIZER_PATH = "/kaggle/input/roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_MODELS = 5
NUM_EPOCHS2 = 4
ROBERTA_PATH2 = "../input/robertalarge"
TOKENIZER_PATH2 = "../input/robertalarge"


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


class LitModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = AutoConfig.from_pretrained(ROBERTA_PATH2)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )
        self.config.update({"num_labels": 1})

        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH2, config=self.config)

        self.attention = nn.Sequential(
            nn.Linear(1024, 768), nn.Tanh(), nn.Linear(768, 1), nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(nn.Linear(1024, 1))
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


def tuningOfARD(X_train_, y):
    def objective(trial):
        # alpha をsuggest（提案）する範囲の指定
        alpha_1 = trial.suggest_loguniform("alpha_1", 1e-8, 10.0)
        alpha_2 = trial.suggest_loguniform("alpha_2", 1e-8, 10.0)
        lambda_1 = trial.suggest_loguniform("lambda_1", 1e-8, 10000.0)
        lambda_2 = trial.suggest_loguniform("lambda_2", 1e-8, 10000.0)
        tmp = []
        kf = KFold(n_splits=5, random_state=2021, shuffle=True)
        for train_index, test_index in kf.split(X_train_, y):
            reg = ARDRegression(
                alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2
            )
            reg.fit(X_train_[train_index], y[train_index])
            tmp.append(
                mean_squared_error(
                    reg.predict(X_train_[test_index]), y[test_index], squared=False
                )
            )
        return np.mean(np.array(tmp))

    return objective


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
    train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv(
        "/kaggle/input/commonlitreadabilityprize/sample_submission.csv"
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Remove incomplete entries if any.
    train_df.drop(
        train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
        inplace=True,
    )
    train_df.reset_index(drop=True, inplace=True)
    # dropping some columns
    train_df = train_df[["id", "excerpt", "target", "standard_error"]]
    # encoding train and test strings
    X_train = model.encode(train_df.excerpt, device="cuda")
    X_test = model.encode(test_df.excerpt, device="cuda")

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

    tokenizer2 = AutoTokenizer.from_pretrained(TOKENIZER_PATH2)
    all_predictions2 = np.zeros((NUM_MODELS, len(test_df)))
    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    for model_index in range(NUM_MODELS):
        model_path = f"../input/drop-roberta/model_{model_index + 1}.pth"
        print(f"\nUsing {model_path}")

        model = LitModel2()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        all_predictions2[model_index] = predict(model, test_loader)

        del model
        gc.collect()

    model2_predictions = all_predictions2.mean(axis=0)

    all_predictions3 = np.zeros((NUM_MODELS, len(test_df)))
    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    for model_index in range(NUM_MODELS):
        model_path = f"../input/my-itpt-commonlit/model_{model_index + 1}.pth"
        print(f"\nUsing {model_path}")

        model = LitModel2()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        all_predictions3[model_index] = predict(model, test_loader)

        del model
        gc.collect()
    model3_predictions = all_predictions3.mean(axis=0)

    svd1 = TruncatedSVD(n_components=3, n_iter=10, random_state=42)
    svd1.fit(X_train)
    X_train_svd = svd1.transform(X_train)
    X_test_svd = svd1.transform(X_test)

    np.save("shigeria_pred1", model1_predictions.reshape(-1, 1))
    np.save("shigeria_pred2", model2_predictions.reshape(-1, 1))
    np.save("shigeria_pred3", model3_predictions.reshape(-1, 1))
    np.save("X_train_svd", X_train_svd)
    np.save("X_test_svd", X_test_svd)
