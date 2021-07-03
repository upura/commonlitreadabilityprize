import os
import gc
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold


gc.enable()
NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 248
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
ROBERTA_PATH = 'roberta-large'
TOKENIZER_PATH = 'roberta-large'
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
        #self.text = [text.replace("\n", " ") for text in self.text]

        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])

        if self.inference_only:
            return (input_ids, attention_mask)
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states":True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)  
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(1024, 1)              
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

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


def train(model, model_path, train_loader, val_loader,
          optimizer, scheduler=None, num_epochs=NUM_EPOCHS):
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

                print(f"Epoch: {epoch} batch_num: {batch_num}",
                      f"val_rmse: {val_rmse:0.4}")

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
                    print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                          f"(from epoch {best_epoch})")

                start = time.time()
            step += 1
    return best_val_rmse


def create_optimizer(model):
    named_parameters = list(model.named_parameters())

    roberta_parameters = named_parameters[:197]
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]

    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 2e-5

        if layer_num >= 69:
            lr = 5e-5

        if layer_num >= 133:
            lr = 1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)


if __name__ == '__main__':

    train_df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    # Remove incomplete entries if any.
    train_df.drop(train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
                  inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
    submission_df = pd.read_csv("../input/commonlitreadabilityprize/sample_submission.csv")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    gc.collect()

    SEED = 1000
    list_val_rmse = []

    kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):    
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
        model_path = f"large_model_{fold + 1}.pth"

        set_random_seed(SEED + fold)

        train_dataset = LitDataset(train_df.loc[train_indices])    
        val_dataset = LitDataset(train_df.loc[val_indices])    

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                drop_last=True, shuffle=True, num_workers=2)    
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                drop_last=False, shuffle=False, num_workers=2)    

        set_random_seed(SEED + fold)
        model = LitModel().to(DEVICE)

        optimizer = create_optimizer(model)                        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=50)

        list_val_rmse.append(train(model, model_path, train_loader,
                                val_loader, optimizer, scheduler=scheduler))

        del model
        gc.collect()

        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())

    test_dataset = LitDataset(test_df, inference_only=True)
    all_predictions = np.zeros((len(list_val_rmse), len(test_df)))

    test_dataset = LitDataset(test_df, inference_only=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            drop_last=False, shuffle=False, num_workers=2)

    for index in range(len(list_val_rmse)):            
        model_path = f"large_model_{index + 1}.pth"
        print(f"\nUsing {model_path}")
                            
        model = LitModel()
        model.load_state_dict(torch.load(model_path))    
        model.to(DEVICE)
        
        all_predictions[index] = predict(model, test_loader)
        
        del model
        gc.collect()

    predictions = all_predictions.mean(axis=0)
    submission_df.target = predictions
    print(submission_df)
    submission_df.to_csv("submission.csv", index=False)
