import warnings

import pandas as pd
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    train_data = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    test_data = pd.read_csv("../input/commonlitreadabilityprize/test.csv")
    ext_data = pd.read_csv(
        "../input/commonlit-external/dump_of_simple_english_wiki.csv"
    )

    data = pd.concat(
        [train_data[["excerpt"]], test_data[["excerpt"]], ext_data[["excerpt"]]]
    )
    data["excerpt"] = data["excerpt"].apply(lambda x: x.replace("\n", ""))

    text = "\n".join(data.excerpt.tolist())
    with open("text.txt", "w") as f:
        f.write(text)

    model_name = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("./clrp_roberta_base")

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="text.txt",  # mention train text file here
        block_size=256,
    )

    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="text.txt",  # mention valid text file here
        block_size=256,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./clrp_roberta_base_chk",  # select model path for checkpoint
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        save_total_limit=2,
        eval_steps=200,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    trainer.save_model("./clrp_roberta_base")
