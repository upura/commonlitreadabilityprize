import dataclasses


@dataclasses.dataclass
class Cfg:
    PROJECT_NAME = "commonlit"
    RUN_NAME = "exp004"
    NUM_FOLDS = 5
    NUM_CLASS = 1
    NUM_EPOCHS = 3
    NUM_GPUS = 1
    MAX_LEN = 248
    BATCH_SIZE = 16
    MODEL_PATH = "../input/clrp_roberta_base"
    TOKENIZER_PATH = "../input/clrp_roberta_base"
    OUTPUT_PATH = "."
    TRAIN_DF_PATH = "../input/commonlitreadabilityprize/train.csv"
    TEST_DF_PATH = "../input/commonlitreadabilityprize/test.csv"
    TEXT_COL = "excerpt"
    TARGET_COL = "target"
