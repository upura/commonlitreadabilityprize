import sys

sys.path.append("../input/d/takuok/aipipeline/AIPipeline/")
sys.path.append(
    "../input/d/takuok/packages/pytorch-image-models-master/pytorch-image-models-master/"
)
sys.path.append(
    "../input/d/takuok/packages/pretrained-models.pytorch-master/pretrained-models.pytorch-master/"
)
sys.path.append("../input/d/takuok/packages/easydict-master/easydict-master/")
sys.path.append("../input/d/takuok/packages/omegaconf-2.0.6/omegaconf-2.0.6/")
sys.path.append(
    "../input/d/takuok/packages/segmentation_models.pytorch-master/segmentation_models.pytorch-master/"
)
sys.path.append(
    "../input/d/takuok/packages/EfficientNet-PyTorch-master/EfficientNet-PyTorch-master/"
)
sys.path.append(
    "../input/d/takuok/packages/pytorch-metric-learning-master/pytorch-metric-learning-master/src/"
)
sys.path.append(
    "../input/d/takuok/packages/japanize-matplotlib-master/japanize-matplotlib-master/"
)
sys.path.append("../input/d/takuok/packages/einops-master/einops-master/")
sys.path.append("../input/d/takuok/packages/lightly-master/lightly-master/")
sys.path.append("../input/d/takuok/packages/lightly_utils-0.0.2/lightly_utils-0.0.2/")
sys.path.append("../input/d/takuok/packages/hydra-1.0.6/hydra-1.0.6/")
import gc
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from hf import api, factory
from hf.preprocesses import get_target_names, get_target_tasks, preprocess_config
from hf.utils import get_models, load_df, setup_logger, timer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def load_config_with_omega(path, use_edict: bool = True, is_recursion: bool = False):
    assert ".yml" in str(path) or ".json" in str(
        path
    ), '"path" should be ".yml" or ".json"'
    config = OmegaConf.load(path)
    if "_base_" in config:
        config._base_ = [
            b.replace(
                "/home/user/Desktop/RistAIPipeline",
                "../input/d/takuok/aipipeline/AIPipeline",
            )
            .replace("config/", "../input/d/takuok/commonlit-config/")
            .replace(
                "/home/ubuntu/Desktop/RistAIPipeline",
                "../input/d/takuok/aipipeline/AIPipeline",
            )
            for b in config._base_
        ]
        for i, p in enumerate(config._base_):
            if i == 0:
                config_ = load_config_with_omega(p, is_recursion=True)
            else:
                config_ = OmegaConf.merge(
                    config_, load_config_with_omega(p, is_recursion=True)
                )
        config = OmegaConf.merge(config_, config)

    if not is_recursion:
        config = yaml.load(OmegaConf.to_yaml(config, resolve=True))  # pythonのdict形式に直す
        if use_edict:
            config = edict(config)

    return config


def get_config(path):
    config = load_config_with_omega(path)
    config.data.dir = "../input/commonlitreadabilityprize/"
    config.data.out_dir = "./"
    config.model.params.pretrained = None
    # config.dataloader.test.batch_size = 4
    config.dataloader.test.num_workers = os.cpu_count()
    config.model.device = "cuda"

    return config


# ===============
# Functions
# ===============
def main(config, models):
    with timer("split data"):
        target_names = get_target_names(config.target)
        target_tasks = get_target_tasks(config.target)
        test_datasets = factory.get_test_datasets(
            df=test,
            ids=ids,
            target_names=target_names,
            target_tasks=target_tasks,
            name=config.datasets.name,
            params=config.datasets.test_params,
        )
        test_loader = DataLoader(test_datasets, **config.dataloader.test)

    with timer("train"):
        runner = api.TextPredictor(
            target_infos=config.target, target_names=target_names
        )
        pred = runner.predict(test_loader, models, post_process=False)
        del test_loader, runner
        gc.collect()

    return pred


if __name__ == "__main__":
    # ===============
    # Constants
    # ===============
    config_path = "../input/d/takuok/commonlit-config/exp105_roberta_large_itpt.yml"
    config = get_config(config_path)
    config = preprocess_config(config)

    DATA_DIR = Path(config.data.dir)
    TEST_PATH = DATA_DIR / config.data.test_file
    # ===============
    # Settings
    # ===============
    LOGGER_PATH = "log.txt"
    setup_logger(out_file=LOGGER_PATH)

    test = load_df(TEST_PATH, dtype={config.cols.id_col: str})
    test["text"] = test["excerpt"]
    ids = test[config.cols.id_col].values

    model_files = [
        "../input/commonrtx2/exp105_roberta_large_itpt_fold0.pth",
        "../input/commonrtx2/exp105_roberta_large_itpt_fold1.pth",
        "../input/commonrtx2/exp105_roberta_large_itpt_fold2.pth",
        "../input/commonrtx2/exp105_roberta_large_itpt_fold3.pth",
        "../input/commonrtx2/exp105_roberta_large_itpt_fold4.pth",
    ]
    config_path = "../input/d/takuok/commonlit-config/exp105_roberta_large_itpt.yml"
    config = get_config(config_path)
    config = preprocess_config(config)
    config.model.params.num_classes = [c.num_classes for c in config.target]
    config.model.params.backbone = "../input/robertalarge/"
    config.datasets.test_params.transform = [
        {"name": "CleanText"},
        {"name": "Tokenize", "params": {"model_name": config.model.params.backbone}},
    ]

    models = get_models(model_files, config)
    pred = main(config, models)
    del models
    gc.collect()
    torch.cuda.empty_cache()

    model_files = [
        "../input/commonlitrtx/exp108_roberta_large_ep10_fold0.pth",
        "../input/commonlitrtx/exp108_roberta_large_ep10_fold1.pth",
        "../input/commonlitrtx/exp108_roberta_large_ep10_fold2.pth",
        "../input/commonlitrtx/exp108_roberta_large_ep10_fold3.pth",
        "../input/commonlitrtx/exp108_roberta_large_ep10_fold4.pth",
    ]
    config_path = "../input/d/takuok/commonlit-config/exp108_roberta_large_ep10.yml"
    config = get_config(config_path)
    config = preprocess_config(config)
    config.model.params.num_classes = [c.num_classes for c in config.target]
    config.model.params.backbone = "../input/robertalarge/"
    config.datasets.test_params.transform = [
        {"name": "CleanText"},
        {"name": "Tokenize", "params": {"model_name": config.model.params.backbone}},
    ]

    models = get_models(model_files, config)
    pred2 = main(config, models)
    del models
    gc.collect()
    torch.cuda.empty_cache()

    model_files = [
        "../input/commonrtx2/exp096_longformer_large_fold0.pth",
        "../input/commonrtx2/exp096_longformer_large_fold1.pth",
        "../input/commonrtx2/exp096_longformer_large_fold2.pth",
        "../input/commonrtx2/exp096_longformer_large_fold3.pth",
        "../input/commonrtx2/exp096_longformer_large_fold4.pth",
    ]
    config_path = "../input/d/takuok/commonlit-config/exp096_longformer_large.yml"
    config = get_config(config_path)
    config = preprocess_config(config)
    config.model.params.num_classes = [c.num_classes for c in config.target]
    config.model.params.backbone = "../input/longformerlarge/"
    config.datasets.test_params.transform = [
        {"name": "CleanText"},
        {"name": "Tokenize", "params": {"model_name": config.model.params.backbone}},
    ]

    models = get_models(model_files, config)
    pred3 = main(config, models)
    del models
    gc.collect()
    torch.cuda.empty_cache()

    model_files = [
        "../input/commonlitrtx/exp085_electra_large_discriminator_fold0.pth",
        "../input/commonlitrtx/exp085_electra_large_discriminator_fold1.pth",
        "../input/commonlitrtx/exp085_electra_large_discriminator_fold2.pth",
        "../input/commonlitrtx/exp085_electra_large_discriminator_fold3.pth",
        "../input/commonlitrtx/exp085_electra_large_discriminator_fold4.pth",
    ]
    config_path = (
        "../input/d/takuok/commonlit-config/exp085_electra_large_discriminator.yml"
    )
    config = get_config(config_path)
    config = preprocess_config(config)
    config.model.params.num_classes = [c.num_classes for c in config.target]
    config.model.params.backbone = "../input/electra/large-discriminator/"
    config.datasets.test_params.transform = [
        {"name": "CleanText"},
        {"name": "Tokenize", "params": {"model_name": config.model.params.backbone}},
    ]

    models = get_models(model_files, config)
    pred4 = main(config, models)
    del models
    gc.collect()
    torch.cuda.empty_cache()

    np.save("takuoko_exp085", pred4["target"].reshape(-1))
    np.save("takuoko_exp096", pred3["target"].reshape(-1))
    np.save("takuoko_exp105", pred["target"].reshape(-1))
    np.save("takuoko_exp108", pred2["target"].reshape(-1))
