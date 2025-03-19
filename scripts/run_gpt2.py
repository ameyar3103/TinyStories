from transformers import AutoTokenizer
import torch

from models.gpt2 import GPT2
# from data.tinystoriesInstruct import TinyStoriesDataModule, ChunkedDataset
from data.tinystories import TinyStoriesDataModule, ChunkedDataset

from training.trainer import train
from utils.tokenizer import trained_tokenizer, gpt2_tokenizer, gpt_neo_tokenizer
from utils.config import config

torch.set_float32_matmul_precision("medium")


def main():
    if config.tokenizer.name == "gpt2":
        tokenizer = gpt2_tokenizer()
    elif config.tokenizer.name == "gpt_neo":
        tokenizer = gpt_neo_tokenizer()
    else:
        tokenizer = trained_tokenizer(config.tokenizer.cache_dir)

    model = GPT2(tokenizer)
    data_module = TinyStoriesDataModule(tokenizer)

    trainer = train(model, data_module)


if __name__ == "__main__":
    main()
