from transformers import AutoTokenizer
import torch

from models.compressor import Compressor
from data.tinystories import TinyStoriesDataModule, ChunkedDataset
from training.trainer import train
from utils.tokenizer import trained_tokenizer, gpt2_tokenizer, gpt_neo_tokenizer
from utils.config import config

torch.set_float32_matmul_precision("medium")

import os
os.makedirs('/scratch/anish.joishy2', exist_ok=True)

def main():
    if config.tokenizer.name == "gpt2":
        tokenizer = gpt2_tokenizer()
    elif config.tokenizer.name == "gpt_neo":
        tokenizer = gpt_neo_tokenizer()
    else:
        tokenizer = trained_tokenizer(config.tokenizer.cache_dir)
    print(tokenizer.pad_token_id, tokenizer.bos_token_id)
    
    model = Compressor(tokenizer=tokenizer)
    data_module = TinyStoriesDataModule(tokenizer)

    print("Training Model")
    trainer = train(model, data_module)


if __name__ == "__main__":

    main()
