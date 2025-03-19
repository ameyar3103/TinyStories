from transformers import AutoTokenizer


def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", clean_up_tokenization_spaces=False
    )

    tokenizer.model_max_length = 1_000_000  # disable warnings

    tokenizer.add_special_tokens({"cls_token": "<|cls|>"})
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return tokenizer

def gpt_neo_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.model_max_length = 1_000_000  # disable warnings

    tokenizer.add_special_tokens({"cls_token": "<|cls|>"})
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return tokenizer

def main():
    tokenizer = gpt_neo_tokenizer()
    print(tokenizer)


if __name__ == "__main__":
    main()
