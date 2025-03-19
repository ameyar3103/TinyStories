from rouge_score import rouge_scorer
from utils.config import config
import torch
from models.gpt2 import GPT2
import os
import json
import matplotlib.pyplot as plt
from utils.tokenizer import gpt_neo_tokenizer, gpt2_tokenizer
from tqdm import tqdm

bins = [i * 0.01 for i in range(101)]

def calculate_rouge(reference: str, hypothesis: str, n: int, metric: str):
    if metric not in ['precision', 'recall', 'fmeasure']:
        raise ValueError("Metric must be 'precision', 'recall', or 'fmeasure'")

    rouge_type = f'rouge{n}'
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    scores = scorer.score(reference, hypothesis)
    rouge_score = scores[rouge_type]
    
    return getattr(rouge_score, metric)

def get_completions(original_stories, model):
    prompts = [orig_story.split()[:len(orig_story.split())//3] for orig_story in original_stories]
    completions = []
    tokenizer = model.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for prompt in tqdm(prompts):
        prompt = " ".join(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.model.generate(
                input_ids,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        completions.append(output_text)
    return completions

def make_hist(scores, hist_path):
    plt.hist(scores, bins=bins)
    plt.xlabel("Rouge2 Score")
    plt.ylabel("Frequency")
    plt.title(f"{hist_path.split('/')[-1]}")
    plt.savefig(f"{hist_path}.png")
    plt.close()


def rouge2_completion_orig(original_stories,genrated_stories, model_name):
    rouge2_scores = []
    for i in range(len(original_stories)):
        rouge2_scores.append(calculate_rouge(original_stories[i], genrated_stories[i], n=2, metric='precision'))
    save_path = f"data/comp_orig_rouge/{model_name}"
    os.makedirs(f"data/comp_orig_rouge", exist_ok=True)


    with open(f"{save_path}.json", "w") as f:
        json.dump(rouge2_scores, f, indent=4)

    make_hist(rouge2_scores, save_path)
    

    return rouge2_scores

def max_rouge2_self(stories, model_name):
    rouge2_scores = []
    for i in range(len(stories)):
        cur_rouge = []
        for j in range(len(stories)):
            if i != j:
                cur_rouge.append(calculate_rouge(stories[i], stories[j], n=2, metric='fmeasure'))
        rouge2_scores.append(max(cur_rouge))

    save_path = f"data/max_rouge2_self/{model_name}"
    os.makedirs(f"data/max_rouge2_self", exist_ok=True)
    with open(f"{save_path}.json", "w") as f:
        json.dump(rouge2_scores, f, indent=4)

    make_hist(rouge2_scores, save_path)
    return rouge2_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = gpt_neo_tokenizer()

    model_names = ['gpt2_128_8','gpt2_256_8','gpt2_512_8','gpt2_128_12','gpt2_256_12', 'gpt2_512_12']
    for model_name in model_names:
        n_embd = int(model_name.split('_')[1])
        n_layer = int(model_name.split('_')[2])
        model = GPT2.load_from_checkpoint(
            f"checkpoints/{model_name}.ckpt", tokenizer=tokenizer, n_embd=n_embd, n_layer=n_layer
        ).to(device)

        with open("data/100_train_stories.json", "r") as f:
            original_stories = json.load(f)
        
        generated_stories = get_completions(original_stories, model)

        rouge2_completion_orig(original_stories, generated_stories, model_name)
        max_rouge2_self(generated_stories, model_name)

if __name__ == "__main__":
    main()


# def max_rouge_train_comp(train_stories, generated_stories, model_name, n=2):
#     rouge2_scores = []
#     for i in range(len(generated_stories)):
#         cur_rouge = []
#         for j in range(len(train_stories)):
#             cur_rouge.append(calculate_rouge(train_stories[i], generated_stories[j], n=n, metric='precision'))
#         rouge2_scores.append(max(cur_rouge))

#     save_path = f"data/max_rouge_train_comp/{model_name}"
#     os.makedirs(f"data/max_rouge_train_comp", exist_ok=True)

#     with open(f"{save_path}.json", "w") as f:
#         json.dump(rouge2_scores, f, indent=4)

#     make_hist(rouge2_scores, save_path)

#     return rouge2_scores

