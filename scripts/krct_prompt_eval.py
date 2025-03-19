import json
import random
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.gpt2 import GPT2
from utils.llama_together_ai import evaluate_story,evaluate_prompt
from utils.tokenizer import gpt_neo_tokenizer, gpt2_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, num_stories=10, num_repeats=1, option = 1):
    key = "reasoning ability" if option == 1 else ("factual knowledge" if option==0 else "context-tracking ability")
    path = "reasoning_prompts" if option == 1 else ("factual_prompts" if option==0 else "context-tracking_prompts")
    path = "data/"+path+".json"

    with open(path, "r") as f:
        prompts = json.load(f)

    # stories = random.sample(stories, num_stories)
    num_stories = len(prompts)
    factual_knowledge = 0
    stories = []

    for orig_story in tqdm(prompts):
        for _ in range(num_repeats):
            # st = "Given is a prompt. You should Complete it as you find best."
            # ll = len(st)
            # story = st + orig_story
            ll=0
            story = orig_story
            story = story.split()

            while 1:
                temp_story = story[:]
                temp_story = " ".join(temp_story)
                story = temp_story
                break

            input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
            output = model.model.generate(
                input_ids,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode(".")[0],
            )
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            story_for_prompt = story[ll:] + " ***" + output_text[len(story) :]

            # print("story:" , story)
            # print("output_text:", output_text)
            print("story_for_prompt:", story_for_prompt)
            stories.append(story_for_prompt)
            
            # eval_msg = evaluate_prompt(story_for_prompt,option)
            # eval_msg = eval_msg.replace('_', ' ')
            # print("eval_msg:", eval_msg)
            # time.sleep(10)
            # evals = json.loads(eval_msg)
            # factual_knowledge += int(evals[key])

    # factual_knowledge = factual_knowledge / (num_stories * num_repeats)

    # return {
    #     key : factual_knowledge,
    # }
    return stories


def main():
    tokenizer = gpt_neo_tokenizer()
    import sys
    embed = 512
    layers = 8
    heads = 4
    # model = GPT2(tokenizer)
    # checkpoint=torch.load("checkpoints/gpt2_128_12.ckpt")
    # model.load_state_dict(checkpoint["state_dict"])
    # model=model.to(device)
    model_name = f'gptc_{embed}_{layers}' if heads == 4 else f'gpt2_{embed}_{layers}_{heads}'
    model = GPT2.load_from_checkpoint(
        f"checkpoints/{model_name}.ckpt", tokenizer=tokenizer
    ).to(device)

    # prompt = "Once upon a time there was"

    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # output = model.generate(input_ids, max_length = 1000, num_beams=1)
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)
    for option in [0,1,2]:
        folder = "reasoning ability" if option == 1 else ("factual knowledge" if option==0 else "context tracking")
        append = "reason" if option == 1 else ("fact" if option==0 else "eval")
        ret = evaluate_model(model, tokenizer, option=option)
        print(ret)
        json.dump(ret, open(f"data/{folder}/{model_name}_{append}.json", "w"), indent=4)


if __name__ == "__main__":
    main()
