import json
import random
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.gpt2 import GPT2
from utils.llama_together_ai import evaluate_story_instruct
from utils.tokenizer import gpt_neo_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, num_stories=10, num_repeats=2):
    with open("data/50_val_ins_stories.json", "r") as f:
        stories = json.load(f)

    stories = random.sample(stories, num_stories)

    avg_grammar, avg_creativity, avg_consistency, avg_plot_sense,avg_ins_follow_ability = (
        0,
        0,
        0,
        0,
        0,
    )

    for orig_story in tqdm(stories):
        for _ in range(num_repeats):
            st = "Given are the features, words or the summary of a story. You should write the story so that it aligns with them."
            ll = len(st)
            # story = st + orig_story
            story = story

            # while 1:
            #     temp_story = story[:]
            #     temp_story = " ".join(temp_story)
            #     story = temp_story
            #     break
            print("story:", story)
            input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
            output = model.model.generate(
                input_ids,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            story_for_prompt = story[ll:] + output_text[len(story) :]

            # print("story:" , story)
            # print("output_text:", output_text)
            print("story_for_prompt:", story_for_prompt)

            eval_msg = evaluate_story_instruct(story_for_prompt)
            print(eval_msg)
            time.sleep(10)
            evals = json.loads(eval_msg)

            avg_grammar+=evals["grammar"]
            avg_creativity+=evals["creativity"]
            avg_consistency+=evals["consistency"]
            avg_plot_sense+=evals["plot_sense"]
            avg_ins_follow_ability+=evals["instruction_ability"]

    avg_grammar /= num_stories * num_repeats
    avg_creativity /= num_stories * num_repeats
    avg_consistency /= num_stories * num_repeats
    avg_plot_sense /= num_stories * num_repeats
    avg_ins_follow_ability /= num_stories * num_repeats

    return {
        "ins_follow_ability": avg_ins_follow_ability,
        "avg_grammar": avg_grammar,
        "avg_creativity": avg_creativity,
        "avg_consistency": avg_consistency,
        "avg_plot_sense": avg_plot_sense,
    }


def main():
    tokenizer = gpt_neo_tokenizer()

    # model = GPT2(tokenizer)
    # checkpoint=torch.load("checkpoints/gpt2_128_12.ckpt")
    # model.load_state_dict(checkpoint["state_dict"])
    # model=model.to(device)

    model = GPT2.load_from_checkpoint(
        "checkpoints/gpti_512_8.ckpt", tokenizer=tokenizer
    ).to(device)

    # prompt = "Once upon a time there was"

    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # output = model.generate(input_ids, max_length = 1000, num_beams=1)
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)
    ret = evaluate_model(model, tokenizer)
    print(ret)
    json.dump(ret, open("data/instructevals/gpti_128_8_eval.json", "w"), indent=4)


if __name__ == "__main__":
    main()
