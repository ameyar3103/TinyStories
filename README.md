# TinyStories

## Files

- data/tinystories.py: Creates a cache of tinystories dataset and saves it to cache_dir defined in config.py.

- models/gpt2.py: Defines a Pytorch Lightning model based on gpt2 that will be trained from scratch.

- utils/llama_together_ai.py: evaluates a given story on Gemini-1.5-flash using google's API.

- utils/tokenizer.py: imports gpt2 tokenizer and adds required special tokens.

- scripts/{evaluate_*} evaluates various models on grammar, consistency, plot sense, creativity by prompting gemini-1.5-flash

- scripts/interpretability.py runs interpretability analysis.

- scripts/krct_prompts.py, rouge.py do additional analysis as mentioned in the paper. 

- scripts/run_* trains all the models (For Instruct training change the dataset imported at the top of these files)

- data/ have all the analysis results.

- graphs/ have all the graphs plotted in the report.

- training/ uses pytorch lightning to call train on the model. 

## How to run

- All the files in scripts/ are runnable.

- run it as a python module (python3 -m scripts.<module_name>)

## Link to all the trained models

https://drive.google.com/drive/folders/1x4EZk-Ob300gzbrsC0G5sFtYR1uBrZru?usp=sharing

## Link to the Presentation

https://www.canva.com/design/DAGXBy3sDXo/Dj7jOp_PEfV1gCuvFDnJKA/edit?utm_content=DAGXBy3sDXo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Folder tree (with all the trained models and caches)

```
.
├── ANLP_project.pdf
├── checkpoints
│   ├── comp_128_8.ckpt
│   ├── comp_256_8.ckpt
│   ├── comp_512_8.ckpt
│   ├── gpt2_128_12.ckpt
│   ├── gpt2_128_8.ckpt
│   ├── gpt2_256_12.ckpt
│   ├── gpt2_256_8_2.ckpt
│   ├── gpt2_256_8_8.ckpt
│   ├── gpt2_256_8.ckpt
│   ├── gpt2_512_12.ckpt
│   ├── gpt2_512_8.ckpt
│   ├── gptc_128_8.ckpt
│   ├── gptc_256_8.ckpt
│   ├── gptc_512_8.ckpt
│   ├── gpti_128_8.ckpt
│   ├── gpti_256_8.ckpt
│   └── gpti_512_8.ckpt
├── data
│   ├── 100_train_stories.json
│   ├── 50_val_ins_stories.json
│   ├── 50_val_stories.json
│   ├── comapre_evals
│   │   ├── contrastive.png
│   │   ├── heads.png
│   │   ├── hidden_dim.png
│   │   ├── layers.png
│   │   └── pretrained.png
│   ├── comp_orig_rouge
│   │   ├── gpt2_128_12.json
│   │   ├── gpt2_128_12.png
│   │   ├── gpt2_128_8.json
│   │   ├── gpt2_128_8.png
│   │   ├── gpt2_256_12.json
│   │   ├── gpt2_256_12.png
│   │   ├── gpt2_256_8.json
│   │   ├── gpt2_256_8.png
│   │   ├── gpt2_512_12.json
│   │   ├── gpt2_512_12.png
│   │   ├── gpt2_512_8.json
│   │   └── gpt2_512_8.png
│   ├── context tracking
│   │   ├── gpt2_128_12_context.json
│   │   ├── gpt2_128_12_eval.json
│   │   ├── gpt2_128_8_context.json
│   │   ├── gpt2_128_8_eval.json
│   │   ├── gpt2_256_12_context.json
│   │   ├── gpt2_256_12_eval.json
│   │   ├── gpt2_256_8_2_eval.json
│   │   ├── gpt2_256_8_8_eval.json
│   │   ├── gpt2_256_8_context.json
│   │   ├── gpt2_256_8_eval.json
│   │   ├── gpt2_512_12_context.json
│   │   ├── gpt2_512_12_eval.json
│   │   ├── gpt2_512_8_context.json
│   │   ├── gpt2_512_8_eval.json
│   │   ├── gptc_128_8_eval.json
│   │   ├── gptc_256_8_eval.json
│   │   └── gptc_512_8_eval.json
│   ├── context-tracking_prompts.json
│   ├── contrastive_evals
│   │   ├── gpt2_128_12_eval.json
│   │   ├── gpt2_128_8_eval.json
│   │   ├── gpt2_256_12_eval.json
│   │   ├── gpt2_256_8_2_eval.json
│   │   ├── gpt2_256_8_8_eval.json
│   │   ├── gpt2_256_8_eval.json
│   │   ├── gpt2_512_12_eval.json
│   │   ├── gpt2_512_8_eval.json
│   │   ├── gptc_128_8_eval.json
│   │   ├── gptc_256_8_eval.json
│   │   └── gptc_512_8_eval.json
│   ├── factual knowledge
│   │   ├── gpt2_128_12_fact.json
│   │   ├── gpt2_128_8_fact.json
│   │   ├── gpt2_256_12_fact.json
│   │   ├── gpt2_256_8_2_fact.json
│   │   ├── gpt2_256_8_8_fact.json
│   │   ├── gpt2_256_8_fact.json
│   │   ├── gpt2_512_12_fact.json
│   │   ├── gpt2_512_8_fact.json
│   │   ├── gptc_128_8_fact.json
│   │   ├── gptc_256_8_fact.json
│   │   └── gptc_512_8_fact.json
│   ├── factual_prompts.json
│   ├── gpt2_large_eval.json
│   ├── __init__.py
│   ├── instructevals
│   │   ├── gpti_128_8_eval.json
│   │   ├── gpti_256_8_eval.json
│   │   └── gpti_512_8_eval.json
│   ├── max_rouge2_self
│   │   ├── gpt2_128_12.json
│   │   ├── gpt2_128_12.png
│   │   ├── gpt2_128_8.json
│   │   ├── gpt2_128_8.png
│   │   ├── gpt2_256_12.json
│   │   ├── gpt2_256_12.png
│   │   ├── gpt2_256_8.json
│   │   ├── gpt2_256_8.png
│   │   ├── gpt2_512_12.json
│   │   ├── gpt2_512_12.png
│   │   ├── gpt2_512_8.json
│   │   └── gpt2_512_8.png
│   ├── normal_evals
│   │   ├── comp_128_8_eval.json
│   │   ├── comp_256_8_eval.json
│   │   ├── comp_512_8_eval.json
│   │   ├── gpt2_128_12_eval.json
│   │   ├── gpt2_128_8_eval.json
│   │   ├── gpt2_256_12_eval.json
│   │   ├── gpt2_256_8_2_eval.json
│   │   ├── gpt2_256_8_8_eval.json
│   │   ├── gpt2_256_8_eval.json
│   │   ├── gpt2_512_12_eval.json
│   │   ├── gpt2_512_8_eval.json
│   │   ├── gpt2_eval.json
│   │   └── gpt2med_eval.json
│   ├── reasoning ability
│   │   ├── gpt2_128_12_reason.json
│   │   ├── gpt2_128_8_reason.json
│   │   ├── gpt2_256_12_reason.json
│   │   ├── gpt2_256_8_2_reason.json
│   │   ├── gpt2_256_8_8_reason.json
│   │   ├── gpt2_256_8_reason.json
│   │   ├── gpt2_512_12_reason.json
│   │   ├── gpt2_512_8_reason.json
│   │   ├── gptc_128_8_reason.json
│   │   ├── gptc_256_8_reason.json
│   │   └── gptc_512_8_reason.json
│   ├── reasoning_prompts.json
│   ├── tinystoriesInstruct.py
│   ├── tinystoriesinstruct_val.py
│   └── tinystories.py
├── graphs
│   ├── compression.png
│   ├── contrastive.png
│   ├── heads.png
│   ├── hidden_dim.png
│   ├── instruct.png
│   ├── layers.png
│   ├── pretrained.png
│   ├── TrainLoss.png
│   └── ValLoss.png
├── interpret_vis
│   ├── attention_head_0_128_12.png
│   ├── attention_head_0_128_8.png
│   ├── attention_head_0_256_12.png
│   ├── attention_head_0_256_8.png
│   ├── attention_head_0_512_12.png
│   ├── attention_head_0_512_8.png
│   ├── attention_head_1_128_12.png
│   ├── attention_head_1_128_8.png
│   ├── attention_head_1_256_12.png
│   ├── attention_head_1_256_8.png
│   ├── attention_head_1_512_12.png
│   ├── attention_head_1_512_8.png
│   ├── attention_head_2_128_12.png
│   ├── attention_head_2_128_8.png
│   ├── attention_head_2_256_12.png
│   ├── attention_head_2_256_8.png
│   ├── attention_head_2_512_12.png
│   ├── attention_head_2_512_8.png
│   ├── attention_head_3_128_12.png
│   ├── attention_head_3_128_8.png
│   ├── attention_head_3_256_12.png
│   ├── attention_head_3_256_8.png
│   ├── attention_head_3_512_12.png
│   └── attention_head_3_512_8.png
├── models
│   ├── compressor.py
│   ├── gpt2_contrastive.py
│   ├── gpt2.py
│   └── __init__.py
├── neuron_analysis
│   ├── 1-Neuron2-adjs.png
│   └── 4-Neuron1-nouns.png
├── README.md
├── scripts
│   ├── evaluate_gpt2_pretrained.py
│   ├── evaluate_instruct.py
│   ├── evaluate_selftrained.py
│   ├── __init__.py
│   ├── interpretability.py
│   ├── krct_prompt_eval.py
│   ├── model_compare.py
│   ├── rouge_scores.py
│   ├── run_compressor.py
│   ├── run_gpt2_cont.py
│   └── run_gpt2.py
├── training
│   ├── __init__.py
│   └── trainer.py
└── utils
    ├── config.py
    ├── __init__.py
    ├── llama_together_ai.py
    └── tokenizer.py

18 directories, 181 files
```
