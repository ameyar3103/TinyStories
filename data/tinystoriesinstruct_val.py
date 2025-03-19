from datasets import load_dataset
import random
import json
dataset = load_dataset("roneneldan/TinyStoriesInstruct")

# print(dataset)
# indices=random.sample(range(0,len(dataset['validation'])), 50)

first_50_val_stories = []
tempstring=""
for a in dataset['validation']['text']:
    if a[:5]=="Story":
        tempstring+="Story: "
        first_50_val_stories.append(tempstring)
        tempstring=""
    elif(a[:7]=="Summary" or a[:8]=="Features" or a[:6]=="Random" or a[:5]=="Words"):
        tempstring+=a+"\n"
    
    if(len(first_50_val_stories)==50):
        break

# print(first_50_val_stories[0])
json.dump(first_50_val_stories, open("data/50_val_ins_stories.json", "w"), indent=4)
