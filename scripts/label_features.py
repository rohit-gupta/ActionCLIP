from transformers import AutoTokenizer, CLIPTextModelWithProjection

from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import json
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)

labels_folder = '/home/praveen/Research/ActionCLIP/data/direct_prompting_gpt3.5'
out = {}
for f in os.listdir(labels_folder):
    class_label = f.replace('.json', '')
    sub_action_texts = json.load(open(os.path.join(labels_folder, f), 'r'))['text'].split('\n')
    inputs = [f"a video of a person doing {class_label} while {text}" for text in sub_action_texts]
    inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(device)

    outputs = model(**inputs)
    features = outputs.text_embeds
    class_label = class_label.replace(' ', '')
    out[class_label] = features.cpu().detach().numpy()

pickle.dump(out, open('label_features.pkl', 'wb'))

