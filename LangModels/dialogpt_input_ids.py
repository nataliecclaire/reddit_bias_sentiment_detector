import torch
import pandas as pd
import math
import time
from transformers import AutoModelWithLMHead, AutoTokenizer


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'

pretrained_model = 'microsoft/DialoGPT-small' # 'gpt2' # 'roberta-base' # 'bert-base-uncased' # 'minimaxir/reddit' # 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelWithLMHead.from_pretrained(pretrained_model)

with open(data_path + 'bias_annotated/orientation/orientation_bias_manual_train.txt') as f:
    lines = [line.rstrip() for line in f]

for sent in lines:
    print(sent)
    input_ids = tokenizer(sent, add_special_tokens=True, truncation=True, max_length=32)
    tokens = tokenizer.tokenize(sent)
    print(input_ids)
    print(tokens)