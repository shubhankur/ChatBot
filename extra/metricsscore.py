# -*- coding: utf-8 -*-
"""MetricsScore.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16gOEWyauMHvKH-x41rWl_U0Cg8s_vs0o
"""

# Commented out IPython magic to ensure Python compatibility.
# %env CUDA_LAUNCH_BLOCKING=1

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

! pip install transformers==4.27.4

! pip install torchtext==0.10.1

import torch
device = torch.device("cuda")
torch.cuda.init()

from google.colab import drive
drive.mount('/content/gdrive')

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset

model_path = "/content/gdrive/My Drive/Project/model/chitchat_generator.pt"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.load_state_dict(torch.load(model_path, map_location=device))

import re
def getResponse(input_text, model,tokenizer, device):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  input_ids = input_ids.to(device)
  model = model.to(device)
  attention_mask = torch.LongTensor([1] * len(input_ids))
  output_ids = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=50, num_beams=5, 
                                      num_return_sequences=3, 
                                      no_repeat_ngram_size=2, 
                                      early_stopping=True)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  bot_response = re.search(r'Bot: (.+)', output_text).group(1)

  return bot_response

with open('/content/gdrive/My Drive/Project/dataset/combined_test.txt', 'r') as f:
    test_data = f.readlines()

type(test_data[0])

def prepare(test_data):
  input = {}
  response = {}
  conv_id = 1
  idx = 1
  for i in range(len(test_data)):
    input[idx]=[]
    response[idx]=[]
    if(test_data[i]=="\n" or test_data[i]=='\n'):
      idx+=1
    for data in test_data:
      if(conv_id>=idx):
        break
      if(data.startswith("User")):
        input[conv_id].append(data)
      elif(data.startswith("Bot")):
        response[conv_id].append(data)
      else:
        conv_id+=1
  return input,response

input, response = prepare(test_data)

len(input), len(response)

from nltk.translate.bleu_score import corpus_bleu
def getBLEUScore(input, response, model,tokenizer, device):
  generated_response = {}
  for idx, text in input.items():
    generated_response[idx] = getResponse(text, model,tokenizer, device)
  bleu_score = corpus_bleu(generated_response.values, response.values)
  return bleu_score

input_ids = tokenizer.encode("Hello, I am very sad", return_tensors='pt')
input_ids = input_ids.to(device)
model = model.to(device)
output_ids = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=50, num_beams=5, 
                                      num_return_sequences=3, 
                                      no_repeat_ngram_size=2, 
                                      early_stopping=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
bot_response = re.search(r'Bot: (.+)', output_text).group(1).strip()
output_text, bot_response

bleu_score = getBLEUScore(input, response, model, tokenizer, device)
bleu_score

device