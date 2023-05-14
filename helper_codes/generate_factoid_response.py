import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download

REPO_ID = "shubhankur/chatbot"
FILENAME = "models/factoid_generator.pt"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

import re
def getFactoidResponse(input_text):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  if(len(input_ids)<=0):
    print(input_text)
    return None
  output_ids = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=70,early_stopping=True)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  messages = output_text.split("\n")
  first_bot_response = None
  for message in messages:
    if message.startswith("Bot:"):
        first_bot_response = message.strip()
        break
  return first_bot_response

