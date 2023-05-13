import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "shubhankur/chatbot"
id_FILENAME = "models/id.pt"
ego_FILENAME = "models/ego.pt"
superego_FILENAME = "models/superego.pt"

id_model = GPT2LMHeadModel.from_pretrained('gpt2')
ego_model = GPT2LMHeadModel.from_pretrained('gpt2')
superego_model = GPT2LMHeadModel.from_pretrained('gpt2')

id_model_path = hf_hub_download(repo_id=REPO_ID, filename=id_FILENAME)
ego_model_path = hf_hub_download(repo_id=REPO_ID, filename=ego_FILENAME)
superego_model_path = hf_hub_download(repo_id=REPO_ID, filename=superego_FILENAME)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')

id_model.load_state_dict(torch.load(id_model_path,map_location=torch.device('cpu')))
ego_model.load_state_dict(torch.load(ego_model_path,map_location=torch.device('cpu')))
superego_model.load_state_dict(torch.load(superego_model_path,map_location=torch.device('cpu')))

def getIdResponse(input_text):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  if(len(input_ids)<=0):
    print(input_text)
    return None
  output_ids = id_model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=70,early_stopping=True)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  messages = output_text.split("\n")
  first_bot_response = None
  for message in messages:
    if message.startswith("Bot:"):
        first_bot_response = message.strip()
        break
  return first_bot_response

def getEgoResponse(input_text):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  if(len(input_ids)<=0):
    print(input_text)
    return None
  output_ids = ego_model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=70,early_stopping=True)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  messages = output_text.split("\n")
  first_bot_response = None
  for message in messages:
    if message.startswith("Bot:"):
        first_bot_response = message.strip()
        break
  return first_bot_response

def getSuperEgoResponse(input_text):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  if(len(input_ids)<=0):
    print(input_text)
    return None
  output_ids = superego_model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=70,early_stopping=True)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  messages = output_text.split("\n")
  first_bot_response = None
  for message in messages:
    if message.startswith("Bot:"):
        first_bot_response = message.strip()
        break
  return first_bot_response