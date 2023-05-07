from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Initialize a new GPT-2 configuration and model
config = GPT2Config(vocab_size=10000, n_positions=256, n_ctx=256, n_embd=512, n_layer=12, n_head=8)
#config = GPT2Config(vocab_size=10000, n_positions=256, n_ctx=256, n_embd=512, n_layer=12, n_head=8)

model = GPT2LMHeadModel(config=config)

# Load your own dataset and tokenize it
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
train_data = TextDataset(tokenizer=tokenizer, file_path='train.txt', block_size=128)

# Define your training hyperparameters
training_args = TrainingArguments(
    output_dir='./models',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Define the training trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    prediction_loss_only=True,
)
trainer.train()

# Use the trained model to generate text
def generate_text(history, prompt, model, tokenizer):
    context = ' '.join(history[-4:]) # Use a sliding window of 4 previous turns as context
    input_text = context + ' ' + prompt
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=100, temperature=0.7, do_sample=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # output_ids = model.generate(input_ids, max_length=100, num_return_sequences=3, temperature=0.7, do_sample=True)
    # output_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return output_text
