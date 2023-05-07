from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare your dataset
train_data = TextDataset(tokenizer=tokenizer, file_path='train.txt', block_size=128)

# Define your fine-tuning hyperparameters
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Define the fine-tuning trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    prediction_loss_only=True,
)
trainer.train()

# Evaluate the fine-tuned model
eval_data = TextDataset(tokenizer=tokenizer, file_path='valid.txt', block_size=128)
eval_loss = trainer.evaluate(eval_data)

# Use the fine-tuned model to generate responses in your chatbot
def generate_response(user_input, model, tokenizer):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, temperature=0.7, do_sample=True)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text
