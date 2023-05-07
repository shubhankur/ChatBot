from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define a function to generate responses based on user input and context
def generate_response(user_input, conversation_history, model, tokenizer):
    # Encode the conversation history and user input
    input_ids = tokenizer.encode(conversation_history + user_input, return_tensors='pt')

    # Generate a response using the GPT-2 model
    response_ids = model.generate(input_ids, max_length=1000, temperature=0.7, do_sample=True)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Return the response text
    return response_text

# Define the main function to manage the dialog
def main():
    # Initialize an empty conversation history
    conversation_history = ""

    # Start the conversation with a greeting
    print("Hi, welcome to our chatbot! How can I help you today?")

    # Loop through the conversation until the user ends it
    while True:
        # Get the user's input
        user_input = input("> ")

        # Generate a response based on the user input and conversation history
        response_text = generate_response(user_input, conversation_history, model, tokenizer)

        # Print the chatbot's response
        print(response_text)

        # Update the conversation history with the user input and response
        conversation_history += user_input + " " + response_text + " "

        # Check if the user has ended the conversation
        if user_input == "bye":
            break

# Call the main function to start the conversation
if __name__ == "__main__":
    main()
