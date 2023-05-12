#get answer from solr
#if solr returns an answer --> Use BERT to find most similar question and respond
#if not ---> Use generator to genrate an answer

import sys 
import os
sys.path.append(os.path.abspath("helper_codes"))
from solr import get_responses_custom
from bert import get_most_similar_response
from generate_factoid_response import getFactoidResponse
from classifier import classify
from generate_chitchat_response import getChitChatResponse

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["OPTIONS", "GET", "POST", "PUT", "DELETE"], allow_headers=["*"])
convstarted = 0
msgcounter = 0

# Define a route to handle incoming chatbot messages
@app.route('/', methods=['GET'])
def index():
    # Get the message from the request body
    # Do some processing with the message (e.g., pass it to a machine learning model)
    message = request
    # Return a response with the chatbot's message
    return jsonify({'message': 'This is a response from the chatbot!'})

@app.route('/chat', methods=['POST'])
def get_user_message():
    # Get the message from the request body
    data = request.get_json()
    user_input = data['message']
    msg = ""
    type = classify(user_input)
    if(type==1):
        user_input = "User: "+user_input
        msg = getChitChatResponse(user_input)
    else:
        msg = getFactoidResponse(user_input)
    # if(convstarted==0):
    #     msg = "Hey! Do you want to participate in an"
    # Return a response with the chatbot's message
    if (msg.startswith("Bot")):
        msg = msg[5:]
    return jsonify({'message': msg})

if __name__ == '__main__':
    app.run()

def get_factoid_response(user_input):
    responses = get_responses_custom(user_input)
    numFound = responses['response']['numFound']
    response = ""
    if(numFound>0):
        docs = responses['response']['docs']
        questions = []
        for each in docs:
            found_question = each['question'][0]
            questions.append(found_question)
        index = get_most_similar_response(user_input, questions)
        doc = docs[index]
        response = doc['answer'][0]
        print("Result found in IR")
        ir_response = response
    else:
        print("Result not found in IR. Generating response")
        response = getResponse(user_input)
        neural_response = response
    print(f"User Query: {user_input}")
    print(f"Response : {response}")
    return response

if __name__ == '__main__':
    app.run()