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
personalityEvaluation = 0
awaitingPersonalityEvaluation = 0
userpersonality = ""

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
    global awaitingPersonalityEvaluation 
    global personalityEvaluation
    if(awaitingPersonalityEvaluation==1):
        if(user_input == "No"):
            personalityEvaluation = -1
            awaitingPersonalityEvaluation = 0
            return jsonify({'message': "Okay! Let's Chat Then"})
        elif(user_input =="Yes"):
            personalityEvaluation = 1
            return jsonify({'message': "Please type 1 for id, 2 for ego and 3 for superego"})
        elif(user_input == "1" or user_input == "2" or user_input == "3"):
            if(user_input == "1"):
                userpersonality="id"
            if(user_input== "2"):
                userpersonality="ego"
            if(user_input=="3"):
                userpersonality="superego"
            personalityEvaluation = 1
            awaitingPersonalityEvaluation=0
        else:
            personalityEvaluation = -1
            awaitingPersonalityEvaluation = 0
    type = classify(user_input)
    if(type==1):
        user_input = "User: "+user_input
        if(personalityEvaluation==0):
            msg = "Do want to participate in personality evaluation?"
            awaitingPersonalityEvaluation = 1
        elif(personalityEvaluation=1):
            msg = "getSigmudResponse"
        else:
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