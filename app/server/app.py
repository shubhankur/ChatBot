#get answer from solr
#if solr returns an answer --> Use BERT to find most similar question and respond
#if not ---> Use generator to genrate an answer

import sys 
import os
import re
sys.path.append(os.path.abspath("helper_codes"))
import helper_codes
from helper_codes.solr import get_responses_custom
from helper_codes.bert import get_most_similar_response
from helper_codes.generate_factoid_response import getFactoidResponse
from helper_codes.classifier import classify
from helper_codes.generate_chitchat_response import getChitChatResponse
from helper_codes.generate_sigmund_response import getIdResponse, getEgoResponse, getSuperEgoResponse

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["OPTIONS", "GET", "POST", "PUT", "DELETE"], allow_headers=["*"])
convstarted = 0
msgcounter = 0
personalityEvaluation = 0
awaitingPersonalityEvaluation = 0
userpersonality = ""
conv_history = ""
conv_counter = 0

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
    del_flag = 0
    data = request.get_json()
    user_input = data['message']
    msg = ""
    global awaitingPersonalityEvaluation 
    global personalityEvaluation
    global userpersonality
    global conv_history
    global conv_counter
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
            return jsonify({'message': "Thanks for your input! Let's Chat Then"})
        else:
            personalityEvaluation = -1
            awaitingPersonalityEvaluation = 0
    type = classify(user_input)
    if(type==1):
        user_input = "User: "+user_input
        if(personalityEvaluation==0):
            msg = "Do want to participate in personality evaluation?"
            awaitingPersonalityEvaluation = 1
        elif(personalityEvaluation==1):
            if(userpersonality=="id"):
                msg = getIdResponse(user_input)
            elif(userpersonality=="ego"):
                msg = getEgoResponse(user_input)
            elif(userpersonality=="superego"):
                msg = getSuperEgoResponse(user_input)
        else:
            msg = getChitChatResponse(user_input)
    else:
        msg = get_factoid_response(user_input)
    if (msg.startswith("Bot")):
        msg = msg[5:]
    sentences = re.split(r'[.!?]+',msg)

    # Create a list of unique sentences in the order they appear
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence not in unique_sentences:
            if sentence not in ("", " ", "."):
                unique_sentences.append(sentence)

    # Join the unique sentences back into a single string
    filtered_response = ". ".join(unique_sentences)

    conv_history=conv_history + user_input + "\n"
    conv_history=conv_history +filtered_response + "\n"
    conv_counter+=1
    if(conv_counter>=20):
        conv_history = ""
        conv_counter = 0
        del_flag = 1
    return jsonify({'message': filtered_response, 'personality':userpersonality, 'count':conv_counter, 'del_flag':del_flag})

@app.route('/eval', methods=['POST'])
def eval():
    data = request.get_json()
    user_input = data['option']
    global personalityEvaluation
    global userpersonality
    if(user_input=="id"):
        userpersonality="id"
        personalityEvaluation = 1
    elif(user_input=="ego"):
        userpersonality="ego"
        personalityEvaluation = 1
    elif(user_input=="super_ego"):
        userpersonality="superego"
        personalityEvaluation = 1
    elif(user_input=="reset"):
        userpersonality = ""
        personalityEvaluation = -1
    # Return a response with the chatbsot's message
    return jsonify({'message': 'This is a response from the chatbot!'})

@app.route('/clear', methods = ['POST'])
def clear_chat():
    global conv_history
    global conv_counter
    conv_history = ""
    conv_counter = 0
    return jsonify({'message': 'Chat Cleared'})


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
        index, score = get_most_similar_response(user_input, questions)
        doc = docs[index]
        if(score<0.4):
            getFactoidResponse(user_input)
        response = doc['question'][0] +"\n" + doc['answer'][0]
        print("Result found in IR")
        print(score)
        ir_response = response
    else:
        print("Result not found in IR. Generating response")
        response = getFactoidResponse(user_input)
        neural_response = response
    print(f"User Query: {user_input}")
    print(f"Response : {response}")
    return response

if __name__ == '__main__':
    app.run()