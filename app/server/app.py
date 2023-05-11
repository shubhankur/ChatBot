#get answer from solr
#if solr returns an answer --> Use BERT to find most similar question and respond
#if not ---> Use generator to genrate an answer

import sys 
import os
sys.path.append(os.path.abspath("helper_codes"))
from solr import get_responses_custom
from bert import get_most_similar_response
from generate_factoid_response import getResponse

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

get_factoid_response("Who starred in barefoot?")