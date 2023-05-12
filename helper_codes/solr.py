import pysolr
import json
import requests
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
#Create a Solr client
solr = pysolr.Solr('http://69.55.55.225:8983/solr/#/chatbot_new/')
push_url = 'http://69.55.55.225:8983/solr/#/chatbot_new/'
fetch_url = 'http://69.55.55.225:8983/solr/#/chatbot_new/'

def get_responses_custom(input):
    input = input.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    user_query_words = input.split()
    user_query_words_filtered = [word for word in user_query_words if word not in stop_words]
    # user_query_filtered = ' '.join(user_query_words_filtered)
    # solr_query = " ".join(["question:*{}*".format(word) for word in user_query_filtered.split(" ")])
    solr_query = " ".join(["question:*{}*".format(word) for word in user_query_words_filtered])
    solr_query = solr_query.replace("* ", "* AND ").replace(" *", "") + "~"
    solr_params = {'q': solr_query, 'fl': 'question, answer'}
    response = requests.get(fetch_url, params=solr_params)
    # parse the response and extract the answer
    json_response = response.json()
    return json_response

    
def push_data(dataset):
    # Add the dataset to the Solr core
    solr.add(dataset)
    # Commit the changes
    solr.commit()

def delete_all():
    solr.delete(q='*:*')

# delete_all()

def push_json(data):
    # Define the headers for the request
    headers = {'Content-type': 'application/json'}
    # Convert the data to JSON format
    json_data = json.dumps(data)
    # Post the data to the Solr API
    response = requests.post(push_url, headers=headers, data=json_data)
    # Check the response status code
    if response.status_code == 200:
        print('Data indexed successfully.')
    else:
        print(f'Error indexing data: {response.content}')

# push_json("datasets/factoid/combined_train.json")

def get_responses(input):
    # Execute the query

    results = solr.search(q=input)
    # Print the answer
    for result in results:
        answer = result['answer']
        print(f"Answer: {answer}")