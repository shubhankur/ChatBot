import torch
from sentence_transformers import SentenceTransformer
# Load a pre-trained SBERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')
def get_most_similar_response(input_query, responses):
  response_embeddings = model.encode(responses)
  response_embeddings_tensor = torch.tensor(response_embeddings)
  input_query_embedding = model.encode(input_query)
  input_query_embedding_tensor = torch.tensor(input_query_embedding)
  similarity_scores = torch.nn.functional.cosine_similarity(input_query_embedding_tensor, response_embeddings_tensor)
  most_similar_idx = similarity_scores.argmax().item()
  score = similarity_scores[torch.tensor(most_similar_idx)].item()
  return most_similar_idx, score

input_query = "What is the capital of France?"
responses = [
  "The capital of France is Paris.",
  "France is a beautiful country with a rich history.",
  "I have no idea what the capital of France is.",
  "Paris is one of my favorite cities in the world."
]

index, score = get_most_similar_response(input_query, responses)
print(index)