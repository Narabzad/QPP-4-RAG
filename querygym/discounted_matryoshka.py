import numpy as np
import json
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model_name=model_name.split('/')[-1]
def generate_embeddings(texts):
    # Encode the list of texts in a batch
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get embeddings from the model
    outputs = model(**inputs)
    
    # Use the mean of the last hidden state outputs as embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    return embeddings


def reciprocal_volume(query_embedding, doc_embeddings):
    """
    Computes the Reciprocal Volume (RV) for a given query and top-k document embeddings.
    
    Parameters:
    - query_embedding: np.ndarray of shape (d,)
        Dense embedding of the query.
    - doc_embeddings: np.ndarray of shape (k, d)
        Dense embeddings of the top-k retrieved documents.
    
    Returns:
    - rv_score: float
        Reciprocal Volume score.
    """
    # Stack query and document embeddings to compute the hypercube
    combined_embeddings = np.vstack([query_embedding, doc_embeddings])  # Shape: (k+1, d)
    
    # Calculate the side length for each dimension
    max_vals = np.max(combined_embeddings, axis=0)
    min_vals = np.min(combined_embeddings, axis=0)
    side_lengths = max_vals - min_vals  # Shape: (d,)
    
    # Compute the log of side lengths
    log_side_lengths = np.log(side_lengths)
    
    # Compute the Reciprocal Volume
    rv_score = -1 / np.sum(log_side_lengths)
    
    return rv_score


def discounted_matryoshka(query_embedding, doc_embeddings):
    """
    Computes the Discounted Matryoshka (DM) predictor without rank sensitivity.
    
    Parameters:
    - query_embedding: np.ndarray of shape (d,)
        Dense embedding of the query.
    - doc_embeddings: np.ndarray of shape (k, d)
        Dense embeddings of the top-k retrieved documents.
    
    Returns:
    - dm_score: float
        Discounted Matryoshka score.
    """
    k = doc_embeddings.shape[0]
    discount_factor = 0.9  # Constant discount factor
    dm_score = 0.0
    
    for j in range(1, k + 1):
        current_docs = doc_embeddings[:j, :]
        rv_j = reciprocal_volume(query_embedding, current_docs)
        dm_score += rv_j * discount_factor
    
    return dm_score



output=f'discounted_matryoshka_{model_name}_triviaqa_test.txt'
# Load the dataset
with open('data/triviaqa_llama3_8B_dataset_test.json', 'r') as file:
    dataset = json.load(file)

done=[]
try:
    done_file = open(output,'r')
    
    for line in done:
        pid, exact_match, rv_score = line.strip().split()
        done.append(pid)
except:
    pass
output=open(output,'a')

for pid in dataset:
    if pid in done:
        continue
    # Example usage
    main_prompt = dataset[pid]['main']['prompt']
    vaariations=[]
    for v in  dataset[pid]['variation']:
        vaariations.append(v['prompt'])
    mp_emb =  generate_embeddings([main_prompt])
    v_emb = generate_embeddings(vaariations)

    rv_score = discounted_matryoshka(mp_emb, v_emb)
    output.write(f"{pid} {dataset[pid]['main']['exact_match']} {rv_score}\n")
    print(rv_score, dataset[pid]['main']['exact_match'])
output.close()