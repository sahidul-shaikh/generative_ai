"""
Create vector database with product description.
Store product category and product price in the metadata of each vecor embedding.
"""

# imports

import os
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import pickle
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment and variable
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# constant
DB = "products_vectorstore"
collection_name = "products"

# Log in to HuggingFace
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# Read the file train file. The records will be inserted in the vector db
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

# create instance for Chroma db
client = chromadb.PersistentClient(path=DB)
# Check if the collection exists and delete it if it does
existing_collection_names = [collection.name for collection in client.list_collections()]
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

collection = client.create_collection(collection_name)

# Instantiate vector embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Utility function for for plucking out only the product description
def description(item):
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]


# Insert record in vector db.
# 1. decsription of the product
# 2. vector
# 3. metadata -> product category and product price
for i in tqdm(range(0, len(train), 1000)):
    documents = [description(item) for item in train[i: i+1000]]
    vectors = model.encode(documents).astype(float).tolist()
    metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
    ids = [f"doc_{j}" for j in range(i, i+1000)]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )
