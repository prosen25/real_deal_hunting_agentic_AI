from dotenv import load_dotenv
import os
from huggingface_hub import Collection, login
from openai import OpenAI
import chromadb
from tqdm import tqdm
from agents.items import Item

# Load the environment variables
load_dotenv(override=True)

# Global variables
VECTOR_DB = "products_vectorstore"
LITE_MODE = True
EMBEDDING_MODEL = "text-embedding-3-small"

def get_data_from_huggingface() -> list[Item]:
    """ Login into huggingface and get data which will be loaded in vector database """

    # Login to huggingface
    print("Logging into Hugging Face...")
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token, add_to_git_credential=False)

    print("Logged in. Fetching dataset...")

    # Get data from huggingface
    username = "ed-donner"
    dataset_name = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset_name=dataset_name)

    print(f"Downloaded {len(train)} data from Hugging face.")

    return train

def load_vector_db() -> Collection:
    """ Load data into chroma vector database """
    
    # Fetch data from huggingface
    train = get_data_from_huggingface()

    # Get the database client
    print("Creating Chroma DB Client.")
    db_client = chromadb.PersistentClient(path=VECTOR_DB)

    openai = OpenAI()

    # Check if the collection exists, if not create it
    collection_name = "products"
    existing_collection_names = [collection.name for collection in db_client.list_collections()]

    if collection_name not in existing_collection_names:
        collection = db_client.create_collection(name=collection_name)
        print("Loading Chroma DB...")
        for i in tqdm(range(0, len(train), 1000)):
            documents = [item.summary for item in train[i: i+1000]]
            response = openai.embeddings.create(input=documents, model=EMBEDDING_MODEL)
            embeddings = [item.embedding for item in response.data]
            metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
            ids = [f"doc_{j}" for j in range(i, i+1000)]
            ids = ids[:len(documents)]
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    return db_client.get_or_create_collection(collection_name)

if __name__ == "__main__":
    collection = load_vector_db()
    print(f"Loaded {collection.count()} items in vector database!")