import chromadb
from chromadb.config import Settings
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Path where Chroma DB is stored
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sapphire_collection"

# Initialize Chroma client (persistent)
client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    )
)

# Same embedding + loader as indexing time
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

# Load existing collection
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    data_loader=data_loader
)

# -------- QUERY --------
query_text = "blue embroidered women dress"
n_results = 5

results = collection.query(
    query_texts=[query_text],
    n_results=n_results,
    include=["uris", "distances", "metadatas"]
)

# -------- DISPLAY RESULTS --------
print(f"\nTop {n_results} results for: '{query_text}'\n")

for i in range(len(results["uris"][0])):
    print(f"Result #{i + 1}")
    print("Image Path:", results["uris"][0][i])
    print("Distance:", results["distances"][0][i])
    print("Metadata:", results["metadatas"][0][i])
    print("-" * 40)
