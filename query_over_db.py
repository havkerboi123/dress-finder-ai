import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
import os

CHROMA_PATH = "/Users/mhmh/Desktop/dress-finder-ai/chroma_db"

#connecting to existing database
client = chromadb.PersistentClient(path=CHROMA_PATH)

#same embedding function and data loader
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()


collection = client.get_collection(
    name="sapphire_collection",
    embedding_function=embedding_function,
    data_loader=data_loader
)

def query_by_text(query_text, n_results=5):
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["distances", "uris", "metadatas"]
    )
    
   
    for i, (img_id, distance, uri) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['uris'][0]
    ), 1):
        print(f"\n{i}. Image ID: {img_id}")
        print(f"   Path: {uri}")
        print(f"   Distance: {distance:.4f}")
    
    return results

def display_results(results):
    """Optional: Display images if you have matplotlib"""
    try:
        import matplotlib.pyplot as plt
        
        uris = results['uris'][0]
        n = len(uris)
        
        fig, axes = plt.subplots(1, min(n, 5), figsize=(15, 3))
        if n == 1:
            axes = [axes]
        
        for i, uri in enumerate(uris[:5]):
            img = Image.open(uri)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Result {i+1}")
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("dimag kharab")

# Example usage
if __name__ == "__main__":
   
    results = query_by_text("blue drees plain no patterns etc", n_results=3)
    display_results(results)
    
    