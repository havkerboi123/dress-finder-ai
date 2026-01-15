import chromadb
import os
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
#uses the ViT-B-32 model by default. 

CHROMA_PATH = "/Users/mhmh/Desktop/dress-finder-ai/chroma_db"

# Use PersistentClient instead of Client
client = chromadb.PersistentClient(path=CHROMA_PATH)

data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.get_or_create_collection(
    name="sapphire_collection",
    embedding_function=embedding_function,
    data_loader=data_loader
)

image_folder = "/Users/mhmh/Desktop/dress-finder-ai/images"
images = [
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_ids = [str(i + 1) for i in range(len(images))]

if images:
    collection.add(ids=image_ids, uris=images)
    print(f"Added {len(images)} images to Chroma.")
    print(f"Database saved to: {CHROMA_PATH}")
else:
    print("No images found to embed.")