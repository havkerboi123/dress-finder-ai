import chromadb
import os
from chromadb.utils.data_loaders import ImageLoader

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
#Uses the ViT-B-32 model with the laion2b_s34b_b79k pretrained weights by default. 


client = chromadb.Client()

data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.create_collection(
    name='sapphire_collection',
    embedding_function=embedding_function,
    data_loader=data_loader
)


image_folder = "/Users/mhmh/Desktop/dress-finder-ai/images"
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

image_ids = [str(i+1) for i in range(len(images))]

img = client.get_collection(name="sapphire_collection")

if images:
    collection.add(ids=image_ids, uris=images)
else:
    print("No images found to embed.")

# # Step 3: Query for the most relevant image
# result = image_vdb.query(query_texts="pink shirt", n_results=1, include=["uris"])

# if result["uris"]:
#     top_image_path = result["uris"][0]
#     print(f"Top matched image: {top_image_path}")
# else:
#     raise ValueError("No relevant image found in the vector database.")

