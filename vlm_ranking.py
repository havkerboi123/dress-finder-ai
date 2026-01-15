import base64
from openai import OpenAI

client = OpenAI()

# Encoding to base64 as the model expects
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def rank_images(image_paths, user_prompt):
    """
    Rank images based on user prompt and return the best match URI
    
    Args:
        image_paths: List of 3 image file paths
        user_prompt: User's search query/description
    
    Returns:
        str: URI of the best matched image
    """
    
    # Encode all 3 images first
    image1_base64 = encode_image(image_paths[0])
    image2_base64 = encode_image(image_paths[1])
    image3_base64 = encode_image(image_paths[2])
    
    # Make the API call with all images
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": """You are an expert image ranking system specializing in visual search and matching. Your task is to analyze multiple images and identify which one best matches a user's search query based on visual attributes like color, style, pattern, fit, and overall aesthetic. You must respond with ONLY the exact file path of the best matching image - no explanations, no additional text, just the file path. If there is no best matching dress, suggest closest one."""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""The user is looking for: "{user_prompt}"

Here are 3 images to analyze:

Image 1 path: {image_paths[0]}
Image 2 path: {image_paths[1]}
Image 3 path: {image_paths[2]}

Analyze each image and return ONLY the file path of the best match."""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image1_base64}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image2_base64}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image3_base64}"
                    }
                ]
            }
        ]
    )
    
    # Extract the best match URI
    best_match_uri = response.output_text.strip()
    
    return best_match_uri

# Example usage
if __name__ == "__main__":
    # Example image paths (replace with actual paths)
    image_paths = [
        "/Users/mhmh/Desktop/dress-finder-ai/images/image1.jpg",
        "/Users/mhmh/Desktop/dress-finder-ai/images/image2.jpg",
        "/Users/mhmh/Desktop/dress-finder-ai/images/image3.jpg"
    ]
    
    user_prompt = "elegant red evening dress"
    
    try:
        best_image = rank_images(image_paths, user_prompt)
        print(f"Best match: {best_image}")
    except Exception as e:
        print(f"Error: {e}")