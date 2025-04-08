import requests
import json
import base64

# Function to convert image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_chatgpt_with_image(image_path, question):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",  # Replace with your actual API key
        "Content-Type": "application/json"
    }

    image_base64 = encode_image(image_path)

    payload = {
        "model": "gpt-4o-mini",  # Use the latest model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]}
        ],
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
image_path = "/Users/kumarsatyam/Desktop/hackathon/pizza.png"  # Replace with your image path
question = "How many calories are in this pizza, is it healthy?"
response = ask_chatgpt_with_image(image_path, question)
print(response)
