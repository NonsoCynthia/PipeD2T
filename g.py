import requests

def generate_response(model, prompt, url="http://localhost:11434/api/generate"):
    data = {
        "model": model,
        "prompt": prompt
    }
    # Send the POST request with JSON data
    response = requests.post(url, json=data)
    # Check for successful response
    if response.status_code == 200:
        # Get the JSON response data
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None