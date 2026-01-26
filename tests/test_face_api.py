import base64
import requests

API_URL = "http://localhost:8000/verify"

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

payload = {
    "user_id": "user_001",
    "image_base64": image_to_base64("test_images/selfie1.jpg")
}

response = requests.post(API_URL, json=payload)

print("STATUS:", response.status_code)
print("HEADERS:", response.headers)
print("RAW RESPONSE:")
print(response.text)
