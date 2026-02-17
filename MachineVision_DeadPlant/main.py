import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = "https://www.nyckel.com/connect/token"

client_id = os.getenv("NYCKEL_CLIENT_ID")
client_secret = os.getenv("NYCKEL_CLIENT_SECRET")

if not client_id or not client_secret:
    raise RuntimeError("Missing NYCKEL_CLIENT_ID/NYCKEL_CLIENT_SECRET in .env or environment.")

resp = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
    timeout=30,
)
resp.raise_for_status()
token = resp.json()["access_token"]
print("Token ok. Length:", len(token))

ACCESS_TOKEN = token
url = "https://www.nyckel.com/v1/functions/if-a-plant-is-dead/invoke"

# ✅ folder containing images
folder_path = r"C:\Users\akash\Downloads\Dead_plant\TestImages"

# ✅ loop through all images in the folder
for filename in os.listdir(folder_path):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(folder_path, filename)

    with open(image_path, "rb") as f:
        result = requests.post(
            url,
            headers={"accessToken": ACCESS_TOKEN},
            files={"data": f},
            timeout=60
        )

    if result.status_code == 200:
        data = result.json()
        label = data.get("labelName")
        confidence = data.get("confidence")
        print(f"{filename} -> {label} ({confidence:.3f})")
    else:
        print(f"{filename} -> ERROR {result.status_code}: {result.text}")
