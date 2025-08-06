import base64, requests, urllib.parse

# 1) Get token
client_id     = "83f63049def741f48e05e1d2b0b89bad"
client_secret = "97c4c4a9f5564091818655a9183a6673"

auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
token_res = requests.post(
    "https://accounts.spotify.com/api/token",
    headers={ "Authorization": f"Basic {auth_header}" },
    data={ "grant_type": "client_credentials" }
)
token = token_res.json()["access_token"]

# 2) Search
artist = "Coldplay"
track  = "Yellow"
query = urllib.parse.quote(f"track:{track} artist:{artist}")

res = requests.get(
    f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1",
    headers={ "Authorization": f"Bearer {token}" }
)
data = res.json()

if data["tracks"]["items"]:
    item = data["tracks"]["items"][0]
    print("Track:", item["name"])
    print("Artist:", item["artists"][0]["name"])
    print("URL:", item["external_urls"]["spotify"])
else:
    print("No match found.")
