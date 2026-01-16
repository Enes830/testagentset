import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

load_dotenv()

# Configuration
NAMESPACE_ID = "ns_cmkgncf8s000104l5l1i4rfq7"
API_TOKEN = "agentset_SrLjTpFifBLnctIA"

# Headers for API requests
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

query = "population of egypt?"

# Search for relevant context using REST API
print(f"Searching for: {query}")

# Make the search request
search_url = f"https://api.agentset.ai/v1/namespace/{NAMESPACE_ID}/search"
payload = {
    "query": query,
    "topK": 10,
    "rerank": True,
    "rerankLimit": 10,
    "rerankModel": "zeroentropy:zerank-2",
    "filter": {},
    "minScore": 0.5,
    "includeRelationships": False,
    "includeMetadata": True,
    "keywordFilter": "",
    "mode": "semantic"
}

response = requests.post(search_url, headers=headers, json=payload)

print(f"Response Status: {response.status_code}")

if response.status_code == 200:
    results = response.json()
    print(f"Full Response:\n{json.dumps(results, indent=2)}")
    
    # Extract context from search results
    context = ""
    if "data" in results:
        for item in results["data"]:
            if "text" in item:
                context += item["text"] + "\n"
    
    print(f"\nExtracted Context:\n{context}")
    print(f"\nRaw Results:\n{json.dumps(results, indent=2)}")
else:
    print(f"Error: {response.status_code}")
    print(f"Response: {response.text}")


# Generate a response using OpenAI (if needed)
# openai = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

# response = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "system",
#             "content": f"Answer questions based on the following context:{context}",
#         },
#         {
#             "role": "user",
#             "content": query,
#         },
#     ],
# )

# print(response.choices[0].message.content)
