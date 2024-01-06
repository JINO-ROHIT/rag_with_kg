import weaviate
import os

from dotenv import load_dotenv
load_dotenv()

auth_config = weaviate.AuthApiKey(api_key = os.environ['WEAVIATE_API_KEY'])

client = weaviate.Client(
  url = os.environ['WEAVIATE_CLIENT_URL'],
  auth_client_secret = auth_config
)

print('authentication successful')