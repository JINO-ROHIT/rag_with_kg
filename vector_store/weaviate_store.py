import weaviate
import os
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

class Weaviate_Store:
    def __init__(self, store_name: str) -> None:
      self.store_name = store_name
      auth_config = weaviate.AuthApiKey(api_key = os.environ['WEAVIATE_API_KEY'])

      self.client = weaviate.Client(
        url = os.environ['WEAVIATE_CLIENT_URL'],
        auth_client_secret = auth_config
      )

      class_obj = {
          "class": self.store_name,
          "vectorizer": "none",
      }
      self.client.schema.create_class(class_obj)

      logger.info('Store has been created')

    def store_vectors(self, data):
      self.client.batch.configure(batch_size=100)
      with self.client.batch as batch:
          for i, d in data.iterrows():
            print(f"importing question: {i+1}")

            properties = {
                "source": d["node_1"],
                "relation": d["edge"],
                "target": d["node_2"],
                "chunk": d["chunk"]
            }

            batch.add_data_object(properties, self.store_name, vector=d["vectors"])
      
      total_docs = self.client.query.aggregate(self.store_name).with_meta_count().do()
      logger.info(f'Total items : {total_docs}')
    
    def keyword_search(self, query: str , top_k: int = 5):
      response = (
                  self.client.query.get(self.store_name, ["source", "target", "relation", "chunk"])
                  .with_bm25(query = query)
                  .with_limit(top_k)
                  .do()
                  )
      return response

       