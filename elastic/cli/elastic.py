# script for testing locally
from elasticsearch import Elasticsearch
from bert_serving.client import BertClient
from elasticsearch.exceptions import ConnectionError, NotFoundError

# total number of responses
SEARCH_SIZE = 1

# establishing connections
bc = BertClient(ip='localhost', output_fmt='list', check_length=False)
client = Elasticsearch('localhost:9200')

# this query is used as the search term, feel free to change
query = 'machine learning'
query_vector = bc.encode([query])[0]

query2 = 'завідувач 1312312323'
query_vector2 = bc.encode([query2])[0]

script_query = {
    "function_score": {
        # "query": {
        #     "match": {
        #         "title": {
        #             "query": "Similarity Search "
        #         }
        #     }
        # },
        "functions": [
            {
                "script_score": {
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'topic_description_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
            # {
            #     "script_score": {
            #         "script": {
            #             "source": "cosineSimilarity(params.query_vector, 'topic_description_vector') + 1.0",
            #             "params": {
            #                 "query_vector": query_vector2
            #             }
            #         }
            #     }
            # }
        ],
        "score_mode": "max",
        "boost_mode": "multiply",
        "boost": 1
    }
}

try:
    response = client.search(
         index='grants',  # name of the index
         body={
             "size": SEARCH_SIZE,
             "query": script_query,
             "_source": {"includes": ["title", "abstract"]}
         }
     )
    print(response)
except ConnectionError:
    print("[WARNING] docker isn't up and running!")
except NotFoundError:
    print("[WARNING] no such index!")
