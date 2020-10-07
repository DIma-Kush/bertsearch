"""
Example script to create elasticsearch documents.
"""
import argparse
import json

from bert_serving.client import BertClient
bc = BertClient(output_fmt='list', check_length=False)


def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'name': doc['name'],
        'description': doc['description'],
        'description_vector': emb
    }


def load_dataset(path):
    docs = []
    with open(path) as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        
        description = result.get('description', '')
        name = result.get('name', '')
        id = result.get('id', '')

        if description != '\n\n' and description != '':
            doc = {
                'id': id,
                'name': name,
                'description': description
            }
            docs.append(doc)

    return docs


def bulk_predict(docs, batch_size=256):
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        for doc in batch_docs:
            print(doc['id'])
            embeddings = bc.encode([doc['description']])
            for emb in embeddings:
                yield emb


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', default='data/grants_huge.jsonl', help='data for creating documents.')
    parser.add_argument('--save', default='documents_huge.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='grants_huge', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
