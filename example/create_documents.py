"""
Example script to create elasticsearch documents.
"""
import argparse
import json

from pandas import read_csv
from bert_serving.client import BertClient
bc = BertClient(output_fmt='list', check_length=False)


def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'title': doc['title'],
        'abstract': doc['abstract'],
        'abstract_vector': emb
    }


def load_dataset(path):
    docs = []
    df = read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': series.title,
            'abstract': series.abstract
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['abstract'] for doc in batch_docs])
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
    parser.add_argument('--data', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='grants', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
