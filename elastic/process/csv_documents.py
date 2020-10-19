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
        'purpose': doc['purpose'],
        'documents_submission_date_start': doc['documents_submission_date_start'],
        'documents_submission_date_end': doc['documents_submission_date_end'],
        'documents_submission_time_end': doc['documents_submission_time_end'],
        'is_urgent': doc['is_urgent'],
        'competition_results_date': doc['competition_results_date'],
        'fund_name': doc['fund_name'],
        'country': doc['country'],
        'allowed_participant_countries': doc['allowed_participant_countries'],
        'id': doc['id'],
        'field_of_knoweledge': doc['field_of_knoweledge'],
        # 'specific_objectives': doc['specific_objectives'], #!
        # 'expected_impact': doc['expected_impact'], #!
        'topic_description': doc['topic_description'], #!
        'allowed_participants': doc['allowed_participants'],
        'allowed_participants_age': doc['allowed_participants_age'],
        'program_budget': doc['program_budget'],
        'project_budget': doc['project_budget'],
        'type': doc['type'],
        'is_scientific_degree_required': doc['is_scientific_degree_required'],
        'minimal_scientist_experience': doc['minimal_scientist_experience'],
        'link': doc['link'],
        # 'specific_objectives_vector': emb,
        # 'expected_impact_vector': emb,
        'topic_description_vector': emb
    }


def load_dataset(path):
    docs = []
    df = read_csv(path, na_filter=False)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': getattr(series, 'title', ''),
            'purpose': getattr(series, 'purpose', ''),
            'documents_submission_date_start': getattr(series, 'documents_submission_date_start', 'null'),
            'documents_submission_date_end': getattr(series, 'documents_submission_date_end', 'null'),
            'documents_submission_time_end': getattr(series, 'documents_submission_time_end', ''),
            'is_urgent': getattr(series, 'is_urgent', ''),
            'competition_results_date': getattr(series, 'competition_results_date', 'null'),
            'fund_name': getattr(series, 'fund_name', ''),
            'country': getattr(series, 'country', ''), 
            'allowed_participant_countries': getattr(series, 'allowed_participant_countries', ''),
            'id': getattr(series, 'id', ''),
            'field_of_knoweledge': getattr(series, 'field_of_knoweledge', ''),
            'topic_description': str(getattr(series, 'topic_description', '')) + str(getattr(series, 'specific_objectives', '')) + str(getattr(series, 'expected_impact', '')), # concatenate those line for recognition purposes
            'allowed_participants': getattr(series, 'allowed_participants', ''),
            'allowed_participants_age': getattr(series, 'allowed_participants_age', ''),
            'program_budget': getattr(series, 'program_budget', ''),
            'project_budget': getattr(series, 'project_budget', ''),
            'type': getattr(series, 'type', ''),
            'is_scientific_degree_required': getattr(series, 'is_scientific_degree_required', ''),
            'minimal_scientist_experience': getattr(series, 'minimal_scientist_experience', ''),
            'link': getattr(series, 'link', '')
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        for doc in batch_docs:
            if doc['topic_description'] != '':
                embeddings = bc.encode([doc['topic_description']])
                for emb in embeddings:
                    yield emb
            else:
                yield []


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w', encoding='utf8') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', default='data/grants.csv', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='grants', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
