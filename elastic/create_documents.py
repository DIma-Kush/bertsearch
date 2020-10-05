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
    df = read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': series.title,
            'purpose': series.purpose,
            'documents_submission_date_start': series.documents_submission_date_start,
            'documents_submission_date_end': series.documents_submission_date_end,
            'documents_submission_time_end': series.documents_submission_time_end,
            'is_urgent': series.is_urgent,
            'competition_results_date': series.competition_results_date,
            'fund_name': series.fund_name,
            'country': series.country,
            'allowed_participant_countries': series.allowed_participant_countries,
            'id': series.id,
            'field_of_knoweledge': series.field_of_knoweledge,
            'topic_description': series.topic_description + series.specific_objectives + series.expected_impact, # concatenate those line for recognition purposes
            'allowed_participants': series.allowed_participants,
            'allowed_participants_age': series.allowed_participants_age,
            'program_budget': series.program_budget,
            'project_budget': series.project_budget,
            'type': series.type,
            'is_scientific_degree_required': series.is_scientific_degree_required,
            'minimal_scientist_experience': series.minimal_scientist_experience,
            'link': series.link
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['topic_description'] for doc in batch_docs])
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
    parser.add_argument('--data', default='data/grants.csv', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='grants', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
