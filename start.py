import os
# create or recreate an index (database)
os.system("python .\elastic\create_index.py")
print('create index finished')

# create a documents
os.system("python .\elastic\process\csv_documents.py")
print('create documents finished')

# import/index a documents into the database(index)
os.system("python .\elastic\index_documents.py")
print('index documents finished')