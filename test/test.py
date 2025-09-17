from pinecone import Pinecone 
import os


pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )


# List all indexes
indexes = pc.list_indexes()
print(indexes)
# Check if your index exists
index_name = os.getenv('pinecone_index_name')
if index_name in indexes[0].get('name'):
    print(f"Index '{index_name}' exists.")
else:
    print(f"Index '{index_name}' does not exist.")
