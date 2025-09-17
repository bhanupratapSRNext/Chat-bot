from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
index_name=os.getenv('pinecone_index_name')


pc = Pinecone(api_key=PINECONE_API_KEY)


def create_pinecone_index():
    try:
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        )
        print(f"Index '{index_name}' created successfully")
    except Exception as e:
        print(f"Error creating index: {str(e)}") 