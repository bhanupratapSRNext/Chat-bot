from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



#Split the Data into Text Chunks 

# sementic cunking
# def text_split(extracted_data):
#     text_splitter = SemanticChunker(
#             OpenAIEmbeddings(), breakpoint_threshold_type="percentile" 
#             )
#     chunk_texts = text_splitter.create_documents(extracted_data)
#     return chunk_texts


# Recursive text split
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks




#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings




def check_index_exists(pc, index_name):
    indexes = pc.list_indexes()
    # Check if index exists
    if index_name in [index.name for index in indexes]:
        # Get the index statistics
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        # If vector count is greater than 0, index has data
        return stats.total_vector_count > 0
    return False
    