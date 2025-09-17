from flask import Flask, render_template, jsonify, request
from Blueprints.helper import download_hugging_face_embeddings,load_pdf_file,text_split,check_index_exists
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from Blueprints.prompt import *
from pinecone import Pinecone 
import os

app = Flask(__name__)

load_dotenv()

index_name=os.getenv('pinecone_index_name')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

# extracted_text_data=load_pdf_file("Data")

# chunk=text_split(extracted_text_data)

# docsear = PineconeVectorStore.from_documents(
#     documents=chunk,
#     index_name=index_name,
#     embedding=embeddings, 
# )

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Check if index exists and has data
if not check_index_exists(pc, index_name):
    # Only do these steps if index doesn't exist or is empty
    extracted_text_data = load_pdf_file("Data")
    chunk = text_split(extracted_text_data)
    
    # Create new embeddings and store in Pinecone
    docsearch = PineconeVectorStore.from_documents(
        documents=chunk,
        index_name=index_name,
        embedding=embeddings,
    )
else:
    # If index exists, just connect to it
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    # input = msg
    # print(input)
    response = rag_chain.invoke({"input": msg})
    # print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)