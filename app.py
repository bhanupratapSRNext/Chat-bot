from flask import Flask, render_template, jsonify, request
from Blueprints.helper import download_hugging_face_embeddings,load_pdf_file,text_split,check_index_exists
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


from dotenv import load_dotenv
from Blueprints.prompt import *
from pinecone import Pinecone 
from typing import Dict
import os

app = Flask(__name__)

load_dotenv()

index_name=os.getenv('PINECONE_INDEX_NAME')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

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


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
         MessagesPlaceholder("history"), 
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
base_chain = create_retrieval_chain(retriever, question_answer_chain)


SESSION_STORE: Dict[str, ChatMessageHistory] = {}
ROLLING_K = 8  # keep the last 8 messages (user + ai combined)

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Returns a per-session ChatMessageHistory and enforces a rolling window.
    We trim here (on every access) to avoid subclassing Pydantic models.
    """
    hist = SESSION_STORE.get(session_id)
    if hist is None:
        hist = ChatMessageHistory()
        SESSION_STORE[session_id] = hist

    # enforce rolling window
    msgs = hist.messages
    if len(msgs) > ROLLING_K:
        hist.messages = msgs[-ROLLING_K:]
    return hist

# Wrap the chain with RunnableWithMessageHistory
chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",      # key in your invoke payload that holds the user's message
    history_messages_key="history", 
    output_messages_key="answer",
    )

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_query = request.form["msg"]
    session_id = request.form.get("session_id", "default-session")

    response = chain.invoke(
        {"input": user_query},
        config={"configurable": {"session_id": session_id}}
        )
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
