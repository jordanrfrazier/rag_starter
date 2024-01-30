import os
from dotenv import load_dotenv

from ingestion.load import load_dataset, load_from_url
from ingestion.chunking import token_text_split, recursive_character_text_split
from ingestion.embeddings import open_ai_embeddings
from ingestion.storage.astradb import initialize_astra_db
from retrieval.chains import as_retriever, basic_chat, conversational_retrieval_with_memory, retrieval_qa, conversational_llm_with_memory, conversational_retrieval_agent_with_memory
from retrieval.prompts import PHILOSOPHER_PROMPT
from generation.models import chat_open_ai
from generation.query_loop import query_loop

from langchain_core.documents import Document

def load_data(): 
    documents = load_from_url(
        "https://raw.githubusercontent.com/CassioML/cassio-website/main/docs/frameworks/langchain/texts/amontillado.txt",
        "data/amontillado.txt",
    )
    return documents

def split(documents):
    split_documents = token_text_split(documents, chunk_size=512, chunk_overlap=64)
    return split_documents


def prompt():
    prompt = """
    You are a very smart and helpful assistant that only knows about the provided context. Do not answer
    any questions that are not related to the context. Answer with extreme detail, pulling 
    quotes and supporting context directly from the provided context.  
    """
    return prompt


def retrieval_chain(retriever, model, prompt):
    chain = conversational_retrieval_with_memory(retriever, model, prompt)
    return chain


def rag_starter_app():
    # Initialize environment variables
    load_dotenv()
    astra_db_token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    # Ingestion
    documents = load_data()

    # Chunking
    split_documents = split(documents)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    vstore = initialize_astra_db(collection, embedding, astra_db_token, api_endpoint)

    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval / Generation
    my_prompt = prompt()
    retriever = as_retriever(vstore)
    model = chat_open_ai(model="gpt-3.5-turbo")
    chain = conversational_retrieval_agent_with_memory(retriever, model, my_prompt)

    print(f"Initializing model with prompt:\n{my_prompt}")
    query_loop(chain)


rag_starter_app()
