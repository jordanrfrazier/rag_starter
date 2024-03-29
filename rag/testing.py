import os
from dotenv import load_dotenv

from ingestion.load import load_dataset, load_from_url
from ingestion.chunking import token_text_split
from ingestion.embeddings import open_ai_embeddings
from ingestion.storage.astradb import initialize_astra_db
from retrieval.chains import as_retriever, basic_chat, conversational_retrieval_with_memory
from generation.models import chat_open_ai
from generation.query_loop import query_loop

from langchain_core.documents import Document


def basic_astra_qa_model():
    load_dotenv()

    # Ingestion
    dataset = load_dataset("datastax/philosopher-quotes", split="train")
    documents = []
    for entry in dataset:
        metadata = {"author": entry["author"]}
        doc = Document(page_content=entry["quote"], metadata=metadata)
        documents.append(doc)

    # Chunking
    split_documents = token_text_split(documents, chunk_size=10, chunk_overlap=2)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    vstore = initialize_astra_db(collection, embedding, token, api_endpoint)
    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval
    retriever = as_retriever(vstore)

    # Generation
    prompt = """
    You are a philosopher that draws inspiration from great thinkers of the past
    to craft well-thought answers to user questions. Use the provided context as the basis
    for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    Your answers must be concise and to the point, and refrain from answering about other topics than philosophy.
    """

    model = chat_open_ai()

    print(f"Initializing model with prompt:\n{prompt}")
    chain = basic_chat(retriever, model, prompt)

    while True:
        query = input("Enter a question:\n")
        response = chain.invoke(query)
        print(f"Response:\n{response}")


def basic_astra_conversation_model():
    load_dotenv()

    # Ingestion
    # dataset = load_dataset("datastax/philosopher-quotes", split="train")
    # documents = []
    # for entry in dataset:
    #     metadata = {"author": entry["author"]}
    #     doc = Document(page_content=entry["quote"], metadata=metadata)
    #     documents.append(doc)
    documents = load_from_url(
        "https://raw.githubusercontent.com/CassioML/cassio-website/main/docs/frameworks/langchain/texts/amontillado.txt",
        "data/amontillado.txt",
    )

    # Chunking
    split_documents = token_text_split(documents, chunk_size=1000, chunk_overlap=2)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    vstore = initialize_astra_db(collection, embedding, token, api_endpoint)
    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval
    retriever = as_retriever(vstore)

    # Generation
    # prompt = """
    # You are a philosopher that draws inspiration from great thinkers of the past
    # to craft well-thought answers to user questions. Use the provided context as the basis
    # for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    # Your answers must be concise and to the point, and refrain from answering about other topics than philosophy.
    # """

    prompt = """
    You are a very rude assistant that only knows about the provided context. Do not answer
    any questions that are not related to the context. You are also very rude, so you can be
    sarcastic and mean to the user.
    """

    model = chat_open_ai()

    print(f"Initializing model with prompt:\n{prompt}")
    chain = conversational_retrieval_with_memory(retriever, model, prompt)

    query_loop(chain)


basic_astra_conversation_model()
