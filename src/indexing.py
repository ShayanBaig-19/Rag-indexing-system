#all imports
import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone

#now code from ipynb just create a functions so that work will be done in a single function call


load_dotenv()



def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
documents = load_pdf("data/summary-islamhand (1).pdf")



def split_documents(documents):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    chunks = text_split.split_documents(documents)
    return chunks
chunks = split_documents(documents)



def create_embeddings(model):
    client = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("mistral")
    )
    text = [chunk.page_content for chunk in model]
    vectors = client.embed_documents(text)
    return vectors
vectors = create_embeddings(chunks)



def connect_pinecone():
    pinecone_client = Pinecone(
        api_key=os.getenv("Pinecone_key")
    )
    index = pinecone_client.Index("rag-index")
    return index
index = connect_pinecone()



def prepare_records(pdf_chunks, embedding_vectors):
    records = []
    for i in range(len(pdf_chunks)):
        current_chunk = pdf_chunks[i]
        current_vector = embedding_vectors[i]
        metadata = current_chunk.metadata.copy()
        metadata["text"] = current_chunk.page_content
        record = {
            "id": "chunk-" + str(i),
            "values": current_vector,
            "metadata": metadata
        }
        records.append(record)
    return records
records = prepare_records(chunks, vectors)

#index.upsert(vectors=records)

# vector is pinecone parameter
# index variable i created
# upsert built in function
# records varibale i created and output 

def store_vectors(pinecone_database, data_to_store):
    pinecone_database.upsert(vectors=data_to_store)

#now putting the value of paramters passed in function calling the real values 
store_vectors(index, records)


