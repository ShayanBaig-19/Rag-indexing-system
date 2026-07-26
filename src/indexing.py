#all imports
import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from pinecone import Pinecone

#now code from ipynb just create a functions so that work will be done in a single function call

def load_environment():
    load_dotenv()

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
documents = load_pdf("../data/summary-islamhand (1).pdf")

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





