from setuptools import find_packages, setup

setup(
    name="rag_indexing_system",
    version="0.0.1",
    author="Shayan Baig",
    packages=find_packages(),
    install_requires=[
        "pypdf"
        "langchain",
        "langchain-text-splitters",
        "langchain-mistralai",
        "python-dotenv",
        "pinecone",
        "ipykernel",
        "langchain-community",
    ]
)