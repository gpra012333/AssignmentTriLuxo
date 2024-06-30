import openai
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Initialize the WebBaseLoader with the URL
loader = WebBaseLoader(urls=[url])

# Load the data from the URL
documents = loader.load()

# Create a text splitter to split long documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Generate embeddings for the documents
document_embeddings = embeddings.embed_documents(split_documents)

# Initialize the FAISS vector store
vector_store = FAISS(embeddings.dimension)

# Add the embeddings to the vector store
vector_store.add_embeddings(document_embeddings)

# Save the vector store to disk
vector_store.save('vector_store.faiss')

print("Embeddings have been created and stored in vector_store.faiss")
