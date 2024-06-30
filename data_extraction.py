from langchain.document_loaders import WebBaseLoader

# URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Initialize the WebBaseLoader with the URL
loader = WebBaseLoader(urls=[url])

# Load the data from the URL
documents = loader.load()

# Print the extracted data
for doc in documents:
    print(doc)
