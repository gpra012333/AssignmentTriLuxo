import os
import openai
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the Flask application
app = Flask(__name__)
api = Api(app)

# Load the FAISS vector store
vector_store = FAISS.load('vector_store.faiss')

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

class QueryHandler(Resource):
    def post(self):
        # Get the query from the request
        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'})

        # Generate embeddings for the query
        query_embedding = embeddings.embed_query(query)

        # Find the most similar documents from the vector store
        results = vector_store.query(query_embedding, top_k=5)

        # Return the results
        return jsonify({'results': results})

# Add the QueryHandler resource to the API
api.add_resource(QueryHandler, '/query')

if __name__ == '__main__':
    app.run(debug=True)
