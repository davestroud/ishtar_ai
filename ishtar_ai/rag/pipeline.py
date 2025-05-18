"""RAG pipeline using LangChain and Llama 4."""
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
import os

# Initialize Pinecone
_pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
_pinecone_env = os.environ.get("PINECONE_ENV", "")
_pinecone_index = os.environ.get("PINECONE_INDEX", "ishtar-ai")

pinecone_client = PineconeClient(api_key=_pinecone_api_key, environment=_pinecone_env)
vectorstore = Pinecone(
    index_name=_pinecone_index,
    embedding=OpenAIEmbeddings(),
    pinecone_api_key=_pinecone_api_key,
    environment=_pinecone_env,
)

llm = OpenAI(model="llama-4")

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

async def query_pipeline(query: str) -> str:
    """Run the RAG pipeline and return the answer text."""
    result = qa_chain.run(query)
    return result
