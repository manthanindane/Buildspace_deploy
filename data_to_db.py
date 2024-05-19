import openai
import langchain
import qdrant_client
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.qdrant import Qdrant
from langchain.llms import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

# print(os.environ['QDRANT_LOCAL_COLLECTION'])

# Reading File
def scraped_data(file):
    with open(file, 'r', encoding='utf-8') as file:
        data = file.read()
    file.close()
    return data


data = scraped_data('data.txt')
# print(data)


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)


docs = text_splitter.split_text(data)
# print(len(docs))


# os.environ['OPENAI_API_KEY']

#Initialize ChatOpenAI and OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
# print(embeddings)


# creating an instance of Qdrant_client
# Initialize Qdrant client and collection
client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_LOCALHOST"),
        api_key=os.getenv("OPENAI_API_KEY")
    )


collection_name = os.getenv("QDRANT_LOCAL_COLLECTION")
# print("Collection name:", collection_name)


import os
# print(os.environ)


collection_name = os.getenv("QDRANT_LOCAL_COLLECTION")
client.recreate_collection(
    collection_name= collection_name,
    vectors_config= models.VectorParams(size = 1536, distance=models.Distance.COSINE))

# Initialising vectorstore
vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_LOCAL_COLLECTION"),
        embeddings=OpenAIEmbeddings()
    )
vectorstore.add_texts(docs)


