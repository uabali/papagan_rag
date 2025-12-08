from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

try:
    loader = PyPDFLoader("data/yzetik.pdf")
    documents = loader.load()
except Exception as e:
    print(f"PDF install error: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

llm = Ollama(
    model="llama3:8b",
    temperature=0.1
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template = """
You are a retrieval-based assistant.
Answer the user's question ONLY using the given CONTEXT.
Do NOT use external knowledge.
Do NOT make assumptions or hallucinate.

Rules:
- Write the answer in TURKISH but using ONLY ASCII characters.
- Do NOT use Turkish characters like: ç, ğ, ş, ı, İ, ö, ü.
- The answer must be SHORT, CLEAR, and DIRECT.
- If the answer is not found in the context, respond exactly with:
  "Baglamda cevap bulunamadi."

CONTEXT:
{context}

QUESTION:
{question}

ASCII TURKISH ANSWER:
"""
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

print("---RAG---'):")
while True:
    query = input("Kullanici: ")
    if query.lower() == "exit":
        break

    try:
        response = chain.invoke({"query": query})
        if isinstance(response, dict):
            print(f"\nCevap: {response.get('result', response)}")
        else:
            print(f"\nCevap: {response}")
    except Exception as e:
        print(f"Query error: {e}")