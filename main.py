import os
import torch
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
##
load_dotenv()
PDF_FOLDER = "data"
DB_DIR = "./chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)

def validate_file_constraints(current_count, new_files):
    MAX_BATCH = 50
    MAX_TOTAL = 200
    
    # 1. Check: Batch Size
    if len(new_files) > MAX_BATCH:
        print(f"\n[ERROR] Tek seferde en fazla {MAX_BATCH} dosya yüklenebilir.")
        print(f"Tespit edilen yeni dosya: {len(new_files)}")
        excess = len(new_files) - MAX_BATCH
        print(f"Lütfen {excess} adet dosyayı siliniz.")
        print("Yeni dosyalar (ilk 10):")
        for f in new_files[:10]:
            print(f" - {os.path.basename(f)}")
        if len(new_files) > 10: print(" ...")
        return False

    # 2. Check: Total Size
    if current_count + len(new_files) > MAX_TOTAL:
        print(f"\n[ERROR] Maksimum {MAX_TOTAL} dosya limitine ulaşıldı.")
        print(f"Mevcut: {current_count}, Eklenecek: {len(new_files)}")
        print("Lütfen dosya sayısını azaltınız.")
        return False
        
    return True

def initialize_vectorstore():
    """Initializing Vector Database."""
    print("Initializing Vector Database...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(DB_DIR):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Check for new files to add incrementally
        print("Checking for new files...")
        existing_data = vectorstore.get()
        existing_sources = set()
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    existing_sources.add(m['source'])
        
        # Get all current files
        all_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        new_files = [f for f in all_files if f not in existing_sources]
        
        if new_files:
            if not validate_file_constraints(len(existing_sources), new_files):
                print("Skipping new file addition due to constraints.")
                return vectorstore

            print(f"Found {len(new_files)} new files. Adding to DB...")
            new_docs_content = []
            for f in new_files:
                try:
                    new_docs_content.extend(PyPDFLoader(f).load())
                except Exception as e:
                    print(f"Skipping {f}: {e}")
            
            if new_docs_content:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                chunks = splitter.split_documents(new_docs_content)
                vectorstore.add_documents(chunks)
                print(f"Successfully added {len(new_files)} new files.")
        else:
            print("No new files to add.")
    else:
        print("Creating new ChromaDB from PDFs...")
        files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        
        if not validate_file_constraints(0, files):
            return None

        documents = []
        for f in files:
            try:
                documents.extend(PyPDFLoader(f).load())
            except Exception as e:
                print(f"Skipping {f}: {e}")
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
            docs = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        else:
            print("No PDF documents found.")
            vectorstore = None

    return vectorstore

def create_rag_chain(vectorstore):
    if not vectorstore: return None
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="llama3:8b", temperature=0.1)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful AI assistant.
        Answer the user's question using ONLY the provided CONTEXT.
        Do not make assumptions or use outside knowledge.
        If the answer is not found in the context, say "Bu konuda bağlamda bilgi bulunamadı."
        Respond nicely and strictly in Turkish.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER (in Turkish):
        """
    )
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

def main():
    vectorstore = initialize_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    
    if not rag_chain:
        print("Failed to initialize RAG chain. Please check your data.")
        return

    print("\n")
    print("PAPAGAN")
    print("\n")

    while True:
        try:
            user_input = input("\nUser: ")
            if not user_input.strip(): continue
            if user_input.lower() in ["exit", "quit"]: break
            
            print("Papagan:", end="", flush=True)
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()