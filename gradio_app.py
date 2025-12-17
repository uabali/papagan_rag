import gradio as gr
import os
import glob
import torch
import whisper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Klasörleri oluştur
pdf_folder = "data"
audio_folder = "ses_data"
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

print("🚀 Sistem başlatılıyor...")

# Whisper modelini yükle
print("📥 Whisper modeli yükleniyor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=device)
print(f"✅ Whisper yüklendi ({device})")

# Embeddings modelini yükle
print("📥 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding modeli yüklendi")

# Vector store'u yükle veya oluştur
def initialize_vectorstore():
    """Vector store'u başlat"""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("⚠️ Henüz PDF dosyası yüklenmemiş!")
        return None
    
    print(f"📚 {len(pdf_files)} PDF dosyası bulundu")
    
    if os.path.exists("./chroma_db"):
        print("📂 Mevcut veritabanı yükleniyor...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("🔨 Yeni veritabanı oluşturuluyor...")
        documents = []
        for pdf_path in pdf_files[:5]:  # İlk 5 PDF
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        docs = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    
    print("✅ Veritabanı hazır")
    return vectorstore

vectorstore = initialize_vectorstore()

# LLM'i başlat
print("🤖 Ollama LLM başlatılıyor...")
llm = Ollama(
    model="llama3:8b",
    temperature=0.1
)
print("✅ LLM hazır")

# RAG chain'i oluştur
retriever = None

def create_rag_chain():
    """RAG chain'i oluştur"""
    global retriever
    
    if vectorstore is None:
        return None
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a retrieval-based assistant.
Answer the user's question using ONLY the given CONTEXT.
Do NOT use external knowledge.
Do NOT make assumptions or hallucinate.

Rules:
- Write the answer in TURKISH using ONLY ASCII characters.
- Do NOT use Turkish characters like: ç, ğ, ş, ı, İ, ö, ü.
- The answer should be CLEAR, EXPLANATORY, and 3 to 6 sentences long.
- You may rephrase the context but do NOT add new information.
- If the answer is not found in the context, respond exactly with:
  "Baglamda cevap bulunamadi."

CONTEXT:
{context}

QUESTION:
{question}

DETAILED ASCII TURKISH ANSWER:
"""
    )
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

rag_chain = create_rag_chain()

print("✅ Sistem hazır!\n")


# Gradio fonksiyonları
def transcribe_audio(audio):
    """Ses kaydını metne çevir"""
    if audio is None:
        return ""
    
    try:
        # Ses dosyasını kaydet
        audio_path = os.path.join(audio_folder, "temp_recording.wav")
        
        # Gradio audio'dan gelen tuple: (sample_rate, audio_data)
        import soundfile as sf
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            sf.write(audio_path, audio_data, sample_rate)
        else:
            # Eğer dosya yolu gelirse direkt kullan
            audio_path = audio
        
        # Whisper ile transkripsiyonu al
        result = whisper_model.transcribe(audio_path, language="tr", fp16=False)
        question = result['text'].strip()
        
        return question
    
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def get_sources(question):
    """Soruyla ilgili kaynak dökümanları al"""
    if retriever is None:
        return []
    
    try:
        docs = retriever.get_relevant_documents(question)
        sources = []
        
        for i, doc in enumerate(docs):
            source_info = {
                "index": i + 1,
                "file": os.path.basename(doc.metadata.get("source", "Bilinmeyen")),
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    except:
        return []


def format_sources_html(sources):
    """Kaynakları HTML chip'ler olarak formatla"""
    if not sources:
        return ""
    
    html = '<div style="margin-top: 20px;">'
    html += '<h3 style="font-size: 18px; margin-bottom: 10px; color: #374151;">📚 Kullanılan Kaynaklar:</h3>'
    html += '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">'
    
    for source in sources:
        html += f'''
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            📄 {source["file"]} - Sayfa {source["page"]}
        </div>
        '''
    
    html += '</div>'
    
    # Kaynak detaylarını accordion olarak ekle
    html += '<details style="margin-top: 10px; padding: 15px; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">'
    html += '<summary style="cursor: pointer; font-weight: 600; font-size: 16px; color: #1f2937; margin-bottom: 10px;">🔍 Kaynak Detayları</summary>'
    
    for source in sources:
        html += f'''
        <div style="
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="font-weight: 600; color: #667eea; margin-bottom: 8px; font-size: 15px;">
                Kaynak #{source["index"]}: {source["file"]} (Sayfa {source["page"]})
            </div>
            <div style="color: #4b5563; line-height: 1.6; font-size: 14px;">
                {source["content"]}
            </div>
        </div>
        '''
    
    html += '</details></div>'
    return html


def process_text_query(question):
    """Metin sorusunu işle ve kaynakları göster"""
    if not question or not question.strip():
        return "⚠️ Lütfen bir soru girin!", ""
    
    if rag_chain is None:
        return "⚠️ Lütfen önce PDF dosyaları yükleyin!", ""
    
    try:
        # Cevabı al
        response = rag_chain.invoke(question.strip())
        
        # Kaynakları al
        sources = get_sources(question.strip())
        sources_html = format_sources_html(sources)
        
        return response, sources_html
    except Exception as e:
        return f"❌ Hata: {str(e)}", ""


def upload_pdf(files):
    """PDF dosyalarını yükle ve veritabanını güncelle"""
    global vectorstore, rag_chain
    
    if not files:
        return "⚠️ Lütfen PDF dosyası seçin!"
    
    try:
        uploaded_count = 0
        
        for file in files:
            # Dosyayı data klasörüne kopyala
            import shutil
            file_name = os.path.basename(file.name)
            dest_path = os.path.join(pdf_folder, file_name)
            shutil.copy(file.name, dest_path)
            uploaded_count += 1
        
        # Veritabanını yeniden oluştur
        vectorstore = initialize_vectorstore()
        rag_chain = create_rag_chain()
        
        return f"✅ {uploaded_count} PDF dosyası başarıyla yüklendi ve veritabanı güncellendi!"
    
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def get_pdf_list():
    """Yüklü PDF dosyalarının listesini al"""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        return "📚 Henüz PDF yüklenmedi"
    
    file_list = "\n".join([f"• {os.path.basename(f)}" for f in pdf_files])
    return f"📚 Yüklü PDF Dosyaları ({len(pdf_files)}):\n\n{file_list}"


# Gradio arayüzü
with gr.Blocks(
    title="🦜 Papağan RAG - AI Asistan",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("Inter"),
        text_size=gr.themes.sizes.text_lg,
    ),
    css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 48px !important;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .header p {
            font-size: 20px !important;
            opacity: 0.95;
        }
        /* Daha büyük textbox'lar */
        textarea {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        /* Daha büyük butonlar */
        button {
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
        }
        /* Label'ları büyüt */
        label {
            font-size: 18px !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
        }
        /* Tab'ları büyüt */
        .tab-nav button {
            font-size: 18px !important;
            padding: 12px 20px !important;
        }
        /* Markdown içeriğini büyüt */
        .prose {
            font-size: 16px !important;
        }
        .prose h3 {
            font-size: 20px !important;
        }
    """
) as app:
    
    # Header
    gr.HTML("""
        <div class="header">
            <h1>🦜 Papağan RAG</h1>
            <p>Yapay Zeka Destekli Belge Asistanı - Sesli & Yazılı Soru-Cevap</p>
        </div>
    """)
    
    with gr.Tabs():
        # Ana Sorgu Sekmesi (Birleştirilmiş)
        with gr.Tab("💬 Soru Sor", id="main"):
            with gr.Row():
                # Sol kolon - Girişler
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Sesli veya Yazılı Soru")
                    
                    # Ses kaydı
                    audio_input = gr.Audio(
                        label="Sesli Soru (Mikrofonu kullanın)",
                        type="numpy",
                        sources=["microphone"]
                    )
                    
                    gr.Markdown("**veya**")
                    
                    # Metin girişi
                    text_input = gr.Textbox(
                        label="Yazılı Soru",
                        placeholder="Sorunuzu buraya yazın...",
                        lines=5,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        transcribe_btn = gr.Button(
                            "🎙️ Sesi Metne Çevir",
                            variant="secondary",
                            size="lg",
                            scale=1
                        )
                        submit_btn = gr.Button(
                            "🔍 Sorgula",
                            variant="primary",
                            size="lg",
                            scale=1
                        )
                
                # Sağ kolon - Çıktılar
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Cevap")
                    
                    answer_output = gr.Textbox(
                        label="Cevap",
                        lines=12,
                        max_lines=20,
                        show_copy_button=True
                    )
                    
                    sources_output = gr.HTML(
                        label="Kaynaklar"
                    )
            
            # Ses → Metin çevirme
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=audio_input,
                outputs=text_input
            )
            
            # Sorgulama
            submit_btn.click(
                fn=process_text_query,
                inputs=text_input,
                outputs=[answer_output, sources_output]
            )
            
            # Enter tuşu ile sorgulama
            text_input.submit(
                fn=process_text_query,
                inputs=text_input,
                outputs=[answer_output, sources_output]
            )
            
            gr.Markdown("""
            ---
            ### 📝 Nasıl Kullanılır:
            
            **Sesli Soru için:**
            1. 🎤 Mikrofon simgesine tıklayın
            2. Sorunuzu sesli olarak sorun
            3. Kaydı durdurun
            4. "Sesi Metne Çevir" butonuna tıklayın (metin kutusuna gelecek)
            5. "Sorgula" butonuna tıklayın
            
            **Yazılı Soru için:**
            1. ⌨️ Sorunuzu metin kutusuna yazın
            2. "Sorgula" butonuna tıklayın veya Enter'a basın
            """)
        
        # PDF Yükleme Sekmesi
        with gr.Tab("📁 Belge Yönetimi"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 PDF Yükle")
                    pdf_upload = gr.File(
                        label="PDF Dosyaları Seçin",
                        file_types=[".pdf"],
                        file_count="multiple",
                        height=200
                    )
                    upload_button = gr.Button(
                        "📤 Dosyaları Yükle ve İşle",
                        variant="primary",
                        size="lg"
                    )
                    upload_status = gr.Textbox(
                        label="Yükleme Durumu",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### 📋 Yüklü Dosyalar")
                    pdf_list_button = gr.Button(
                        "� Listeyi Yenile",
                        size="lg"
                    )
                    pdf_list = gr.Textbox(
                        label="Sistemdeki PDF Dosyaları",
                        lines=15,
                        interactive=False,
                        value=get_pdf_list()
                    )
            
            upload_button.click(
                fn=upload_pdf,
                inputs=pdf_upload,
                outputs=upload_status
            ).then(
                fn=get_pdf_list,
                outputs=pdf_list
            )
            
            pdf_list_button.click(
                fn=get_pdf_list,
                outputs=pdf_list
            )
            
            gr.Markdown("""
            ---
            ### ℹ️ Bilgilendirme:
            - 📚 Birden fazla PDF dosyası yükleyebilirsiniz
            - 🔄 Yüklenen dosyalar otomatik olarak sisteme eklenir
            - ⚡ Maksimum 5 PDF dosyası aynı anda işlenir
            - 💾 Dosyalar `data` klasörüne kaydedilir
            """)
    
    # Footer
    gr.Markdown("""
    ---
    ### 🔧 Sistem Bilgisi:
    - **Ses Tanıma:** OpenAI Whisper (Medium)
    - **Dil Modeli:** Llama 3 (8B parametreli)
    - **Embeddings:** BAAI/bge-m3 (Çok dilli)
    - **Vector Database:** ChromaDB
    - **Framework:** LangChain
    
    💡 **İpucu:** Sistemdeki PDF belgelerden en iyi sonuçları almak için açık ve spesifik sorular sorun!
    """)


# Uygulamayı başlat
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )
