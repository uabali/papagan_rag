import gradio as gr
import os
import torch
import numpy as np
import face_recognition
import chromadb
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from uuid import uuid4

# --- YAPILANDIRMA SINIFI ---
class Config:
    # Model Kimlikleri
    # Not: Qwen3 çıktığında burayı güncelleyebilirsiniz. Kod yapısı Qwen2-VL ile aynıdır.
    LLM_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct" 
    # Qwen serisi embedding modeli
    EMBED_MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct" 
    WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
    
    # ChromaDB Ayarları
    CHROMA_PATH = "./chroma_data" # Verilerin kaydedileceği klasör
    COLLECTION_NAME = "secure_rag_collection"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. GÜVENLİK YÖNETİCİSİ (SECURITY) ---
class SecurityManager:
    def __init__(self):
        # Demo amaçlı kullanıcılar. Gerçek hayatta şifreli veritabanında tutulmalı.
        self.users = {
            "admin": {"pass": "123", "role": "admin", "face_enc": None},
            "personel": {"pass": "abc", "role": "user", "face_enc": None}
        }

    def verify(self, username, password, face_image, voice_audio):
        # 1. Kullanıcı ve Şifre Kontrolü
        if username not in self.users:
            return False, "Kullanıcı bulunamadı."
        if self.users[username]["pass"] != password:
            return False, "Hatalı şifre."

        # 2. Yüz Doğrulama (Kayıtlı yüz varsa kontrol et, yoksa uyar ama geliştirme için geç)
        stored_enc = self.users[username]["face_enc"]
        if stored_enc is not None:
            if face_image is None: return False, "Yüz görüntüsü gerekli."
            try:
                # Gelen resimden yüz vektörü çıkar
                face_locs = face_recognition.face_locations(face_image)
                if not face_locs: return False, "Kamerada yüz algılanamadı."
                
                input_enc = face_recognition.face_encodings(face_image, face_locs)[0]
                match = face_recognition.compare_faces([stored_enc], input_enc)[0]
                if not match: return False, "Yüz eşleşmedi! Erişim reddedildi."
            except Exception as e:
                return False, f"Biyometrik hata: {e}"
        
        # 3. Ses Doğrulama (Simülasyon - Dosya var mı?)
        if voice_audio is None:
            return False, "Ses doğrulaması gerekli. Lütfen konuşun."

        return True, self.users[username]["role"]

# --- 2. VERİTABANI YÖNETİCİSİ (CHROMADB) ---
class RAGDatabase:
    def __init__(self):
        print("💾 ChromaDB ve Embedding Modeli Hazırlanıyor...")
        # Embedding Modelini Yükle
        self.embed_model = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL_ID,
            model_kwargs={'device': Config.DEVICE, 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Chroma İstemcisi
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=Config.COLLECTION_NAME)
        
        # Metin Bölücü
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    def add_documents(self, file_paths):
        """Dosyaları okur, böler, vektörleştirir ve kaydeder."""
        if not file_paths: return "Dosya seçilmedi."
        
        count = 0
        for path in file_paths:
            # Dosya tipine göre yükleyici seç
            if path.endswith(".pdf"): loader = PyPDFLoader(path)
            else: loader = TextLoader(path, encoding="utf-8")
            
            docs = loader.load()
            chunks = self.splitter.split_documents(docs)
            
            # Verileri hazırla
            texts = [c.page_content for c in chunks]
            metadatas = [{"source": path} for _ in chunks]
            ids = [str(uuid4()) for _ in chunks]
            
            # Embedding oluştur
            embeddings = self.embed_model.embed_documents(texts)
            
            # Chroma'ya ekle
            self.collection.add(
                ids=ids,
                embeddings=embeddings, # Manuel oluşturduğumuz embeddingleri veriyoruz
                documents=texts,
                metadatas=metadatas
            )
            count += len(chunks)
            
        return f"✅ {len(file_paths)} dosyadan toplam {count} parça veri işlendi ve ChromaDB'ye kaydedildi."

    def search(self, query, k=3):
        """Sorguya en yakın dökümanları getirir."""
        query_vec = self.embed_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k
        )
        # Chroma sonuçları liste içinde liste döner, ilkini alıyoruz
        return results['documents'][0] if results['documents'] else []

# --- 3. ZEKA MOTORU (LLM & WHISPER) ---
class IntelligenceEngine:
    def __init__(self):
        print("🧠 Qwen VL ve Whisper Modelleri Yükleniyor...")
        
        # 1. Whisper (Ses Tanıma)
        self.whisper = pipeline(
            "automatic-speech-recognition",
            model=Config.WHISPER_MODEL_ID,
            device=Config.DEVICE,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32
        )
        
        # 2. Qwen VL (Vision-Language Model)
        # VL modelleri hem resim hem metin anlayabilir.
        self.processor = AutoProcessor.from_pretrained(Config.LLM_MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    def transcribe(self, audio_path):
        """Sesi metne çevirir."""
        if not audio_path: return ""
        result = self.whisper(audio_path)
        return result["text"]

    def generate_response(self, query, context_list):
        """RAG verisi ile cevap üretir."""
        # Context birleştirme
        context_str = "\n\n".join(context_list)
        
        # Prompt Hazırlama
        system_msg = "Sen yardımsever ve güvenli bir asistansın. Aşağıdaki bağlamı (context) kullanarak kullanıcı sorusunu cevapla."
        user_msg = f"BAĞLAM:\n{context_str}\n\nSORU: {query}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Qwen-VL için chat formatı
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Girdileri modele hazırla
        # (Sadece metin gönderdiğimiz için image_inputs=None olabilir, ancak process fonksiyonu bunu halleder)
        inputs = self.processor(
            text=[text_input],
            padding=True,
            return_tensors="pt"
        ).to(Config.DEVICE)
        
        # Üretim
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        # Girdi tokenlarını çıktıdan temizle
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response

# --- 4. ARAYÜZ (GRADIO) ---
class Application:
    def __init__(self):
        self.security = SecurityManager()
        self.db = RAGDatabase()
        self.engine = IntelligenceEngine()

    def login_handler(self, u, p, cam, mic):
        success, result = self.security.verify(u, p, cam, mic)
        
        if not success:
            # Başarısız: Hata mesajı göster, login ekranında kal
            return f"❌ {result}", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
        # Başarılı: Role bak
        role = result
        welcome_msg = f"✅ Hoşgeldin {u} (Yetki: {role})"
        
        # Admin ise upload panelini aç
        is_admin = (role == "admin")
        
        return welcome_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=is_admin)

    def chat_handler(self, message, history, audio_input):
        # 1. Ses girdisi varsa metne ekle/çevir
        if audio_input:
            transcribed = self.engine.transcribe(audio_input)
            if message:
                message = f"{message} (Ses Notu: {transcribed})"
            else:
                message = transcribed
        
        if not message: return "Lütfen konuşun veya yazın."

        # 2. RAG: Chroma'dan veri çek
        relevant_docs = self.db.search(message)
        
        # 3. LLM: Cevapla
        response = self.engine.generate_response(message, relevant_docs)
        
        return response

    def build(self):
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("## 🔐 Güvenli RAG Platformu (ChromaDB + QwenVL)")
            
            # --- LOGIN ---
            with gr.Column(visible=True) as login_view:
                gr.Markdown("### Kimlik Doğrulama")
                with gr.Row():
                    user_box = gr.Textbox(label="Kullanıcı Adı")
                    pass_box = gr.Textbox(label="Şifre", type="password")
                with gr.Row():
                    cam_box = gr.Image(label="Yüz Tanıma", sources=["webcam"], type="numpy")
                    mic_box = gr.Audio(label="Ses İmzası", sources=["microphone"], type="filepath")
                
                login_btn = gr.Button("Sisteme Gir", variant="primary")
                status_lbl = gr.Label(label="Durum")

            # --- ANA EKRAN ---
            with gr.Column(visible=False) as main_view:
                with gr.Row():
                    gr.Markdown("### 🤖 Asistan")
                    logout_btn = gr.Button("Çıkış Yap", size="sm", variant="stop")

                # Admin Paneli (Varsayılan Gizli)
                with gr.Group(visible=False) as admin_panel:
                    gr.Markdown("#### 📁 Veri Yükleme Paneli (Admin)")
                    file_uploader = gr.File(file_count="multiple", label="PDF veya TXT Yükle")
                    process_btn = gr.Button("Dosyaları İşle ve ChromaDB'ye Kaydet")
                    upload_log = gr.Textbox(interactive=False, label="Log")
                    
                    process_btn.click(
                        self.db.add_documents, 
                        inputs=[file_uploader], 
                        outputs=[upload_log]
                    )

                # Chat Arayüzü
                chatbot = gr.ChatInterface(
                    fn=self.chat_handler,
                    additional_inputs=[
                        gr.Audio(sources=["microphone"], type="filepath", label="Sesli Sor")
                    ],
                    title="Qwen AI Chat",
                    description="Dosyalardan sorumlu olduğum konularda bana soru sorabilirsiniz."
                )

            # --- ETKİLEŞİMLER ---
            login_btn.click(
                self.login_handler,
                inputs=[user_box, pass_box, cam_box, mic_box],
                outputs=[status_lbl, login_view, main_view, admin_panel]
            )
            
            logout_btn.click(
                lambda: (gr.update(visible=True), gr.update(visible=False)),
                None,
                [login_view, main_view]
            )
            
        return app

if __name__ == "__main__":
    # ChromaDB eski versiyon uyumluluğu için gerekirse:
    # __import__('pysqlite3')
    # import sys
    # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    platform = Application()
    ui = platform.build()
    ui.launch()