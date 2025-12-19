# Papagan RAG - Talk to Your PDFs

A smart local AI that reads your PDFs and answers questions about them. Nothing leaves your computer - it's all private and runs offline.

## What Does It Do?

Ever wanted to ask questions about your massive PDF collection without reading through everything? That's exactly what this does. 

- Drop your PDFs in the `data/` folder
- Ask questions in **text** or **voice** (yep, you can just talk to it!)
- Get answers in Turkish based only on what's actually in your documents
- If it's not in the PDFs, it'll tell you straight up instead of making stuff up

**New:** Voice input powered by OpenAI Whisper - just speak your question instead of typing!

## How It Works

![RAG Architecture](https://github.com/It-Does-Not-Actually-Matter/papagan_rag/raw/main/rag.jpg)

The system breaks down into 5 steps:
1. **Load** - Reads your PDF files
2. **Split** - Chops documents into bite-sized chunks
3. **Store** - Saves them in a vector database
4. **Search** - Finds the most relevant chunks for your question
5. **Answer** - Uses an LLM to generate a response in Turkish

## What You Need

### Ollama (Must Have)

This project uses Ollama for the language model. Before running anything:

```bash
ollama pull llama3:8b
```

Make sure Ollama is running in the background.

### Python Stuff

Create a virtual environment and activate it:

```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On Mac/Linux:
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** First run will download the Whisper model (~140MB) and the embedding model (~2GB). Grab some coffee.

## Setup

1. Clone this repo:
```bash
git clone https://github.com/It-Does-Not-Actually-Matter/papagan_rag.git
cd papagan_rag
```

2. Create a `data/` folder and throw your PDFs in there:
```bash
mkdir data
# Copy your PDF files into the data/ folder
```

**Limits:**
- Max 50 PDFs at once
- Max 200 PDFs total
- You'll get clear error messages if you hit these

3. Set up Python environment:
```bash
python -m venv env
env\Scripts\activate  # Windows
pip install -r requirements.txt
```

4. Verify Ollama is running:
```bash
ollama list
```

## How to Use

### Command Line Mode (main.py)

Just run:

```bash
python main.py
```

You'll see:

```
PAPAGAN

Type text or 'v' for voice (q to quit): 
```

**Text Input:** Type your question and hit Enter
```
Type text or 'v' for voice (q to quit): What is software design?
Papagan: [Streams answer based on your PDFs...]
```

**Voice Input:** Type `v`, then:
1. Press Enter to start recording
2. Speak your question
3. Press Enter to stop
4. It'll transcribe and answer

```
Type text or 'v' for voice (q to quit): v
Press Enter to start recording...
Recording... Press Enter to stop.
Transcribed: [Your question in text]
Papagan: [Answer...]
```

Type `q` to quit.

### Gradio Web UI (gradio_app.py)

Want a prettier interface?

```bash
python gradio_app.py
```

Opens a chat interface in your browser at `http://localhost:7860`. Type your questions and get responses with a nice UI.

## Tech Stack

| Component | What We Use |
|-----------|-------------|
| **Embedding Model** | BAAI/bge-m3 (multilingual) |
| **LLM** | Llama3 8B via Ollama |
| **Vector Database** | Chroma |
| **Voice Recognition** | OpenAI Whisper (base model) |
| **Chunk Size** | 800 characters |
| **Chunk Overlap** | 120 characters |
| **Similarity Search** | Top 5 matches |
| **Temperature** | 0.1 (focused answers) |


## Troubleshooting

**Whisper model not loading?**  
First run downloads ~140MB. Check your internet connection.

**No microphone detected?**  
Make sure your mic is plugged in and permissions are granted.

**Out of memory?**  
Try using the `tiny` or `small` Whisper model instead of `base`. Edit line 161 in `main.py`:
```python
WHISPER_MODEL = whisper.load_model("small")  # or "tiny"
```

## License

MIT License - do whatever you want with it.

---