# Mini Document Q&A System

Upload any PDF and ask questions about it in plain English. Answers are grounded in the document and cite the source chunk used.

## Tech Stack
- **LLM:** Groq API (Llama 3.3)
- **Search:** TF-IDF (scikit-learn)
- **PDF Parsing:** pypdf

## Setup

```bash
git clone https://github.com/AK-1809-TRIGGER/doc-qa.git
cd doc-qa
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get your free key at https://console.groq.com

## Run

```bash
python app.py
```

Enter your PDF path and start asking questions!

## How It Works

1. Extracts text from PDF
2. Splits into overlapping chunks
3. Builds a TF-IDF search index
4. Finds relevant chunks for your question
5. Sends context + question to Groq LLM for a cited answer
