import os
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix

def search_chunks(query, chunks, vectorizer, matrix, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], round(float(scores[i]), 3)) for i in top_indices]

def answer_question(question, relevant_chunks):
    context = "\n\n".join(
        [f"[Source chunk {i+1}]:\n{chunk}" for i, (chunk, _) in enumerate(relevant_chunks)]
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I could not find this in the document."
Always mention which source chunk(s) you used.

Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    print("=" * 50)
    print("   Mini Document Q&A System (Powered by Gemini)")
    print("=" * 50)

    file_path = input("\nEnter path to your PDF or .txt file: ").strip()
    if not os.path.exists(file_path):
        print("File not found.")
        return

    print("\n📄 Extracting text...")
    text = extract_text(file_path)
    print(f"✅ Extracted {len(text):,} characters")

    print("✂️  Splitting into chunks...")
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("📊 Building search index...")
    vectorizer, matrix = build_index(chunks)
    print("✅ Index ready!\n")

    print("Ask questions about your document. Type 'quit' to exit.\n")

    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        relevant_chunks = search_chunks(question, chunks, vectorizer, matrix)
        answer = answer_question(question, relevant_chunks)
        print(f"\n💡 Answer:\n{answer}\n" + "-"*50)

if __name__ == "__main__":
    main()