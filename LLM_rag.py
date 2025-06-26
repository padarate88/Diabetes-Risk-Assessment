import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate

# === OpenAI client (SDK ≥ 1.0.0) ===
client = OpenAI(api_key="sk-")  # Replace with your actual API key

# === Path config ===
index_path = r"C:\Users\OEM\Desktop\NZ_medical_data\medical_faiss.index"
id2text_path = r"C:\Users\OEM\Desktop\NZ_medical_data\id2text.txt"

# === Load FAISS ID → text mapping ===
def load_id2text(path):
    id2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                idx, text = line.strip().split("\t", 1)
                id2text[int(idx)] = text
            except:
                continue
    return id2text

# === FAISS search class ===
class DiabetesSearcher:
    def __init__(self, index_path, id2text_path):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        self.id2text = load_id2text(id2text_path)

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            text = self.id2text.get(idx, "<Not Found>")
            dist = distances[0][i]
            results.append((i + 1, text, dist))
        return results

# === Prompt template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional medical assistant. You can use the provided context to help answer the user's question, but you are not limited to it. Feel free to incorporate general medical knowledge where appropriate.

[Question]
{question}

[Context]
{context}
"""
)

# === Deduplication ===
def deduplicate_sentences(text):
    seen = set()
    result = []
    for sentence in text.split('. '):
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    return '. '.join(result)

# === Build RAG prompt ===
def build_prompt_from_query(query, searcher, top_k=5):
    results = searcher.search(query, top_k=top_k)
    context = "\n".join([f"{i + 1}. {text}" for i, (_, text, _) in enumerate(results)])
    return prompt_template.format(context=context, question=query)

# === Generate GPT answer ===
def rag_answer(query, searcher, model="gpt-4o"):
    try:
        prompt = build_prompt_from_query(query, searcher)
        print("\n📋 Prompt sent to GPT:\n" + prompt)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and professional medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=500
        )

        reply = response.choices[0].message.content.strip()
        reply = deduplicate_sentences(reply)
        return reply

    except Exception as e:
        return f"❌ Error during GPT generation: {e}"

# === CLI entrypoint ===
if __name__ == "__main__":
    print("🚀 RAG Medical QA System (OpenAI GPT version) is running.")

    searcher = DiabetesSearcher(index_path, id2text_path)

    user_input = input("\n❓ Enter your medical question:\n> ").strip()

    if user_input:
        answer = rag_answer(user_input, searcher)
        print("\n✅ Answer:\n" + answer)
    else:
        print("\n⚠️ No input received. Exiting.")
