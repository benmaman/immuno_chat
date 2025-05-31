import fitz  # PyMuPDF
import textwrap
from openai import OpenAI
import time
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv()

def extract_book_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_num, "text": text})
    return pages



def chunk_page(page_dict, max_chars=2500):
    text = page_dict["text"]
    page_num = page_dict["page"]
    chunks = textwrap.wrap(text, max_chars)
    return [{"text": chunk, "page": page_num} for chunk in chunks]


#step 1 : extract text from the book
book_pages = extract_book_pages("Janeway-s-Immunobiology_compressed.pdf")

#step 2 : chunk the text into manageable pieces
all_chunks = []
for page in book_pages:
    all_chunks.extend(chunk_page(page))


# Preview first few chunks
for c in all_chunks[:3]:
    print(f"Page {c['page']}:\n{c['text']}\n")
# step 3 — Create Embeddings for Each Chunk

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model='text-embedding-3-small'


#now we will do for 100 chuncks:


def get_embedding(text, model="text-embedding-3-small", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
            time.sleep(2)
    raise RuntimeError("Failed to get embedding after retries.")

# Apply to all chunks
all_embeddings = []
for chunk in all_chunks:
    embedding = get_embedding(chunk["text"],model)
    all_embeddings.append(embedding)

metadata = [{"text": chunk["text"], "page": chunk["page"]} for chunk in all_chunks]
metadata


## step 4 


# Step 41: Convert your embeddings to a float32 NumPy array
embedding_dim = len(all_embeddings[0])
embedding_matrix = np.array(all_embeddings).astype("float32")

# Step 2: Create the FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean distance
index.add(embedding_matrix)

# Step 3: Store index + metadata together for use later
print(f"✅ FAISS index created with {index.ntotal} vectors.")


# Step 5: Define the function to ask questions


def ask_question(question, k=10):
    # Step 1: Embed the question
    q_embed = client.embeddings.create(input=[question], model="text-embedding-3-small")
    query_vector = np.array(q_embed.data[0].embedding).astype("float32")

    # Step 2: Retrieve top-k chunks
    D, I = index.search(np.array([query_vector]), k)
    retrieved_chunks = [metadata[i] for i in I[0]]

    # Step 3: Build prompt with citations
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[{i+1}] (Page {chunk['page']}): {chunk['text'][:500].strip()}...\n\n"

    prompt = f"""You are an expert assistant. Use only the context below to answer the question.
                Cite the relevant page numbers. If you don't know, say you don't know.

                Context:
                {context}
                Question: {question}
                Answer:"""

    # Step 4: Ask GPT-4
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


answer = ask_question("what happen when t cell bind to epitope? how t cell expansion happen?")
print("\n🧠 GPT Answer:\n")
print(answer)


#save files
faiss.write_index(index, "book_index.faiss")

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)



