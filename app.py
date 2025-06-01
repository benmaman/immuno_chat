import streamlit as st
import faiss
import numpy as np
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Load FAISS index and metadata
index = faiss.read_index("book_index.faiss")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to embed a question
def get_question_embedding(question, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[question], model=model)
    return np.array(response.data[0].embedding).astype("float32")

# Function to query FAISS and ask GPT-4
@st.cache_data(show_spinner=False)
def retrieve_context(question, k=10):
    query_vector = get_question_embedding(question)
    D, I = index.search(np.array([query_vector]), k)
    retrieved = [metadata[i] for i in I[0]]
    context = ""
    for i, item in enumerate(retrieved):
        context += f"[{i+1}] (Page {item['page']}): {item['text'][:500].strip()}\n\n"
    return context, retrieved

def ask_with_context(history, question, context):
    history_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

    full_prompt = f"""
    You are an expert tutor. Use only the context below to answer the question.
    please try to summmerize the context and answer the question in a concise manner.
    answer like yo u desribe it for Msc or phd student of immunology.
    Always cite the relevant page numbers. If the answer is not in the context, say shortly that you done know.

    Context:
    {context}

    Conversation:
    {history_prompt}

    User: {question}
    Assistant:
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# history=''
# question="how t cell undergo clonal expansion?'"
# # Get the embedding for the question
# context= retrieve_context(question,5)
# # Search the FAISS index for the top 10 most similar chunks
# ask_with_context(history,question,context)




# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="📚 Book Chatbot", layout="wide")
st.title("📖 Chat with Your Janeway's Immunobiology Book")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.info("Enter your OpenAI API key in .streamlit/secrets.toml")
    show_chunks = st.checkbox("🔍 Show retrieved chunks", value=False)
    st.markdown("""
    **Usage tips:**
    - Ask factual or technical questions from the book
    - The model will only answer from your textbook
    - Citations to pages included
    """)

question = st.chat_input("Ask a question from the book...")

if question:
    with st.spinner("Thinking..."):
        context, retrieved = retrieve_context(question)
        answer = ask_with_context(st.session_state.chat_history, question, context)

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")

        if show_chunks:
            with st.expander("📚 Retrieved Chunks and Pages"):
                for i, item in enumerate(retrieved):
                    st.markdown(f"**Chunk {i+1}** (Page {item['page']}):\n{item['text'][:700].strip()}")

# Optional: display conversation history
st.divider()
with st.expander("🗂️ Chat History"):
    for entry in st.session_state.chat_history:
        role = entry['role'].capitalize()
        st.markdown(f"**{role}:** {entry['content']}")
