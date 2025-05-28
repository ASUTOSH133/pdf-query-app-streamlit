import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io
import torch
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="📄 PDF Query Application")

# --- Load Models Efficiently ---
@st.cache_resource
def load_models():
    """Load embeddings and a lightweight generative model"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")  # Faster model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cuda" if torch.cuda.is_available() else "cpu")
    return embedding_model, tokenizer, model

embedding_model, llm_tokenizer, llm_model = load_models()

# --- PDF Processing ---
def extract_text(uploaded_file):
    """Extract text from PDF"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"PDF error: {e}")
        return ""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text):
    """Ensures multiple meaningful chunks are created."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Creates enough context for accurate answers
        chunk_overlap=100,  # Adds overlap for consistency
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return text_splitter.split_text(text)

# # --- Smart Chunking ---
# def split_text(text):
#     """Ensures multiple meaningful chunks are created."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,  # Makes chunks big enough for context-rich retrieval
#         chunk_overlap=100,  # Ensures continuity between sections
#         separators=["\n\n", "\n", ".", "?", "!", " ", ""]
#     )
#     return text_splitter.split_text(text)

# --- Retrieve Relevant Chunks ---
# import random
# def retrieve_chunks(query, chunks, embeddings, top_k=3):
#     """Retrieve the top_k most relevant chunks, introducing variety."""
#     if not chunks or embeddings is None:
#         return "No document content available."

#     query_embedding = embedding_model.encode(query, convert_to_tensor=True)
#     scores = util.cos_sim(query_embedding, embeddings)[0]
#     top_indices = np.argsort(scores.cpu().numpy())[::-1][:top_k]

#     # Introduce variation by shuffling ranked indices slightly
#     mixed_indices = random.sample(list(top_indices), min(top_k, len(top_indices)))
#     relevant_chunks = "\n\n".join([chunks[i] for i in mixed_indices])
    
#     return relevant_chunks

# from rank_bm25 import BM25Okapi

# def retrieve_chunks(query, chunks, embeddings, top_k=3):
#     """Retrieve chunks using both BM25 keyword matching and embedding similarity."""
#     if not chunks or embeddings is None:
#         return "No document content available."

#     # Tokenize the chunks for BM25 ranking
#     tokenized_chunks = [chunk.split() for chunk in chunks]
#     bm25 = BM25Okapi(tokenized_chunks)
#     bm25_scores = bm25.get_scores(query.split())

#     # Compute similarity scores using embeddings
#     query_embedding = embedding_model.encode(query, convert_to_tensor=True)
#     cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

#     # Combine scores (weighted average)
#     final_scores = (0.7 * np.array(cosine_scores.cpu().numpy())) + (0.3 * np.array(bm25_scores))
#     top_indices = np.argsort(final_scores)[::-1][:top_k]

#     relevant_chunks = "\n\n".join([chunks[i] for i in top_indices])
#     return relevant_chunks

from rank_bm25 import BM25Okapi

def retrieve_chunks(query, chunks, embeddings, top_k=3):
    """Retrieve the most relevant chunks using hybrid search while filtering bad results."""
    if not chunks or embeddings is None:
        return "No document content available."

    # BM25 keyword-based ranking
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.split())

    # Embedding similarity ranking
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Weighted scoring (BM25 + embeddings)
    final_scores = (0.7 * np.array(cosine_scores.cpu().numpy())) + (0.3 * np.array(bm25_scores))
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    # Filter out weak matches (avoid pulling non-related chunks)
    relevant_chunks = [chunks[i] for i in top_indices if final_scores[i] > 0.5]  # Threshold filtering
    if not relevant_chunks:
        return "No relevant data found in document."

    return "\n\n".join(relevant_chunks)

def generate_summary(pdf_chunks):
    """Generate a detailed summary of the document using structured AI prompting."""
    if not pdf_chunks or len(pdf_chunks) < 3:
        return "Not enough document data available to generate a proper summary."

    # Combine top chunks for more context
    summary_context = "\n\n".join(pdf_chunks[:5])  # Using first few chunks for broader context

    prompt = (
        f"You are an AI assistant that summarizes research papers accurately.\n\n"
        f"Context:\n{summary_context}\n\n"
        f"Task: Provide a detailed summary of this document.\n\n"
        f"Instructions:\n"
        f"- Capture the key findings, concepts, and takeaways.\n"
        f"- Maintain logical flow and clarity.\n"
        f"- Avoid direct copying—rewrite in a clear and concise way.\n\n"
        f"Summary:"
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(inputs.input_ids, max_new_tokens=300, num_beams=6)  # Increased token size

    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Generate AI Answer ---
def generate_answer(question, context):
    """Generate accurate answers strictly based on retrieved PDF content."""
    if "No relevant data found" in context:
        return "No relevant data available in the document."

    prompt = (
        f"Using the provided document excerpts, answer the question based only on its contents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a precise and structured answer using document references.\n\n"
        f"Answer:"
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(inputs.input_ids, max_new_tokens=256, num_beams=4)

    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)



# def generate_answer(question, context):
#     """Generate answers with more relevance based on structured context."""
#     prompt = (
#         f"You are an AI assistant that answers questions based strictly on the given PDF content.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {question}\n\n"
#         f"Instructions:\n"
#         f"- Answer **only** using information from the provided context.\n"
#         f"- If the context lacks relevant information, say 'No relevant data found in document.'\n"
#         f"- Provide **clear and structured answers**.\n\n"
#         f"Answer:"
#     )

#     inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(llm_model.device)
#     outputs = llm_model.generate(inputs.input_ids, max_new_tokens=256, num_beams=4)

#     return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


# # --- Generate AI Answer ---
# def generate_answer(question, context):
#     """Use LLM to generate unique answers based on distinct questions."""
#     prompt = (
#         f"Given the following document excerpts, provide a detailed and contextually relevant answer.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {question}\n\n"
#         f"Instructions:\n"
#         f"- Ensure the response is **unique** and directly answers the query.\n"
#         f"- Avoid generic statements. Structure the response **logically**.\n"
#         f"- If needed, reference specific details from the extracted text.\n\n"
#         f"Answer:"
#     )

#     inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(llm_model.device)
#     outputs = llm_model.generate(inputs.input_ids, max_new_tokens=256, num_beams=4)

#     return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- UI Enhancements ---
st.title("📄 PDF Document Query Application")

# Sidebar: Upload & Display PDF Info
st.sidebar.header("Upload PDF")
file = st.sidebar.file_uploader("Select a PDF file", type="pdf")

if file:
    file_info = {"name": file.name, "size": file.size}
    st.sidebar.info(f"**Uploaded:** {file_info['name']} ({file_info['size']/1024:.2f} KB)")
    
    # Extract text & process into chunks
    pdf_text = extract_text(file.getvalue())
    chunks = split_text(pdf_text)  # Process text into multiple chunks
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

    st.sidebar.success(f"PDF processed into {len(chunks)} chunks.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.update({
        "pdf_content": pdf_text,
        "pdf_chunks": chunks,
        "pdf_embeddings": embeddings,
    })

# Main Chat UI
if "pdf_content" in st.session_state:
    st.header("📌 Chat Interface")

    # Display Previous Chats (Stores Last 5 Messages)
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for chat in st.session_state.chat_history[-5:]:
            with st.chat_message("user"):
                st.markdown(f"**🗨️ Question:** {chat['user']}")
            with st.chat_message("ai"):
                st.markdown(f"💡 **Answer:** {chat['ai']}")

query = st.chat_input("Ask about the document...")

if query:
    if "summarize" in query.lower():  # If user asks for a summary
        answer = generate_summary(st.session_state["pdf_chunks"])
    else:  # Otherwise, process normally
        context = retrieve_chunks(query, st.session_state["pdf_chunks"], st.session_state["pdf_embeddings"])
        answer = generate_answer(query, context)

    # Display Current Chat
    with st.chat_message("user"):
        st.markdown(f"**🗨️ Question:** {query}")
    with st.chat_message("ai"):
        st.markdown(f"💡 **Answer:** {answer}")

    # Store Chat History (Keep last 5 messages)
    st.session_state.chat_history.append({"user": query, "ai": answer})


    # # User Query Input
    # query = st.chat_input("Ask about the document...")

    # if query:
    #     context = retrieve_chunks(query, st.session_state["pdf_chunks"], st.session_state["pdf_embeddings"])
    #     answer = generate_answer(query, context)

    #     # Display Current Chat
    #     with st.chat_message("user"):
    #         st.markdown(f"**🗨️ Question:** {query}")
    #     with st.chat_message("ai"):
    #         st.markdown(f"💡 **Answer:** {answer}")

    #     # Store Chat History (Keep last 5 messages)
    #     st.session_state.chat_history.append({"user": query, "ai": answer})
