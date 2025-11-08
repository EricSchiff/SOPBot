# app.py
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

# ========== CONFIG ==========
DB_PATH = "embeddings_store"
OPENAI_MODEL = "gpt-4o-mini"
NUM_CHUNKS = 6
USE_MMR = True

# ========== SECRETS HANDLING (Cloud + Local) ==========
def get_api_key():
    """Get API key from Streamlit secrets (cloud) or .env (local)"""
    if "OPENAI_API_KEY" in st.secrets:
        # Running on Streamlit Cloud
        return st.secrets["OPENAI_API_KEY"]
    else:
        # Running locally
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OPENAI_API_KEY not found. Please set it in .env file locally or in Streamlit secrets.")
                st.stop()
            return api_key
        except ImportError:
            st.error("❌ python-dotenv not installed. Run: pip install python-dotenv")
            st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=get_api_key())
except Exception as e:
    st.error(f"❌ Error initializing OpenAI client: {str(e)}")
    st.stop()

# ========== LOAD VECTOR STORE ==========
@st.cache_resource
def load_vectorstore():
    """Load vector store with caching for performance"""
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Vector store not found at {DB_PATH}. Please run process_docs.py first.")
        st.stop()
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

db = load_vectorstore()

# ========== HELPER FUNCTIONS ==========
def format_context_with_metadata(documents):
    """Format retrieved documents with rich metadata for LLM context"""
    context_parts = []
    sources_display = []
    
    for idx, doc in enumerate(documents, 1):
        # Extract all available metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        sop_id = doc.metadata.get("sop_id", source.replace('.pdf', ''))
        section = doc.metadata.get("section_hint", "")
        
        # Use SOP ID as the primary reference (cleaner than filename)
        sop_title = sop_id if sop_id != "Unknown" else source
        
        # Format for LLM - reference by SOP title instead of excerpt number
        context_entry = f"""[{sop_title} - Page {page}]
{f"Section: {section}" if section else ""}

Content:
{doc.page_content}
"""
        context_parts.append(context_entry)
        
        # Format for sidebar display
        sources_display.append({
            'num': idx,
            'source': source,
            'page': page,
            'sop_id': sop_id,
            'sop_title': sop_title,
            'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })
    
    return "\n\n".join(context_parts), sources_display


def retrieve_documents(query, k=NUM_CHUNKS, use_mmr=USE_MMR):
    """Retrieve relevant documents using similarity search or MMR"""
    try:
        if use_mmr:
            # MMR provides diversity in results
            documents = db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k * 3  # Fetch more candidates for better selection
            )
        else:
            documents = db.similarity_search(query, k=k)
        
        return documents
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []


def create_system_prompt():
    """Enhanced system prompt for forensic DNA analyst assistant"""
    return """You are a specialized assistant for forensic DNA analysts, helping them find and understand information from Standard Operating Procedures (SOPs).

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided SOP excerpts
2. ALWAYS cite the SOP title and page number (e.g., "According to the DNA Extraction SOP, page 5..." or "As stated in Sample Handling Protocol, page 3...")
3. Include direct quotes from the SOPs when relevant
4. If the excerpts don't contain the answer, clearly state: "I cannot find this information in the provided SOP excerpts"
5. Maintain technical accuracy and use the exact terminology from the SOPs
6. If procedures have numbered steps, preserve the numbering and order
7. Highlight any safety warnings, cautions, or critical steps
8. If multiple SOPs contain relevant information, mention all sources with their page numbers

FORMAT YOUR RESPONSE:
- Start with a direct answer
- Include relevant quotes with citations (SOP title and page number)
- End with source summary (which SOPs and pages you referenced)"""


def create_user_prompt(question, context):
    """Create the user message with question and context"""
    return f"""ANALYST QUESTION:
{question}

RELEVANT SOP SECTIONS:
{context}

Please answer the question using ONLY the information in the SOP sections above. Remember to cite the SOP title and page number for each reference."""


def get_chat_history_context(max_turns=2):
    """Get recent conversation history for context (last N turns)"""
    if len(st.session_state.messages) <= 1:
        return ""
    
    # Get last N Q&A pairs (excluding current question)
    recent_messages = st.session_state.messages[-(max_turns*2):]
    
    history_parts = ["RECENT CONVERSATION CONTEXT:"]
    for msg in recent_messages:
        role = "Analyst" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content'][:200]}...")  # Truncate long messages
    
    return "\n".join(history_parts) + "\n\n"

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="SOP Chatbot",
    page_icon="📄",
    layout="wide"
)

# Main title and description
st.title("🤖 SOP Chatbot")
st.markdown("*Ask questions about your Standard Operating Procedures*")

# Sidebar for settings and sources
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Retrieval settings
    num_chunks = st.slider("Number of SOP sections to retrieve", 3, 10, NUM_CHUNKS)
    use_mmr = st.checkbox("Use MMR (diverse results)", value=USE_MMR)
    include_history = st.checkbox("Include conversation context", value=False)
    
    st.divider()
    
    # Display sources section (will be populated after query)
    st.header("📚 Retrieved SOPs")
    sources_container = st.container()
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ========== CHAT INTERFACE ==========
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========== USER INPUT & PROCESSING ==========
if prompt := st.chat_input("Ask a question about the SOPs..."):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show processing indicator
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching SOPs..."):
            # 1) Retrieve relevant documents
            documents = retrieve_documents(prompt, k=num_chunks, use_mmr=use_mmr)
            
            if not documents:
                error_msg = "❌ No relevant SOP information found. Please check your question or vector store."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()
            
            # 2) Format context with metadata
            context, sources_display = format_context_with_metadata(documents)
            
            # Display sources in sidebar
            with sources_container:
                for source in sources_display:
                    with st.expander(f"📄 {source['sop_title']} (Page {source['page']})"):
                        st.write(f"**File:** {source['source']}")
                        st.write(f"**Page:** {source['page']}")
                        st.write(f"**Preview:**")
                        st.text(source['preview'])
        
        with st.spinner("🤖 Generating answer..."):
            try:
                # 3) Build messages for OpenAI
                system_msg = create_system_prompt()
                
                # Optionally include conversation history
                history_context = ""
                if include_history:
                    history_context = get_chat_history_context(max_turns=2)
                
                user_msg = history_context + create_user_prompt(prompt, context)
                
                # 4) Call OpenAI
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=1000
                )
                
                answer = response.choices[0].message.content
                
                # 5) Display and store assistant response
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Optional: Show token usage in expander
                with st.expander("ℹ️ Response Details"):
                    st.write(f"**Model:** {OPENAI_MODEL}")
                    st.write(f"**SOP Sections Retrieved:** {len(documents)}")
                    st.write(f"**Tokens Used:** {response.usage.total_tokens}")
                    st.write(f"**Search Method:** {'MMR (Diverse)' if use_mmr else 'Similarity'}")
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ========== FOOTER ==========
st.divider()
st.caption("💡 Tip: Ask specific questions about procedures, safety protocols, equipment, or analysis steps for best results.")
st.caption("⚠️ Always verify critical information with the original SOP documents.")