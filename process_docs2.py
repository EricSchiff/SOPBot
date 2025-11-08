# process_docs.py
import os
import glob
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# Paths
DATA_PATH = "data"
DB_PATH = "embeddings_store"

# Embedding model
embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (common pattern)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)
    return text.strip()

def enhance_metadata(doc: Document, pdf_path: str) -> Document:
    """Add rich metadata for better retrieval and citation"""
    filename = os.path.basename(pdf_path)
    
    # Extract SOP ID or title from filename if possible
    sop_id = filename.replace('.pdf', '').replace('_', ' ')
    
    doc.metadata.update({
        'source': filename,
        'sop_id': sop_id,
        'page': doc.metadata.get('page', 0),
        'file_path': pdf_path,
        # Add character count for quality filtering
        'char_count': len(doc.page_content)
    })
    
    # Try to extract section headers from content
    lines = doc.page_content.split('\n')
    for line in lines[:5]:  # Check first few lines
        if line.isupper() or (line.strip() and len(line.strip()) < 100):
            doc.metadata['section_hint'] = line.strip()
            break
    
    return doc

def filter_low_quality_chunks(docs: List[Document]) -> List[Document]:
    """Remove chunks that are too short or likely contain only headers/footers"""
    filtered = []
    for doc in docs:
        # Skip very short chunks (likely headers/footers)
        if len(doc.page_content.strip()) < 100:
            continue
        
        # Skip chunks that are mostly numbers (likely page numbers or tables)
        alpha_ratio = sum(c.isalpha() for c in doc.page_content) / max(len(doc.page_content), 1)
        if alpha_ratio < 0.3:
            continue
            
        filtered.append(doc)
    
    return filtered

def create_custom_splitter():
    """Create splitter optimized for SOP documents"""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks to maintain procedure context
        chunk_overlap=200,  # More overlap to preserve step sequences
        length_function=len,
        separators=[
            "\n\n\n",  # Multiple line breaks (section separators)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            " ",       # Word breaks
            ""
        ],
        # Keep procedures and numbered lists together when possible
        is_separator_regex=False,
    )

def embed_docs():
    all_docs = []
    processed_files = 0
    
    # Load all PDFs
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {DATA_PATH}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        try:
            print(f"📄 Processing: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Clean and enhance each document
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc = enhance_metadata(doc, pdf_path)
                
                # Only include pages with substantial content
                if len(doc.page_content.strip()) > 50:
                    all_docs.append(doc)
            
            processed_files += 1
            print(f"   ✓ Loaded {len(docs)} pages")
            
        except Exception as e:
            print(f"   ✗ Error processing {pdf_path}: {str(e)}")
            continue
    
    if not all_docs:
        print("⚠️  No valid documents were loaded")
        return
    
    print(f"\n📊 Total pages loaded: {len(all_docs)}")
    
    # Split into chunks with custom splitter
    splitter = create_custom_splitter()
    split_docs = splitter.split_documents(all_docs)
    
    print(f"✂️  Split into {len(split_docs)} chunks")
    
    # Filter low-quality chunks
    split_docs = filter_low_quality_chunks(split_docs)
    
    print(f"🔍 After quality filtering: {len(split_docs)} chunks")
    
    # Create or update vector store
    if os.path.exists(DB_PATH):
        print(f"🗑️  Removing existing database at {DB_PATH}")
        import shutil
        shutil.rmtree(DB_PATH)
    
    # Store in ChromaDB with metadata
    vectorstore = Chroma.from_documents(
        split_docs,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"\n✅ Successfully embedded {len(split_docs)} chunks from {processed_files} PDFs")
    print(f"💾 Vector store saved to: {DB_PATH}")
    
    # Print sample metadata
    if split_docs:
        print(f"\n📋 Sample chunk metadata:")
        sample = split_docs[0]
        for key, value in sample.metadata.items():
            print(f"   {key}: {value}")

def verify_embeddings():
    """Quick verification that embeddings were created successfully"""
    if not os.path.exists(DB_PATH):
        print("❌ No embeddings database found")
        return False
    
    try:
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model
        )
        collection = vectorstore._collection
        count = collection.count()
        print(f"✅ Verification: {count} embeddings in database")
        return True
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting SOP document processing...\n")
    embed_docs()
    print("\n🔍 Verifying embeddings...")
    verify_embeddings()