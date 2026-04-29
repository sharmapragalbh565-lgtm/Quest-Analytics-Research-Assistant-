# Quest Analytics Research Assistant (RAG with Watsonx + LangChain)

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about a research paper (PDF) using:

IBM Watsonx LLM (mistralai/mixtral-8x7b-instruct-v01)

LangChain

Chroma Vector Database

HuggingFace Embeddings

Gradio Interface

The system loads a PDF, splits it into chunks, embeds them into a vector database, retrieves relevant content for a query, and generates answers using a large language model.

# Architecture Overview

The pipeline follows a standard RAG workflow:

PDF → Text Splitting → Embeddings → Vector DB (Chroma)
                                    ↓
User Query → Retriever → Context Injection → Watsonx LLM → Answer

# Technologies Used

LangChain – Orchestration framework for LLM pipelines

IBM Watsonx.ai – LLM provider

ChromaDB – Vector store for embeddings

HuggingFace Sentence Transformers – Embedding model (all-mpnet-base-v2)

Gradio – Web UI

# Project Workflow Breakdown
1️. Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

Loads the PDF file.

Each page becomes a separate LangChain document.

2️. Split the Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

Why this is important:

LLMs have context limits.

Chunking improves retrieval precision.

Overlap preserves semantic continuity.

3️. Generate Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

Each text chunk is converted into a dense vector representation.

Embedding model used:

sentence-transformers/all-mpnet-base-v2

Produces 768-dimensional vectors.

4️. Create & Persist Chroma Vector Database
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)
vectordb.persist()

This allows:

Semantic similarity search

Persistent storage across runs

5️. Create Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

Retrieves top 3 most relevant chunks.

Uses cosine similarity in vector space.

6️. Initialize Watsonx LLM
llm = WatsonxLLM(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.environ["WATSONX_PROJECT_ID"]
)

Model used:

Mixtral 8x7B Instruct

Hosted on IBM Watsonx

7️. Build RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

Chain Type:

"stuff" – Injects retrieved chunks directly into the prompt.

8️. Gradio Interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2),
    outputs="text",
)

Creates a simple web app where users can:

Ask questions about the research paper

Get AI-generated answers

# Security Notice

In your current script, replace:

os.environ["WATSONX_APIKEY"] = "your_api_key_here"

with proper environment variable management (e.g., .env file + python-dotenv).

# Key Features

Semantic search over research PDFs

Persistent vector storage

Scalable RAG architecture

IBM Watsonx integration

Web-based Q&A interface

# Possible Improvements

Add source citation display in UI

Use map_reduce chain type for long documents

Add multi-PDF support

Add conversation memory

Deploy using Docker

Add caching for embeddings

Add streaming responses



# What This Project Demonstrates

End-to-end RAG implementation

Practical LLM + Vector DB integration

Document-based question answering

Production-style AI app structure
