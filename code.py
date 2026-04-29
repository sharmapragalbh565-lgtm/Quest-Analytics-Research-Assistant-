
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import WatsonxLLM


import gradio as gr


import os
os.environ["WATSONX_APIKEY"] = "your_api_key_here"
os.environ["WATSONX_PROJECT_ID"] = "your_project_id_here"


llm = WatsonxLLM(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    url="https://us-south.ml.cloud.ibm.com", # or your region's endpoint
    project_id=os.environ["WATSONX_PROJECT_ID"]
)



pdf_path = "2403.05530.pdf" 


loader = PyPDFLoader(pdf_path)


documents = loader.load()


print(f"Number of pages loaded: {len(documents)}")
print("\nSnippet of the first page:")
print(documents[0].page_content[:500] + "...") # Show first 500 characters


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Number of characters per chunk
    chunk_overlap=200  # Overlap between chunks to maintain context
)

docs = text_splitter.split_documents(documents)


print(f"Number of document chunks created: {len(docs)}")
print("\nFirst chunk of text:")
print(docs[0].page_content)



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

sample_text = "This is a sample text to be embedded."
sample_embedding = embedding_model.embed_query(sample_text)

print("Embedding model initialized successfully.")
print(f"Sample embedding vector length: {len(sample_embedding)}")
print(f"First 10 values of the sample embedding: {sample_embedding[:10]}")



persist_directory = './chroma_db'


vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)


vectordb.persist()

print(f"Vector database created and persisted at '{persist_directory}'.")
print(f"Number of vectors in the database: {vectordb._collection.count()}")


retriever = vectordb.as_retriever(search_kwargs={"k": 3})

test_query = "What is RAG?"
retrieved_docs = retriever.get_relevant_documents(test_query)

print("Retriever created successfully.")
print(f"For the query '{test_query}', retrieved {len(retrieved_docs)} document chunks.")
print("\nContent of the first retrieved chunk:")
print(retrieved_docs[0].page_content[:600] + "...") 


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True
)


def answer_question(question):
    result = qa_chain({"query": question})
    answer = result['result']
    
    return answer



iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question about the document here..."),
    outputs="text",
    title="Quest Analytics Research Assistant",
    description="Ask any question about the loaded research paper."
)


iface.launch(share=True)    
