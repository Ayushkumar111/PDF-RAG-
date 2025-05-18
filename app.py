import streamlit as st 
import tempfile
import os 
from dotenv import load_dotenv
from helpers.loader import load_file 

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Multi-file AI CHAT", layout = "wide")
st.title("Multi-file AI CHAT")

# Initialize session state for the vector database
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.has_documents = False
    st.session_state.doc_count = 0

uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

# Process uploaded files
if uploaded_files and not st.session_state.has_documents:
    all_docs = []

    with st.spinner("Reading and parsing files..."):
        for uploaded_file in uploaded_files:
            file_name, file_ext = os.path.splitext(uploaded_file.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name 

            try:
                st.write(f"Loading {uploaded_file.name}...")
                docs = load_file(temp_file_path)
                # Debug: Show content of each loaded document
                for doc in docs:
                    st.write(f"Content from {uploaded_file.name} (first 100 chars): {doc.page_content[:100]}...")
                
                all_docs.extend(docs)
                os.unlink(temp_file_path)  # Clean up the temporary file
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                continue

        if all_docs:
            st.write(f"Total documents loaded: {len(all_docs)}")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)
            st.write(f"Total chunks after splitting: {len(chunks)}")

            # Vectorstore
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.vectordb = FAISS.from_documents(chunks, embedding=embeddings)
                st.session_state.has_documents = True
                st.session_state.doc_count = len(chunks)
                st.success(f"Successfully processed {len(all_docs)} documents into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Error creating vector database: {str(e)}")

# Create the retriever and LLM
if st.session_state.has_documents:
    st.write(f"Documents in database: {st.session_state.doc_count}")
    
    # Lower the threshold for better retrieval
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    # Query input
    query = st.text_input("Ask a question based on uploaded documents:")
    
    if query:
        if query.lower() in ["exit", "quit"]:
            st.info("Conversation ended. Refresh the page to start again.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Try a direct search without threshold to see what's closest
                    raw_results = st.session_state.vectordb.similarity_search(query, k=3)
                    st.write("Closest matches (ignoring threshold):")
                    for i, doc in enumerate(raw_results):
                        st.write(f"Match {i+1}: {doc.page_content[:100]}...")
                    
                    # Regular retrieval with threshold
                    relevant_docs = retriever.invoke(query)
                    st.write(f"Found {len(relevant_docs)} relevant documents")
                    
                    if not relevant_docs:
                        st.info("No relevant information found in the documents. Answering from general knowledge.")
                        response = llm.invoke(query)
                        st.markdown(f"**Answer:** {response.content}")
                    else:
                        st.success(f"Found {len(relevant_docs)} relevant document sections.")
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        prompt = f"""
                        Based on the following context, please answer the question. If the context doesn't contain 
                        relevant information to answer the questiBon fully, you can use your general knowledge as well.
                        
                        Context:
                        {context}
                        
                        Question: {query}
                        """
                        
                        response = llm.invoke(prompt)
                        st.markdown(f"**Answer:** {response.content}")
                        
                        # Display sources
                        with st.expander("Sources"):
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Source {i+1}**")
                                if "source" in doc.metadata:
                                    st.write(f"File: {doc.metadata['source']}")
                                st.write(doc.page_content[:300] + "...")
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Error retrieving or processing query: {str(e)}")
elif uploaded_files:
    st.info("Processing files... Please wait.")
else:
    st.info("Please upload some documents to get started.")