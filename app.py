import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import pypdf
import tempfile

load_dotenv()

st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")
st.title("Enterprise GenAI Knowledge Assistant")
st.caption("RAG-powered document Q&A with citation-based responses")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

#upload documents sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name

                    try:
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()
                        all_documents.extend(documents)
                    except Exception:
                        #strict=false
                        try:
                            reader = pypdf.PdfReader(temp_path, strict=False)
                            for page_num, page in enumerate(reader.pages):
                                try:
                                    text = page.extract_text() or ""
                                    if text.strip():
                                        all_documents.append(Document(
                                            page_content=text,
                                            metadata={"source": uploaded_file.name, "page": page_num}
                                        ))
                                except Exception:
                                    continue
                        except Exception as fallback_err:
                            st.warning(f"Could not read '{uploaded_file.name}': {fallback_err}. Skipping.")

                    os.remove(temp_path)
                
                #chunk
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                if not all_documents:
                    st.error("No text could be extracted from the uploaded files. Please check that the PDFs contain readable text.")
                    st.stop()

                chunks = text_splitter.split_documents(all_documents)

                if not chunks:
                    st.error("No content chunks were produced. The documents may be empty or image-only PDFs.")
                    st.stop()

                # embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # vector store
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                #LLM
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an enterprise knowledge assistant. Answer based ONLY on the provided context.
At the end of your answer, cite the source document name.

Context: {context}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                document_chain = create_stuff_documents_chain(llm, prompt)
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                
                st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                st.success(f"rocessed {len(uploaded_files)} documents with {len(chunks)} chunks")
        else:
            st.error("Please upload files first")
    
    st.divider()
    st.markdown("### Features")
    st.markdown("- Semantic search")
    st.markdown("- Source citations")
    st.markdown("- Conversational memory")
    st.markdown("- No hallucinations (grounded retrieval)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(f"- {src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if st.session_state.retrieval_chain:
            with st.spinner("Retrieving and generating..."):
                # Convert chat history to LangChain format
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                result = st.session_state.retrieval_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                
                answer = result["answer"]
                
                sources = []
                for doc in result.get("context", []):
                    source_name = doc.metadata.get("source", "Unknown").split("/")[-1]
                    sources.append(source_name)
                sources = list(set(sources))
                
                if sources and "Source:" not in answer and "source:" not in answer:
                    answer += f"\n\n**Source:** {', '.join(sources)}"
                
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.caption(f"- {src}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
        else:
            st.warning("Please upload and process documents first")

st.divider()
st.caption("Built with LangChain 1.x + Groq + FAISS | Enterprise RAG System")