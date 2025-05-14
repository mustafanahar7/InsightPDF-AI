import os
os.environ["STREAMLIT_WATCHER_DISABLE_AUTO_WATCH"] = "true"

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader , TextLoader , UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


os.environ['GROQ_API_KEY']= st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="InsightPDF-AI",page_icon="📄")
########################## Headings ########################## 
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2; font-family: "Segoe UI", sans-serif;'>
        📄 InsightPDF AI
    </h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Upload your Document and ask questions in natural language
    </p>
    <strong>Do not upload sensitive or confidential files.</strong> This tool is for general document analysis only.
    
""", unsafe_allow_html=True)
st.sidebar.title("Upload File")
#################### ***************** #######################


########################## Check session store and file processing ##########################
if 'store' not in st.session_state:
    st.session_state.store={}

if 'is_file_processed' not in st.session_state:
    st.session_state.is_file_processed=False
#################### ***************** #######################    

def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    
########################## Select and initial LLM Model ##########################
select_model = st.sidebar.selectbox(label="Choose the model",options=["Gemma2-9b-It","Llama3-8b-8192","Llama3-70b-8192"])
llm = ChatGroq(model=select_model)
#################### ***************** #######################


########################## Prompt Design ##########################
contextualize_system_prompt ="""
    Give a chat history and latest user question
    which might reference context in chat history ,
    formulate standalone question which can be undertood
    without the chat history , DO NOT answer the question
    just Reformulate it if needed otherwise give as it is

"""
contextualize_prompt = ChatPromptTemplate.from_messages([
    ('system',contextualize_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human',"{input}")
])

## Question Answer
system_prompt =(
"You are an assistant for question-answer task ."
"Use the Following piece of context from Retrieved Document to answer the question "
" If You don't know the answer simply say don't know ."
"\n\n"
"{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human',"{input}")
])

#################### ******** End of Prompt Design********* #######################

########################## Upload File and Processing ##########################
upload_file = st.sidebar.file_uploader("choose a file",type=["pdf","txt","docx"],accept_multiple_files=True)
def process_file_and_create_chain():
    document=[]
    for uploaded_file in upload_file:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()

        temp_path = f"./temp{suffix}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Choose appropriate loader
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif suffix == ".txt":
            loader = TextLoader(temp_path)
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        else:
            st.error("Unsupported file format. Only PDF and TXT files are supported.")
            continue
        
        if loader:
            docs = loader.load()
            document.extend(docs)
        
    with st.spinner("Analyzing Files ..."):
        ## Create chunks and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=200)
        chunk_docs = text_splitter.split_documents(document)
        
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunk_docs,embedding)
        st.session_state.retriever = db.as_retriever()
        # st.stop()
    
    st.session_state.is_file_processed=True
    return st.session_state.retriever

def generate_response(user_input):
    session_id = "default"
    session_history = get_session_history(session_id)
    st.chat_message('user').write(user_input)
    with st.chat_message("assistant"):
        streamlit_callbacks = StreamlitCallbackHandler(st.container())
        response = conversational_rag_chain.invoke({"input":user_input},
                                            config={"configurable":{"session_id":session_id}})
        st.session_state.messages.append({"role":"Assistant","content":response['answer']})
        st.write(response['answer'])    


if upload_file:
    retriever = process_file_and_create_chain()
    ## Create retrieval Chain 
    history_aware_retriever = create_history_aware_retriever(llm , retriever ,contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key = "input",
    history_messages_key="chat_history",
    output_messages_key= "answer"
    )
        
if st.session_state.is_file_processed:
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role":"Assistant","content":"How Can I Help You ?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg["content"])

    user_input = st.chat_input(placeholder="Ask the question")
    if user_input:
        generate_response(user_input)
    
else:
    st.spinner("Document Under-process")
    