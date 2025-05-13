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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


os.environ['GROQ_API_KEY']= st.secrets["GROQ_API_KEY"]

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2; font-family: "Segoe UI", sans-serif;'>
        📄 InsightPDF AI
    </h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Upload your PDF and ask questions in natural language
    </p>
""", unsafe_allow_html=True)

def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
## Groq
# api_key = st.sidebar.text_input("Enter Groq key",type='password')

# if api_key:
# llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=api_key)
select_model = st.sidebar.selectbox(label="Choose the model",options=["Gemma2-9b-It","Llama3-8b-8192","Llama3-70b-8192"])
llm = ChatGroq(model=select_model)

#set title of sidebar
st.sidebar.title("Upload File")
## Get the session name
# session_id= st.sidebar.text_input("Session ID",value="default_session")

## Check session store
if 'store' not in st.session_state:
    st.session_state.store={}
    
## Check File procession
if 'is_file_processed' not in st.session_state:
    st.session_state.is_file_processed=False
    
upload_file = st.sidebar.file_uploader("choose a pdf file",type="pdf",accept_multiple_files=True)

##### LLM Design 

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
" If You don't know the answer simply say don't know , use three sentence max and keep the answer concise"
"\n\n"
"{context}"
)


qa_prompt = ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human',"{input}")
])

def process_file_and_create_chain():
    document=[]
    for uploaded_file in upload_file:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
    
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        document.extend(docs)
        
    with st.spinner("Analyzing Files ..."):
        ## Create chunks and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=200)
        chunk_docs = text_splitter.split_documents(document)
        
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunk_docs,embedding)
        retriever = db.as_retriever()
        # st.stop()
    
    ## Create retrieval Chain 
    history_aware_retriever = create_history_aware_retriever(llm , retriever ,contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key = "input",
    history_messages_key="chat_history",
    output_messages_key= "answer"
    )

    st.session_state.is_file_processed=True

def generate_response(user_input):
    session_id = "default"
    session_history = get_session_history(session_id)
    st.chat_message('user').write(user_input)
    with st.chat_message("assistant"):
        streamlit_callbacks = StreamlitCallbackHandler(st.container())
        response =  st.session_state.conversational_rag_chain.invoke({"input":user_input},
                                            config={"configurable":{"session_id":session_id}})
        st.session_state.messages.append({"role":"Assistant","content":response['answer']})
        st.write(response['answer'])    


if upload_file and not st.session_state.is_file_processed:
    process_file_and_create_chain()
        
if st.session_state.is_file_processed:
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role":"Assistant","content":"How Can I Help You ?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg["content"])

    user_input = st.chat_input(placeholder="Ask the question")
    if user_input:
        generate_response(user_input)
    
else:
    st.spinner("PDF Under-process")
    
# else:
# st.info("Enter Your API Key")        
