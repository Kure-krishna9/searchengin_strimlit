from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st

## Arxiv--Research
## Tools Creation
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,DuckDuckGoSearchAPIWrapper
# Use the inbuild tools of wikipedia
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# wiki.name
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
Arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
from langchain.agents import create_openai_tools_agent,initialize_agent,AgentType

tools=[wiki,Arxiv]

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_token"]=os.getenv("HF_token")
os.environ["Groq_key"]=os.getenv("Groq_key")
os.environ["Langchain_api_key"]=os.getenv("Langchain_api_key")
# os.environ["Langchain_project"]=os.getenv("Tool_agent_llm_project")
os.environ["Langchain_api_key"]=os.getenv("Langchain_api_key")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["Langchain_project"]=os.getenv("Langchain_project")

# Set USER_AGENT to identify your app to web services
os.environ["USER_AGENT"] = "LangChainBot/1.0 (youremail@example.com)"
Embeddings=HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')


loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documet=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=400)
split_doc=documet.split_documents(docs)
vectore_store=FAISS.from_documents(split_doc,embedding=Embeddings)
retriver=vectore_store.as_retriever()

from langchain.tools.retriever import create_retriever_tool
# retriver_tool=create_retriever_tool(retriver,"langsith-search","Search any information about langsmith")
# retriver_tool = create_retriever_tool(retriver, "langsith-search", "Search any information about langsmith")
from langchain.tools import Tool

retriver_tool = Tool(
    name="langsmith-search",
    func=lambda q: "\n\n".join([doc.page_content for doc in retriver.get_relevant_documents(q)]),
    description="Use this tool to search LangSmith internal docs.",
)

# tools
groq_api_key=os.getenv("Groq_key")

# serach=DuckDuckGoSearchAPIWrapper(name="Search")
search = DuckDuckGoSearchAPIWrapper()

# search_tool = Tool(
#     name="Search",
#     func=search.run,
#     description="Use this tool to search the web using DuckDuckGo"
# )
st.title("Langchain search engine")

## Sidebar for setting
st.sidebar.title("Setting")
api_key=st.sidebar.text_input("Enter a Groq API Key",type='password')



# if "messages" not in st.session_state:
#     st.session_state["messages"]=[{
#         "Role":"assistent","content":"Hi ia a chat bot who can search the web." 

#     }]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",  # ✅ fixed "Role" -> "role"
        "content": "Hi! I'm a chatbot who can search the web."
    }]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
# for msg in st.session_state.messages:
#     if isinstance(msg, dict) and "role" in msg and "content" in msg:
#         st.chat_message(msg["role"]).write(msg["content"])
#     else:
#         st.warning("Invalid message format in session state.")

# # if prompt:=st.chat_input(placeholder="what is machine learning"):
# #     st.session_state.messages.append({"role":"user","content":prompt})
# #     st.chat_input("user").write(prompt)
    
# if prompt := st.chat_input(placeholder="What is machine learning?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     # You can now add AI response logic here
#     response = "Machine learning is a field of AI..."  # Placeholder response
#     st.session_state.messages.append({"role": "user", "content": response})
#     st.chat_message("assistant").write(response)

#     llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant",streaming=True)
#     # retriver_tool.name
#     tools=[wiki,Arxiv,retriver_tool]
#     search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)

#     with st.chat_message("assistant"):
#         st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#         response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
#         st.session_state.messages_append({'role':'assistent',"content":response})
#         st.write(response)

if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize model
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=True)
    tools = [wiki, Arxiv, retriver_tool]
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Run agent with proper input
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # ✅ FIXED: only pass the user's prompt
        try:
            response = search_agent.invoke({"input": prompt}, config={"callbacks": [st_cb]})

        except Exception as e:
            response = f"❌ Error: {e}"

        # ✅ Append assistant response
        st.session_state.messages.append({"role": "assistant", "page_content": response})
        st.write(response)

