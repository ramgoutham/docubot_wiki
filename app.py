import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
import wikipediaapi
import os
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

def get_wikipedia_data(query):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en', extract_format=wikipediaapi.ExtractFormat.WIKI, user_agent='streamlit-app/1.0'
    )
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return "Sorry, I couldn't find information on that topic in Wikipedia."

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap= 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key = 'chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def train_lda_model(text_chunks):
    # Tokenize and preprocess text
    tokenized_text = [chunk.split() for chunk in text_chunks]

    # Create dictionary and corpus
    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    return lda_model, dictionary

def handle_userinput(user_question, source):
    if source == "document":
        response = st.session_state.conversation({'question': user_question})
        if response:
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", "Sorry, I can't find the answer."), unsafe_allow_html=True)
    elif source == "wikipedia":
        response = get_wikipedia_data(user_question)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Document-based CHATBOT",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("QA system from Documents or Wikipedia :books:")

    # Add tabs for document and Wikipedia queries
    tabs = ["Document", "Wikipedia"]
    selected_tab = st.radio("Select source:", tabs)

    user_question_key = f"{selected_tab.lower()}_user_question"  # Unique key for text_input

    # Render input box based on selected tab
    if selected_tab == "Document":
        user_question = st.text_input("Ask a question:", key=user_question_key)
    elif selected_tab == "Wikipedia":
        user_question = st.text_input("Ask a question (Wikipedia):", key=user_question_key)

    if user_question:
        handle_userinput(user_question, selected_tab.lower())

    # Document tab functionality
    if selected_tab == "Document":
        with st.sidebar:
            st.subheader("Uploaded documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process the uploaded docs"):
                with st.spinner("Document being processed"):
                    # Get the PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vector_store(text_chunks)

                    # Train LDA model
                    lda_model, dictionary = train_lda_model(text_chunks)

                    # Display topics
                    st.subheader("Extracted Topics:")
                    for topic_id, words in lda_model.print_topics():
                        st.write(f"Topic {topic_id}: {words}")

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    # Wikipedia tab functionality
    elif selected_tab == "Wikipedia":
        if user_question:
            handle_userinput(user_question, "wikipedia")


if __name__ == '__main__':
    main()
