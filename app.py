import streamlit as st
import openai
import pandas as pd
import json
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

with open("config.json", "r") as f:
    config = json.load(f)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

df = pd.read_csv('data/amazon.csv')

features_needed = ['product_id', 'product_name', 'category', 'discounted_price',
       'actual_price', 'discount_percentage', 'rating', 'rating_count',
       'about_product','review_title','review_content']
df = df[features_needed]


df["text"] = df.apply(lambda row: f"Product id: {row['product_id']}\nProduct name: {row['product_name']}\nCategory: {row['category']}\nDiscounted price: {row['discounted_price']}\nActual price: {row['actual_price']}\nDiscount percentage: {row['discount_percentage']}\nRating: {row['rating']}\nRating count: {row['rating_count']}\nAbout product: {row['about_product']}\nReview title: {row['review_title']}\nReview content: {row['review_content']}", axis=1)


embeddings = OpenAIEmbeddings(model='text-embedding-3-large')


persist_db = "./chroma.db"
# vector_store = Chroma(collection_name="product_data",embedding_function=embeddings, persist_directory=persist_db)

# for i, row in df.iterrows():
#     vector_store.add_texts([row['text']], 
#                       metadatas=[{"product_id":row['product_id'], "product_name":row['product_name'], "category":row['category'], "discounted_price":row['discounted_price'], "actual_price":row['actual_price'], "discount_percentage":row['discount_percentage'], "rating":row['rating'], "rating_count":row['rating_count'], "about_product":row['about_product'], "review_title":row['review_title'], "review_content":row['review_content']}]
#                       )

vector_store = Chroma(collection_name="product_data",embedding_function=embeddings, persist_directory=persist_db)

def search_product(user_query, top_k=1):
    results = vector_store.similarity_search(user_query, k=top_k)
    r_list = []
    if results:
        for res in results:
            r_list.append(res.metadata["product_name"] + " - " + res.page_content)
        return r_list
    else:
        return ["No matching product found."]

def generate_llm_response():
    system_prompt = """
    You are an amazon shopping assistant helping users to decide on the products.
    Relevant context fetched from database : {context}
    Previous conversation: {chat_history}
    For the given query by the user {user_input}.
    Summarize the answer in a few sentences. Be concise and informative.
    """
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    llm = ChatOpenAI(temperature=0.0, openai_api_key=os.environ["OPENAI_API_KEY"])
    

    chain = LLMChain(
        llm=llm, 
        prompt=prompt_template, 
        output_key="answer"
    )
    
    return chain


if __name__ == "__main__":
    st.set_page_config(page_title="Amazon Product Chatbot - RAG")
    st.title("Amazon Product Chatbot - RAG")
    chain = generate_llm_response()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "context" not in st.session_state:
        st.session_state.context = None 

    user_input = st.text_input("Please enter your query", key='input')

    if user_input.strip():  
        if st.session_state.context is None:
            st.session_state.context = search_product(user_query=user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

        formatted_history = []
        for chat in st.session_state.chat_history:
            formatted_history.append({
            "role": chat["role"],
            "content": chat["content"]
            })

        response = chain.invoke({
            "user_input": user_input, 
            "chat_history": formatted_history,
            "context": st.session_state.context
        })
        
        st.write(response["answer"])

        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

        st.write("="*70)
        st.markdown("**Chat History:**")

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**User**: *{chat['content']}*")
        elif chat["role"] == "assistant":
            st.markdown(f"**Assistant**: {chat['content']}")