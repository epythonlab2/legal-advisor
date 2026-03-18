import streamlit as st
import requests

st.title("Ethio Legal Assistant")

q = st.text_input("Ask a legal question")

if st.button("Ask"):
    r = requests.post("http://localhost:8000/ask", params={"question": q})
    st.write(r.json())