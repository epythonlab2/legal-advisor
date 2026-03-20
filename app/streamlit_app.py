import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.title("⚖️ የኢትዮጵያ ሕግ አጋዥ(ነጋሪት ጋዜጣ)")

query = st.text_input("የሕግ ጥያቄዎን ያስገቡ")

if st.button("ፈልግ") and query:

    payload = {
        "query": query,
        "top_k": 5
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            results = response.json()

            if not results:
                st.warning("ውጤት አልተገኘም")
            else:
                st.success(f"{len(results)} ውጤቶች ተገኝተዋል")

                for i, doc in enumerate(results, 1):
                    st.subheader(f"ውጤት {i}")
                    st.write(doc["content"])

                    with st.expander("መረጃ (Metadata)"):
                        st.json(doc["metadata"])

        else:
            st.error(f"የAPI ስህተት: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"የግንኙነት ስህተት: {e}")