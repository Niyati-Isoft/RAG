import weaviate, langchain_community
print(weaviate.__version__)
print(langchain_community.__version__)
import streamlit as st, weaviate, os

st.set_page_config(page_title="Weaviate Sanity", layout="centered")
cfg = st.secrets.get("weaviate", {})
url = cfg.get("url"); key = cfg.get("api_key")

st.write("Client:", weaviate.__version__)
if not url: st.stop()

client = weaviate.Client(url=url, auth_client_secret=weaviate.AuthApiKey(key)) if key else weaviate.Client(url=url)
schema = client.schema.get()
st.success("Connected to Weaviate ✅")
st.write("Classes:", [c["class"] for c in schema.get("classes", [])])
