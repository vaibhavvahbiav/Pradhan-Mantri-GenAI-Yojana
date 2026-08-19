import streamlit as st
import pandas as pd
st.title("Hello From ChatJiPiTi!!")
name=st.text_input("Ask your Questions")
st.write("mera pehla streamlit ka parivaar ka beta hu mein !!!!  - Shendre ")
if st.button("greet kar la!!"):
    st.success(f"Hello , La {name}")

upload_file=st.file_uploader("upload kar CSV file teri gend mein!! ", type ='csv')
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)

st.slider("choose a range", 0.1)
st.selectbox("select a fruit",["apple","banana","mango"])
st.multiselect("select language",["java","python","c++"])
st.checkbox("i agree to terms and conditions!!")