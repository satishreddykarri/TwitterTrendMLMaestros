import streamlit as st
import pandas as pd
from model import predict_trending_topic, df, get_contexts_for_country

# Streamlit UI
st.title("Twitter Trending Topics Predictor")

country = st.selectbox("Select Country", df['country'].unique())
context_options = get_contexts_for_country(country)
context = st.selectbox("Select Context", context_options)

if st.button("Predict Trending Topic"):
    trending_topic = predict_trending_topic(country, context)
    st.success(f"Predicted Trending Topic: {trending_topic}")