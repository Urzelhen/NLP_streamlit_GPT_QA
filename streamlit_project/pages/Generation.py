import streamlit as st
import pandas as pd

st.sidebar.markdown("Generation")

st.write("""
# Generation
""")
question_text = st.text_input('Задайте promt')

def features():
    max_length = st.sidebar.slider('max_length', 10, 500, 100)
    num_beams = st.sidebar.slider('num_beams', 1, 10, 5)
    temperature = st.sidebar.slider('temperature', 1, 10000, 10)
    top_k = st.sidebar.slider('top_k', 5, 100, 50)
    top_p = st.sidebar.slider('top_p', 0.0, 1.0, 0.5)
    no_repeat_ngram_size = st.sidebar.slider('no_repeat_ngram_size', 1, 10, 5)
    num_return_sequences = st.sidebar.slider('num_return_sequences', 1, 10, 5)

    data = {'max_length': max_length,
            'num_beams': num_beams,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'num_return_sequences': num_return_sequences
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = features()