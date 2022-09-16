import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

checkpoint = "Grossmend/rudialogpt3_medium_based_on_gpt2"   
tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

st.sidebar.markdown("QA")

st.image("/home/valentina/ds_offline/learning/12-nlp/streamlit_project_others/QA/qa.png",
        width=200,
        )

question_text = st.text_input('Задайте вопрос')
context_text = st.text_input('Напишите контекст')

if ((question_text != '') and (context_text != '')):
        test_input = question_text + context_text + tokenizer.eos_token +  "|1|2|"

        st.write('a1')
        input_ids = tokenizer([test_input], return_tensors="pt").input_ids
        st.write('a2')
        response = tokenizer.decode(model.generate(input_ids.cpu(),
                                                max_length=len(tokenizer([test_input], return_tensors="pt").input_ids[0]) + 32,
                                                temperature=0.6,
                                                num_beams=2,
                                                repetition_penalty=10.,
                                                do_sample=False).cpu()[:, input_ids.shape[-1]:][0], skip_special_tokens=False)
        st.write('a3')
        st.write(str(response).replace('</s>','').replace('\t',''))
        st.write('a4')

