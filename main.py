import streamlit as st

import pandas as pd
import numpy as np

import joblib

#loading the trained model file

model_file=joblib.load(open("emo_class_model_27aug.pkl","rb"))

def predict_emotion(input):
    return model_file.predict([input])

def predict_probability(input):
    return model_file.predict_proba([input])

def main():
    st.title("Emotion Detector")
    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        st.subheader("Express your Emotion Here:")

        with st.form(key="emo_form"):
            raw_text= st.text_area("Type your Emotion here")
            submit_text=st.form_submit_button(label="Submit")

        if submit_text:
            col1,col2=st.columns(2)
            prediction=predict_emotion(raw_text)
            probability=predict_probability(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence : {}".format(np.max(probability)))

            with col2:
                st.success("Prediction probability")
                st.write(probability)




    elif choice=="Monitor":
        st.subheader("Monitor my app")

    else:
        st.subheader("Know about me. Thanks")





if __name__=='__main__':
    main()
