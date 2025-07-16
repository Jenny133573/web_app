import streamlit as st
import numpy as np
from machine_learning_py import get_model
from explore_page import show_data
import pandas as pd
model=get_model()
data=show_data()
def show_predict_page():
    st.title('Annual Developer Salaries Predictor')
    st.write('### We need information as the following to predict')

    # add select box
    countries=(
        'United States of America',
        'Germany',
        'United Kingdom of Great britain and Northern Ireland',
        'India',
        'France',
        'Canada',
        'Ukraine',
        'Netherlands',
        'Italy',
        'Australia',
        'Sweden',
        'Brazil',
        'Poland',
        'Switzerland',
        'Austria'
    )

    educations=(
        "Bachelor's degree",
        "Post grad",
        "Less than a Bachelors"
    )
    country=st.selectbox('Country', countries)
    education=st.selectbox('Education Level', educations)
    experience=st.slider('Years of experience',0,50,3)

    ok=st.button('Calculate Salary')
    if ok:
        sample=pd.DataFrame({'Country': country,
                             'EdLevel': education,
                             'YearsProCode': experience})
        sample_encoded=pd.get_dummies(sample)
        sample_encoded=sample_encoded.reindex(columns=data.columns.drop('Salary'), fill_value=0)
        predicted_salary=np.expm1(model.predict(sample_encoded))
        st.subheader(f'The estimated salary is ${predicted_salary:.2f}')