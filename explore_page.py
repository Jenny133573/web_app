import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from machine_learning_py import shorten_categories, clean_experience, clean_education
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np


@st.cache
def show_data():
    df=pd.read_csv(r"C:\Users\Danny\Downloads\survey_results_public (1).csv")
    df=df[['YearsCodePro', 'Country', 'EdLevel', 'Employment', 'ConvertedCompYearly']]
    df=df.rename({'ConvertedCompyearly': 'Salary'},axis=1)
    features=['Country', 'EdLevel', 'Employment', 'YearsCodePro']
    df[features]=SimpleImputer(strategy='most_frequent').fit_transform(df[features])
    df_model=df[features+ ['Salary']]

    df_model=pd.get_dummies(df_model, drop_first=True)
    # Split into datasets for training
    known=df_model[df_model['Salary'].notna()]
    unknown=df_model[df_model['Salary'].isna()]

    X=known.drop('Salary', axis=1)
    y=np.log1p(known.Salary)
    model=RandomForestRegressor(n_estimators=100, random_state=50).fit(X,y)

    predicted_log=model.predict(unknown.drop('Salary', axis=1))
    predicted=np.expm1(predicted_log)
    df.loc[df.Salary.isna(), 'Salary']=predicted

    country_map=shorten_categories(df.COuntry.value_count(),400)
    df['Country']=df['Country'].map(country_map)
    df=df[df['Employment']=='Employed, full-time']
    df=df.drop('Employment', axis=1)
    df=df[df['Salary']<=1000000]
    df=df[df['Salary']>=10000]
    df=df[df['Country']!='Other']
    df['YearsCodePro']=df['YearsCodePro'].apply(clean_experience)
    df['EdLevel']=df['EdLevel'].apply(clean_education)
    return df

df=show_data()


def show_explore_page():
    st.title('Annual Developer Salaries')

    st.write(
        '''
### Stack Overflow Developer Survey 2024 (newest)
'''
)
    data=df.Country().value_counts()

    fig1, ax1=plt.subplots()
    ax1.pie(data, labels=data.index, autopct='%1.1f%%', starangle=90)
    ax1.axis=True

    st.write('### Number of data from different countries')
    st.pyplot(ax1)

    st.write(
        '''
### Average salary based on different countries
'''
)
    data=df.groupby('Country')['Salary'].mean().sort_values(ascending=False)
    st.bar_chart(data)

    st.write(
        '''
### Average salary based on experience
'''
)
    data=df.groupby('YearsCodePro')['Salary'].mean().sort_values(ascending=False)
    st.line_chart(data)
