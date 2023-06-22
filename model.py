import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open("model.pkl","rb"))

st.title("Heart Disease Prediction using ML")
st.image("""https://drive.google.com/uc?export=view&id=1j70ye8HLzzjYx8-T0R8485V24xlCmg9S""")

st.write('---')
st.write('**Description of Dataset**')
st.write('**BMI** - Computed body mass index')
st.write('**Smoking** - Smoked at Least 100 Cigarettes')
st.write('**Alcohol Drinking** - Heavy Alcohol Consumption Calculated Variable (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)')
st.write('**Stroke** - Ever Diagnosed with a Stroke')
st.write('**Physical Health** - Number of Days Physical Health Not Good (Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?)')
st.write('**Mental Health** -  Number of Days Mental Health Not Good (Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?)')
st.write('**Diff Walking** - Difficulty Walking or Climbing Stairs')
st.write('**Sex** - Are you male or female?')
st.write('**Age Category** - Fourteen-level age category')
st.write('**Race** - Imputed race/ethnicity value')
st.write('**Diabetic** - Ever Diagnosed with a Diabetic')
st.write('**Physical Activity** - Exercise in Past 30 Days')
st.write('**Gen Health** - General Health')
st.write('**Sleep Time** - On average, how many hours of sleep do you get in a 24-hour period?')
st.write('**Asthma** - Ever Diagnosed with a Diabetic')
st.write('**Kidney Disease** - Ever Diagnosed with a Kidney Disease')
st.write('**Skin Cancer** - Ever Diagnosed with a Skin Cancer')

st.header('Enter the Features of the Heart Disease:')

BMI = st.number_input('BMI:', min_value=10.0, max_value=100.0)

Smoking = st.selectbox('Smoking:', ['Yes', 'No'])

AlcoholDrinking = st.selectbox('Alcohol Drinking:', ['Yes', 'No'])

Stroke = st.selectbox('Stroke:', ['Yes', 'No'])

PhysicalHealth = st.number_input('Physical Health:', min_value=0, max_value=30)

MentalHealth = st.number_input('Mental Health:', min_value=0, max_value=30)

DiffWalking = st.selectbox('Diff Walking:', ['Yes', 'No'])

Sex = st.selectbox('Gender:', ['Male', 'Female'])

AgeCategory = st.selectbox('Age Category:', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])

Race = st.selectbox('Race:', ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'])

Diabetic = st.selectbox('Diabetic:', ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])

PhysicalActivity = st.selectbox('Physical Activity:', ['Yes', 'No'])

GenHealth = st.selectbox('Gen Health:', ['Very good', 'Fair', 'Good', 'Poor', 'Excellent'])

SleepTime = st.number_input('Sleep Time:', min_value=0, max_value=24)

Asthma = st.selectbox('Asthma:', ['Yes', 'No'])

KidneyDisease = st.selectbox('Kidney Disease:', ['Yes', 'No'])

SkinCancer = st.selectbox('Skin Cancer:', ['Yes', 'No'])

def Predict(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer):

    df = pd.DataFrame([[BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer]], columns=["BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"])
    
    df["Smoking_No"] = df["Smoking"].apply(lambda x: 1 if x=="No" else 0)
    df["Smoking_Yes"] = df["Smoking"].apply(lambda x: 1 if x=="Yes" else 0)

    df["AlcoholDrinking_No"] = df["AlcoholDrinking"].apply(lambda x: 1 if x=="No" else 0)
    df["AlcoholDrinking_Yes"] = df["AlcoholDrinking"].apply(lambda x: 1 if x=="Yes" else 0)

    df["Stroke_No"] = df["Stroke"].apply(lambda x: 1 if x=="No" else 0)
    df["Stroke_Yes"] = df["Stroke"].apply(lambda x: 1 if x=="Yes" else 0)

    df["DiffWalking_No"] = df["DiffWalking"].apply(lambda x: 1 if x=="No" else 0)
    df["DiffWalking_Yes"] = df["DiffWalking"].apply(lambda x: 1 if x=="Yes" else 0)

    df["Sex_Female"] = df["Sex"].apply(lambda x: 1 if x=="Female" else 0)
    df["Sex_Male"] = df["Sex"].apply(lambda x: 1 if x=="Male" else 0)

    df["AgeCategory_21"] = df["AgeCategory"].apply(lambda x: 1 if x=="18-24" else 0)
    df["AgeCategory_27"] = df["AgeCategory"].apply(lambda x: 1 if x=="25-29" else 0)
    df["AgeCategory_32"] = df["AgeCategory"].apply(lambda x: 1 if x=="30-34" else 0)
    df["AgeCategory_37"] = df["AgeCategory"].apply(lambda x: 1 if x=="35-39" else 0)
    df["AgeCategory_42"] = df["AgeCategory"].apply(lambda x: 1 if x=="40-44" else 0)
    df["AgeCategory_47"] = df["AgeCategory"].apply(lambda x: 1 if x=="45-49" else 0)
    df["AgeCategory_52"] = df["AgeCategory"].apply(lambda x: 1 if x=="50-54" else 0)
    df["AgeCategory_57"] = df["AgeCategory"].apply(lambda x: 1 if x=="55-59" else 0)
    df["AgeCategory_62"] = df["AgeCategory"].apply(lambda x: 1 if x=="60-64" else 0)
    df["AgeCategory_67"] = df["AgeCategory"].apply(lambda x: 1 if x=="65-69" else 0)
    df["AgeCategory_72"] = df["AgeCategory"].apply(lambda x: 1 if x=="70-74" else 0)
    df["AgeCategory_77"] = df["AgeCategory"].apply(lambda x: 1 if x=="75-79" else 0)
    df["AgeCategory_80"] = df["AgeCategory"].apply(lambda x: 1 if x=="80 or older" else 0)

    df["Race_American Indian/Alaskan Native"] = df["Race"].apply(lambda x: 1 if x=="American Indian/Alaskan Native" else 0)
    df["Race_Asian"] = df["Race"].apply(lambda x: 1 if x=="Asian" else 0)
    df["Race_Black"] = df["Race"].apply(lambda x: 1 if x=="Black" else 0)
    df["Race_Hispanic"] = df["Race"].apply(lambda x: 1 if x=="Hispanic" else 0)
    df["Race_Other"] = df["Race"].apply(lambda x: 1 if x=="Other" else 0)
    df["Race_White"] = df["Race"].apply(lambda x: 1 if x=="White" else 0)

    df["Diabetic_No"] = df["Diabetic"].apply(lambda x: 1 if x=="No" else 0)
    df["Diabetic_No, borderline diabetes"] = df["Diabetic"].apply(lambda x: 1 if x=="No, borderline diabetes" else 0)
    df["Diabetic_Yes"] = df["Diabetic"].apply(lambda x: 1 if x=="Yes" else 0)
    df["Diabetic_Yes (during pregnancy)"] = df["Diabetic"].apply(lambda x: 1 if x=="Yes (during pregnancy)" else 0)

    df["PhysicalActivity_No"] = df["PhysicalActivity"].apply(lambda x: 1 if x=="No" else 0)
    df["PhysicalActivity_Yes"] = df["PhysicalActivity"].apply(lambda x: 1 if x=="Yes" else 0)

    df["GenHealth_Excellent"] = df["GenHealth"].apply(lambda x: 1 if x=="Excellent" else 0)
    df["GenHealth_Fair"] = df["GenHealth"].apply(lambda x: 1 if x=="Fair" else 0)
    df["GenHealth_Good"] = df["GenHealth"].apply(lambda x: 1 if x=="Good" else 0)
    df["GenHealth_Poor"] = df["GenHealth"].apply(lambda x: 1 if x=="Poor" else 0)
    df["GenHealth_Very good"] = df["GenHealth"].apply(lambda x: 1 if x=="Very good" else 0)

    df["Asthma_No"] = df["Asthma"].apply(lambda x: 1 if x=="No" else 0)
    df["Asthma_Yes"] = df["Asthma"].apply(lambda x: 1 if x=="Yes" else 0)

    df["KidneyDisease_No"] = df["KidneyDisease"].apply(lambda x: 1 if x=="No" else 0)
    df["KidneyDisease_Yes"] = df["KidneyDisease"].apply(lambda x: 1 if x=="Yes" else 0)

    df["SkinCancer_No"] = df["SkinCancer"].apply(lambda x: 1 if x=="No" else 0)
    df["SkinCancer_Yes"] = df["SkinCancer"].apply(lambda x: 1 if x=="Yes" else 0)

    df = df.drop("Smoking", axis =1)
    df = df.drop("AlcoholDrinking", axis =1)
    df = df.drop("Stroke", axis =1)
    df = df.drop("DiffWalking", axis =1)
    df = df.drop("Sex", axis =1)
    df = df.drop("AgeCategory", axis =1)
    df = df.drop("Race", axis =1)
    df = df.drop("Diabetic", axis =1)
    df = df.drop("PhysicalActivity", axis =1)
    df = df.drop("GenHealth", axis =1)
    df = df.drop("Asthma", axis =1)
    df = df.drop("KidneyDisease", axis =1)
    df = df.drop("SkinCancer", axis =1)

    df = StandardScaler().fit_transform(df)

    prediction = model.predict(df)

    return prediction

if st.button('Predict Heart Disease'):
    Results = Predict(BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer)
    if Results[0] == 0:
        st.success("Hurray! you don't have any Heart Disease")
    elif Results[0] == 1:
        st.success("OOPS! you have heart disease")
