# импорт библиотек
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# загрузка модели
model = joblib.load("sleep_model.pkl")
data = pd.read_csv('sleep_health_lifestyle_dataset.csv')


for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype in ['int64', 'float64']:
          median_value = data[col].median() # Замена на медиану для численных признаков
          data[col] = data[col].fillna(median_value)
        else:
            if col == 'Sleep Disorder' and data[col].isna().any():
                data[col] = data[col].fillna("None")
            else:
                mode_value = data[col].mode()[0]  # Замена на моду для категориальных признаков
                data[col] = data[col].fillna(mode_value)

data = data.drop(['Person ID','Blood Pressure (systolic/diastolic)'], axis=1)
# Категориальные признаки
categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
numerical_features = ['Age', 'Sleep Duration (hours)', 'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)', 'Heart Rate (bpm)', 'Daily Steps']

# Создаем энкодер и обучаем его на всех данных, чтобы не было проблем с новыми значениями
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X = data.drop('Quality of Sleep (scale: 1-10)', axis=1)
encoder.fit(X[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)


all_columns = [
    'Gender',
    'Age',
    'Occupation',
    'Sleep Duration (hours)',
    'Physical Activity Level (minutes/day)',
    'Stress Level (scale: 1-10)',
    'BMI Category',
    'Heart Rate (bpm)',
    'Daily Steps',
    'Sleep Disorder'
    ]

# Указание контента сайта
st.title("Прогнозирование качества сна")

gender = st.selectbox('Пол', options=data['Gender'].unique())
age = st.number_input(
    "Возраст",
    min_value=18,
    max_value=99,
    value=18,
    step=1
)
sleep_duration = st.number_input(
    "Продолжительность сна",
    min_value=0.0,
    max_value=12.0,
    value=0.0,
    step=0.1
)
physical_activity = st.number_input(
    "Физическая активность",
    min_value=1,
    max_value=200,
    value=1,
    step=10
)
stress_level = st.number_input(
    "Уровень стресса",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)
heart_rate = st.number_input(
    "Пульс",
    min_value=50,
    max_value=100,
    value=50,
    step=1
)
daily_steps = st.number_input(
    "Шагов в день",
    min_value=2000,
    max_value=20000,
    value=2000,
    step=100
)

occupation = st.selectbox('Занятость', options=data['Occupation'].unique())

BMI = st.selectbox('Степень ожирения', options=data['BMI Category'].unique())

sleep_disorder = st.selectbox('Нарушения сна', options=data['Sleep Disorder'].unique())

# Преобразование данных и прогноз
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Occupation': [occupation],
    'Sleep Duration (hours)': [sleep_duration],
    'Physical Activity Level (minutes/day)': [physical_activity],
    'Stress Level (scale: 1-10)': [stress_level],
    'BMI Category': [BMI],
    'Heart Rate (bpm)': [heart_rate],
    'Daily Steps': [daily_steps],
    'Sleep Disorder': [sleep_disorder],
})
# Отделяем числовые и категориальные признаки
input_data_num = input_data[numerical_features]
input_data_cat = input_data[categorical_features]


# Кодируем категориальные признаки
input_data_encoded = encoder.transform(input_data_cat)
input_data_encoded = pd.DataFrame(input_data_encoded, columns = feature_names)


# Объединяем числовые и закодированные категориальные признаки
input_data_processed = pd.concat([input_data_num, input_data_encoded], axis = 1)

if st.button('Предсказать качество сна'):
    prediction = model.predict(input_data_processed)[0]
    st.success(f'Ваше качество сна: {prediction :.1f}')
