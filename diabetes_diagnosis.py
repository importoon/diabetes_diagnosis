import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. 모델 불러오기 (없으면 학습)
@st.cache_resource
def load_model():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # 임시로 모델 학습
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # 대신 Pima Indians 데이터셋을 불러와서 학습하는 게 좋아
        df = pd.read_csv("diabetes.csv")
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        with open("diabetes_model.pkl", "wb") as f:
            pickle.dump(model, f)
        return model

model = load_model()

# 2. Streamlit UI
st.title("🩺 당뇨병 예측 시뮬레이션")
st.write("아래 생체지표를 입력하면 당뇨병 위험 확률을 예측합니다.")

# 3. 사용자 입력
pregnancies = st.slider("임신 횟수", 0, 20, 1)
glucose = st.slider("혈당 수치 (Glucose)", 50, 200, 120)
blood_pressure = st.slider("혈압 (Blood Pressure)", 30, 130, 70)
skin_thickness = st.slider("피하지방 두께 (SkinThickness)", 0, 100, 20)
insulin = st.slider("인슐린 수치 (Insulin)", 0, 900, 80)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.slider("당뇨 유전 영향도 (Diabetes Pedigree Function)", 0.0, 2.5, 0.5)
age = st.slider("나이", 10, 100, 30)

# 4. 예측 버튼
if st.button("🔍 예측하기"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ 당뇨병일 가능성이 높습니다! (예측 확률: {proba * 100:.2f}%)")
    else:
        st.success(f"✅ 당뇨병일 가능성이 낮습니다. (예측 확률: {proba * 100:.2f}%)")

# 5. 참고
st.markdown("---")
st.caption("※ 본 시뮬레이션은 학습된 머신러닝 모델을 바탕으로 하며, 실제 진단은 의료 전문가의 상담을 필요로 합니다.")
