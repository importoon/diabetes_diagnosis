import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# 모델 로딩 또는 학습
@st.cache_resource
def load_model():
    try:
        with open("diabetes_model_no_preg.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # 데이터 로딩 및 전처리
        df = pd.read_csv("diabetes.csv")
        df = df.drop(columns=["Pregnancies"])  # 임신 횟수 제거
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        with open("diabetes_model_no_preg.pkl", "wb") as f:
            pickle.dump(model, f)
        return model

model = load_model()

# 사용자 인터페이스
st.title("💡 임신 변수 제외: 당뇨병 예측 시뮬레이션")
st.write("당뇨와 관련된 주요 생체 지표를 입력하면 발병 위험을 예측합니다.")

# 사용자 입력 (임신 변수 제외)
glucose = st.slider("혈당 수치 (Glucose)", 50, 200, 120)
blood_pressure = st.slider("혈압 (Blood Pressure)", 30, 130, 70)
skin_thickness = st.slider("피하지방 두께 (SkinThickness)", 0, 100, 20)
insulin = st.slider("인슐린 수치 (Insulin)", 0, 900, 80)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.slider("당뇨 유전 영향도 (Diabetes Pedigree Function)", 0.0, 2.5, 0.5)
age = st.slider("나이", 10, 100, 30)

# 예측
if st.button("🔍 예측하기"):
    input_data = np.array([[glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ 당뇨병일 가능성이 높습니다! (예측 확률: {proba * 100:.2f}%)")
    else:
        st.success(f"✅ 당뇨병일 가능성이 낮습니다. (예측 확률: {proba * 100:.2f}%)")

# 참고 메시지
st.markdown("---")
st.caption("※ 이 시뮬레이션은 임신 변수(Pregnancies)를 제외하여 성별에 관계없이 더 일반적인 분석을 지향합니다.")
