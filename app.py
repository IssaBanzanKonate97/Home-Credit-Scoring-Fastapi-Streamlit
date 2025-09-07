# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Scoring")

st.title("Scoring")
st.caption("FastAPI + Streamlit")

api_url = st.text_input("URL de l'API", "http://127.0.0.1:8000")

st.subheader("Variables d'entrée")
ext1 = st.number_input("EXT_SOURCE_1", 0.0, 1.0, 0.50, step=0.01)
ext2 = st.number_input("EXT_SOURCE_2", 0.0, 1.0, 0.50, step=0.01)
ext3 = st.number_input("EXT_SOURCE_3", 0.0, 1.0, 0.50, step=0.01)
days_birth = st.number_input("DAYS_BIRTH (positif = ancienneté en jours)", 0.0, 30000.0, 14000.0, step=1.0)
amt_credit = st.number_input("AMT_CREDIT", 0.0, 5_000_000.0, 200_000.0, step=1000.0)

if st.button("Prédire"):
    payload = {
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "DAYS_BIRTH": days_birth,
        "AMT_CREDIT": amt_credit
    }
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        r.raise_for_status()
        prob = r.json().get("prob_default")
        st.success(f"Probabilité de défaut : **{prob:.3f}**")
        st.progress(min(max(prob, 0.0), 1.0))
    except Exception as e:
        st.error(f"Erreur d'appel API : {e}")
