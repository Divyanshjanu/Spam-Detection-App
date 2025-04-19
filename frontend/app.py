# app.py

import streamlit as st
import requests
import plotly.express as px
import pandas as pd

#  Set BACKEND URL to your deployed backend on Cloud Run
BACKEND_URL = "https://app-backend-1027897761252.northamerica-northeast1.run.app"

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title(" Spam Detector")

# User Input
user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter a message!")
    else:
        with st.spinner("Validating and Analyzing..."):
            try:
                model_choice = "XGBoost"  # Using XGBoost

                # Step 1: Validate
                validate_response = requests.post(f"{BACKEND_URL}/validate", json={"text": user_input, "model_choice": model_choice})
                validate_result = validate_response.json()

                if validate_result.get("valid", False):
                    st.success("Input is valid!")

                    # Step 2: Predict
                    pred_response = requests.post(f"{BACKEND_URL}/predict", json={"text": user_input, "model_choice": model_choice})
                    pred_result = pred_response.json()

                    if "prediction" in pred_result:
                        st.subheader("Prediction Result:")
                        st.write(f"**{pred_result['prediction'].upper()}** ({pred_result['probability']}%)")

                        # Step 3: Explain
                        explain_response = requests.post(f"{BACKEND_URL}/explain", json={"text": user_input, "model_choice": model_choice})
                        explain_result = explain_response.json()

                        if "feature_names" in explain_result and "shap_values" in explain_result:
                            shap_df = pd.DataFrame({
                                "Feature": explain_result["feature_names"],
                                "SHAP Value": explain_result["shap_values"]
                            }).sort_values("SHAP Value", ascending=False).head(20)

                            fig = px.scatter(
                                shap_df,
                                x="SHAP Value",
                                y="Feature",
                                color="SHAP Value",
                                color_continuous_scale="RdBu",
                                title=" Feature Importance (SHAP Beeswarm Plot)",
                                size_max=10
                            )

                            fig.update_layout(
                                title_font_size=24,
                                xaxis_title="SHAP Value",
                                yaxis_title="Feature",
                                font=dict(size=14),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Couldn't generate SHAP Beeswarm plot.")

                    else:
                        st.error(f"Prediction failed: {pred_result.get('error', 'Unknown error')}")
                else:
                    st.error(" Input failed validation! Please enter a proper message.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
