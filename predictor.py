import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model and test data
model = joblib.load('XGBoost.pkl')
X_test = pd.read_csv('X_test.csv')

# Define feature names
feature_names = [
    "SLC6A13", "ANLN", "MARCO", "SYT13", "ARG2", "MEFV", "ZNF29P",
    "FLVCR2", "PTGFR", "CRISP2", "EME1", "IL22RA2", "SLC29A4",
    "CYBB", "LRRC25", "SCN8A", "LILRA6", "CTD_3080P12_3", "PECAM1"
]

st.title("Prediction of the Risk of Non-Small Cell Lung Cancer Based on the Expression Levels of Diabetes-Related Genes.")

# Create input widgets for each feature
SLC6A13 = st.number_input("SLC6A13", min_value=0, max_value=100000, value=161)
ANLN = st.number_input("ANLN", min_value=0, max_value=100000, value=161)
MARCO = st.number_input("MARCO", min_value=0, max_value=100000, value=1439)
SYT13 = st.number_input("SYT13", min_value=0, max_value=100000, value=12)
ARG2 = st.number_input("ARG2", min_value=0, max_value=100000, value=3224)
MEFV = st.number_input("MEFV", min_value=0, max_value=100000, value=34)
ZNF29P = st.number_input("ZNF29P", min_value=0, max_value=100000, value=1)
FLVCR2 = st.number_input("FLVCR2", min_value=0, max_value=100000, value=654)
PTGFR = st.number_input("PTGFR", min_value=0, max_value=100000, value=24)
CRISP2 = st.number_input("CRISP2", min_value=0, max_value=100000, value=44)
EME1 = st.number_input("EME1", min_value=0, max_value=100000, value=495)
IL22RA2 = st.number_input("IL22RA2", min_value=0, max_value=100000, value=12)
SLC29A4 = st.number_input("SLC29A4", min_value=0, max_value=100000, value=913)
CYBB = st.number_input("CYBB", min_value=0, max_value=100000, value=1629)
LRRC25 = st.number_input("LRRC25", min_value=0, max_value=100000, value=288)
SCN8A = st.number_input("SCN8A", min_value=0, max_value=100000, value=714)
LILRA6 = st.number_input("LILRA6", min_value=0, max_value=100000, value=113)
CTD_3080P12_3 = st.number_input("CTD_3080P12_3", min_value=0, max_value=100000, value=1)
PECAM1 = st.number_input("PECAM1", min_value=0, max_value=100000, value=5020)

# Store the input values in a list
feature_values = [
    SLC6A13, ANLN, MARCO, SYT13, ARG2, MEFV, ZNF29P, FLVCR2, PTGFR,
    CRISP2, EME1, IL22RA2, SLC29A4, CYBB, LRRC25, SCN8A, LILRA6,
    CTD_3080P12_3, PECAM1
]
feature = pd.DataFrame([feature_values], columns=feature_names)

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    # Get the predicted class and probabilities
    predicted_class = model.predict(feature)[0]
    predicted_proba = model.predict_proba(feature)[0]
    
    # Display the results
    st.write(f"**Predicted Class:** {predicted_class} (1: Tumor, 0: Normal)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    
    # Provide advice based on the predicted class
    if predicted_class == 1:
        advice = ("According to our model, we're sorry to tell you that you're at high risk of having non - small cell lung cancer. Please contact a professional doctor for a thorough check - up as soon as possible. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    else:
        advice = ("According to our model, we're glad to inform you that your risk of non - small cell lung cancer is low. But if you feel unwell, consult a professional doctor. Wish you good health. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    st.write(advice)
    
    # SHAP analysis
    st.subheader("SHAP Force Plot Explanation")
    
    # Create a SHAP explainer
    explainer_shap = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer_shap.shap_values(feature)
    
    # Ensure we get the correct SHAP values for the predicted class
    class_idx = int(predicted_class)  # define class_idx here
    
    if isinstance(shap_values, list):
        shap_value = shap_values[class_idx]
    else:
        shap_value = shap_values
    
    # Create a DataFrame for the input feature
    input_df = pd.DataFrame(feature_values, index=feature_names).T
    
    # Determine the expected value based on the predicted class
    if isinstance(explainer_shap.expected_value, (list, np.ndarray)):
        expected_value = explainer_shap.expected_value[class_idx]
    else:
        expected_value = explainer_shap.expected_value
    
    # Create a subheader that reflects the class being explained
    class_name = "Tumor" if class_idx == 1 else "Normal"
    st.subheader(f"SHAP Force Plot for Class {class_name}")
    
    # Plot the force_plot for the predicted class
    shap.force_plot(
        expected_value,
        shap_value,
        input_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()