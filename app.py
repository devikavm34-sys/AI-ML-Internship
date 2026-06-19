import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/iris_model.joblib")

st.title("🌸 Iris Flower Classifier")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

if st.button("Predict"):

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    prediction = model.predict(data)
    probabilities = model.predict_proba(data)[0]

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(
        f"Predicted Species: {species[prediction[0]]}"
    )

    st.subheader("Confidence Scores")

    for i in range(len(species)):
        st.write(
            f"{species[i]}: {probabilities[i]*100:.2f}%"
        )
        st.progress(float(probabilities[i]))