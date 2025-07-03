import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import joblib
import os

# -----------------------------
# Data generation and model training
# -----------------------------
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    size = np.random.normal(1500, 500, 200)
    price = size * 100 + np.random.normal(0, 10000, 200)
    df = pd.DataFrame({'size_sqft': size, 'price': price})

    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, df, rmse, r2

# -----------------------------
# Streamlit App Interface
# -----------------------------
def main():
    st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
    st.title("🏠 Simple House Price Predictor")
    st.markdown("Enter the house size to predict its sale price, and see how it fits into overall pricing trends.")

    # Train model and fetch data
    model, df, rmse, r2 = generate_and_train_model()

    # User input
    size = st.slider("📏 Enter house size (sqft)", 500, 5000, 1500)

    if st.button("🔮 Predict Price"):
        prediction = model.predict([[size]])[0]
        st.success(f"💰 Estimated Price: **${prediction:,.2f}**")

        st.subheader("📊 Model Evaluation")
        st.write(f"• RMSE: **{rmse:,.2f}**")
        st.write(f"• R² Score: **{r2:.2f}**")

        # Plotting
        st.subheader("📈 House Size vs Price Visualization")
        fig = px.scatter(df, x='size_sqft', y='price', title="Size vs Price (Synthetic Market Data)")
        fig.add_scatter(x=[size], y=[prediction], mode='markers',
                        marker=dict(color='red', size=12),
                        name="Your Prediction")
        st.plotly_chart(fig, use_container_width=True)

        # Export model
        joblib.dump(model, "house_price_model.pkl")
        with open("house_price_model.pkl", "rb") as f:
            st.download_button("📥 Download Trained Model", f, file_name="house_price_model.pkl")

if __name__ == "__main__":
    main()
