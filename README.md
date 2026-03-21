# 🏠 Mumbai House Price Prediction

A machine learning project that predicts house prices in Mumbai using real-world housing data. The model takes inputs like location, area, and number of bedrooms to estimate a property's price.

---

##  Introduction

Housing prices in Mumbai vary significantly depending on location, size, and other factors. For buyers and sellers, estimating a fair price is often difficult without proper data analysis.

This project aims to solve that by using machine learning to predict house prices based on real-world data.

---

##  Problem Statement

Property valuation is complex because:

* Prices depend on multiple factors
* Data is often inconsistent or incomplete
* Manual estimation can be inaccurate

The goal is to build a model that can learn from historical data and provide reliable price predictions.

---

##  Objectives

* Predict house prices with good accuracy
* Work with real-world housing data
* Identify important features affecting price
* Compare multiple models and choose the best one
* Provide an easy interface for users to get predictions

---

##  Model Overview

Different regression models were tested during development, including:

* Linear Regression
* Decision Tree
* Random Forest

After comparison, **LightGBM** was selected as the final model due to better performance.

---

##  Dataset

The dataset is based on **real-world housing data from Mumbai**, including features such as:

* Location
* Area (in square feet)
* Number of bedrooms (BHK)
* Price

### Preprocessing steps:

* Removed missing and inconsistent data
* Handled outliers
* Encoded location data
* Performed feature scaling where required

---

##  Implementation Pipeline

1. Load and clean dataset
2. Perform exploratory data analysis (EDA)
3. Feature engineering and encoding
4. Train-test split
5. Train multiple models
6. Compare performance
7. Select best model (LightGBM)
8. Make predictions
9. Build Streamlit interface

---

##  Evaluation Metrics

The model was evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

### Final Result:

* **R² Score: ~0.8 on test data**

This indicates the model explains a significant portion of variance in housing prices.

---

##  Challenges Faced

**1. Data quality issues**
Real-world data had missing values and inconsistencies
→ Cleaned and filtered dataset

**2. Handling location feature**
Location is a key factor but categorical
→ Used encoding techniques

**3. Outliers in price and area**
Extreme values affected model performance
→ Removed unrealistic entries

**4. Model selection**
Different models gave different results
→ Compared multiple models and selected LightGBM

---

##  Results

* LightGBM performed best among tested models
* Achieved R² score of ~0.8 on test data
* Predictions are reasonably close to actual prices
* Model generalizes well on unseen data

---

##  Streamlit App

A simple web interface was built using **Streamlit** to make predictions easily.

### Run the app:

```bash
streamlit run app.py
```

Users can input property details and get predicted prices instantly.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib / Seaborn
* Streamlit

---



---

##  Future Improvements

* Add more features (amenities, property age, etc.)
* Improve UI design
* Deploy Streamlit app online
* Use larger datasets for better accuracy

---



---

## ⭐

If you found this useful, feel free to star the repo.
