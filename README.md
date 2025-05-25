# 💧 Water Quality Prediction Using Ensemble Learning

This project uses machine learning ensemble methods—**Bagging** and **Stacking**—to predict key water quality parameters in **Taal Lake**. It compares the performance of these two ensemble approaches to determine which model better forecasts the values of environmental indicators over time.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features Predicted](#features-predicted)
- [Algorithms Used](#algorithms-used)
- [Output Metrics](#output-metrics)
- [Dependencies](#dependencies)
- [Author](#author)
- [License](#license)

---

## 🧠 Overview

Water quality plays a crucial role in monitoring environmental health. By using historical water quality data, this project:
- Trains models using historical monthly data.
- Compares the performance of two ensemble methods: **Voting Regressor with Bagging** vs. **Stacking Regressor**.
- Evaluates performance using multiple metrics and a **paired t-test**.

---

## 🔬 Features Predicted

The following six water quality parameters are predicted:

- 🌡️ Water Temperature  
- 🧪 pH  
- 🧫 Ammonia  
- 🧬 Nitrate  
- 🧂 Phosphate  
- 💧 Dissolved Oxygen  

---

## 🤖 Algorithms Used

- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **Bagging Regressor** (used in VotingRegressor)
- **Voting Regressor** (ensemble of bagged models)
- **Stacking Regressor** with Linear Regression as meta-model

---

## 📊 Output Metrics

Each model evaluation prints the following metrics on the validation set:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Mean Squared Log Error (MSLE)
- Explained Variance

Additionally, the comparison script performs paired t-tests to check statistical significance between Voting and Stacking performance.

---

## 📦 Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

---

## 👨‍💻 Author
Mark Jezreel Antivo

BS Computer Engineering, Cavite State University – Main Campus

📧 markantivo50@gmail.com

---

## 📄 License
This project is created for academic purposes and research. You are welcome to use and reference it for educational and non-commercial use.