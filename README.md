ROCKVSMINE: Sonar Signal Classification using Machine Learning

📌 Project Overview

ROCKVSMINE is a supervised machine learning project that classifies underwater objects as either rock or mine based on sonar signal data. The project uses a custom sonar dataset collected from internal or simulated sources. Each data point consists of 60 numerical features representing the intensity of sonar signals reflected from objects. The final model uses Logistic Regression to make predictions and achieves good performance on unseen data.

This project demonstrates the full machine learning pipeline — from data loading and preprocessing to model training, evaluation, and prediction.


🧠 Technologies & Libraries Used

Python

Google Colab

NumPy – numerical operations

Pandas – data handling

Scikit-learn – model training & evaluation

Logistic Regression – ML algorithm used

(Optional: Matplotlib/Seaborn if you plan to visualize)

📁 Dataset Details

Source: Custom dataset ([Copy of sonar data.csv](https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view))

Shape: 208 samples × 61 columns

Features: 60 sonar signal readings per sample

Label (Target):

R → Rock

M → Mine

Each row contains sonar signal readings at different frequencies, with the 61st column indicating the object type.

