ROCKVSMINE: Sonar Signal Classification using Machine Learning

----------------------------------------------------------------------------------------------------------------------------------------------

📌 Project Overview

ROCKVSMINE is a supervised machine learning project that classifies underwater objects as either rock or mine based on sonar signal data. The project uses a custom sonar dataset collected from internal or simulated sources. Each data point consists of 60 numerical features representing the intensity of sonar signals reflected from objects. The final model uses Logistic Regression to make predictions and achieves good performance on unseen data.

This project demonstrates the full machine learning pipeline — from data loading and preprocessing to model training, evaluation, and prediction.

-------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Technologies & Libraries Used:

Python

Google Colab

NumPy – numerical operations

Pandas – data handling

Scikit-learn – model training & evaluation

Logistic Regression – ML algorithm used

(Optional: Matplotlib/Seaborn if you plan to visualize)

-------------------------------------------------------------------------------------------------------------------------------------------------


📁 Dataset Details:

Source: Custom dataset ([Copy of sonar data.csv](https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view))

Shape: 208 samples × 61 columns

Features: 60 sonar signal readings per sample

Label (Target):

R → Rock

M → Mine

Each row contains sonar signal readings at different frequencies, with the 61st column indicating the object type.

-------------------------------------------------------------------------------------------------------------------------------------------------

🧪 Project Workflow:

1.DATA COLLECTION & PREPROCESSING:

Loaded CSV using pandas.read_csv()

Inspected data using .head(), .shape, and .describe()

Ensured label distribution is approximately balanced

<img width="1485" height="108" alt="image" src="https://github.com/user-attachments/assets/113bad80-39c9-4276-b4b0-b33d9f97e036" />

<img width="364" height="106" alt="image" src="https://github.com/user-attachments/assets/ecce392d-1ed0-4247-8021-3914fdd0d90c" />

<img width="1861" height="373" alt="image" src="https://github.com/user-attachments/assets/8e4d95bc-f066-40a9-b934-a0528005de1b" />



2.EXPLORATORY DATA ANALYSIS:

Checked value counts:

Rocks (R) = 97

Mines (M) = 111

Used .groupby().mean() to inspect feature averages by class

<img width="1854" height="226" alt="image" src="https://github.com/user-attachments/assets/c4bde864-7cd9-4502-b389-ab807c05c5d4" />



3.Feature-Label Separation:

    X = sonar_data.drop(columns=60, axis=1)

    Y = sonar_data[60]

X: 60 sonar features

Y: Classification label

<img width="783" height="652" alt="image" src="https://github.com/user-attachments/assets/add452ef-7783-4790-8239-e1657ee573a8" />



4.Train-Test Splitting:

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

Training samples: 187

Test samples: 21

Stratification ensures class balance

<img width="865" height="116" alt="image" src="https://github.com/user-attachments/assets/6349aa6c-3512-48ae-9c6f-d2a86cfc765e" />



5.Model Training (Logistic Regression):

    model = LogisticRegression()

    model.fit(X_train, Y_train)

Trained Logistic Regression model using default parameters



6.Model Evaluation:

    accuracy_score(Y_train, model.predict(X_train))  # ~83.4%

    accuracy_score(Y_test, model.predict(X_test))    # ~76.1%

Training accuracy: ~83.42%

Testing accuracy: ~76.19%

<img width="687" height="73" alt="image" src="https://github.com/user-attachments/assets/236e8f3c-1963-4f55-8fb2-784e36e00df1" />



7.Making Predictions:

    input_data = ( ... )  # Example sonar signal readings

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    prediction = model.predict(input_data_as_numpy_array)

    if prediction[0] == 'R':

        print("The object is Rock")

    else:

        print("The object is Mine")

<img width="1861" height="331" alt="image" src="https://github.com/user-attachments/assets/f50b55ca-b8e0-43f1-981a-3f199e759292" />

-------------------------------------------------------------------------------------------------------------------------------------------------

📌 Model Insights:

Performs well on unseen data

Shows minimal overfitting (small accuracy drop from train to test)

Ideal for basic classification tasks in signal-based detection

-------------------------------------------------------------------------------------------------------------------------------------------------

🏁 How to Run

1.Clone this repo:
    
    git clone https://github.com/yourusername/ROCKVSMINE.git

    cd ROCKVSMINE

2.Install required packages:
   
    pip install -r requirements.txt

3.Run the notebook:

Open the .ipynb file in Jupyter Notebook or Google Colab


-------------------------------------------------------------------------------------------------------------------------------------------------


🔮 Future Improvements
Try other models like:

Support Vector Machine (SVM)

Random Forest

KNN

Perform cross-validation for more robustness

Visualize confusion matrix and ROC curve

-------------------------------------------------------------------------------------------------------------------------------------------------

🙌 Acknowledgment

Custom dataset prepared for this project

Tools used: Google Colab, Python, Scikit-learn

-------------------------------------------------------------------------------------------------------------------------------------------------


📬 Contact

For questions or suggestions, feel free to reach out via or open an issue.
