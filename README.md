# Credit EDA using Machine Learning
<img width="1000" height="601" alt="image" src="https://github.com/user-attachments/assets/a324848d-6295-4805-9028-3efdb4ff472a" />

Credit Card Default Risk Analysis is the process of identifying and predicting whether a credit card customer is likely to default on their payment obligations in the upcoming billing cycle. This project leverages exploratory data analysis (EDA), machine learning, and deep learning techniques to analyze historical customer data, uncover behavioral patterns, and build predictive models that assess default risk. The system helps financial institutions make informed credit decisions, minimize financial losses, and improve risk management strategies.

### Project Objective:

The primary objective of this project is to design and implement an intelligent data-driven system that accurately predicts credit card default risk. The specific goals include:

- To analyze customer demographic, billing, repayment, and credit limit data using Exploratory Data Analysis (EDA).
- To identify important factors that influence credit card default behavior.
- To build and compare multiple predictive models, including Logistic Regression, Random Forest, and Deep Learning (Neural Networks).
- To handle class imbalance and improve prediction reliability using appropriate evaluation metrics.
- To provide a scalable and reusable framework suitable for real-world financial risk assessment applications.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ace1abc-918d-481f-a086-abc502073545" alt="EDA Output">
</p>

### Modules Used:

- __Data Loading Module__
     - Loads the dataset directly from a CSV file.
     - Ensures compatibility with real-world datasets obtained from platforms such as Kaggle.

- __Data Cleaning & Preprocessing Module__
     - Removes duplicate records and irrelevant columns.
     - Handles outliers using robust statistical techniques.
     - Scales numerical features to improve model performance.

- __Exploratory Data Analysis (EDA) Module__
     - Visualizes class imbalance between default and non-default customers.
     - Generates correlation heatmaps to identify relationships among features.
     - Helps in understanding customer behavior patterns before modeling.

- __Feature Engineering Module__
     - Separates independent features and target variables.
     - Prepares data for machine learning and deep learning models.

- __Machine Learning Module__
     - Implements Logistic Regression as a baseline model.
     - Uses Random Forest for advanced non-linear pattern detection.
     - Evaluates models using accuracy, ROC-AUC, confusion matrix, and classification reports.

- __Deep Learning Module__
     - Builds a Neural Network using TensorFlow and Keras.
     - Applies dropout and early stopping to reduce overfitting.
     - Compares deep learning performance with traditional ML models.

- __Model Evaluation & Comparison Module__
     - Compares different models based on performance metrics.
     - Identifies the most reliable model for credit default prediction.

- __Model Persistence Module__
     - Saves trained models and scalers for future inference.
     - Enables easy integration with deployment systems.

### Installation:
To run this project locally, follow these steps:

- __Clone the repository:__ <br>


         git clone https://github.com/Shaikh-Aiman/Credit-EDA-using-Machine-Learning


- __Navigate to the project directory:__
cd Credit-EDA-using-Machine-Learning
- __Ensure you have Python installed on your system.__
- __Install the required dependencies.__
- __Run the application:__
python main.py

### Dataset Details

The dataset used in this project is the Default of Credit Card Clients Dataset, obtained from Kaggle and originally published by the UCI Machine Learning Repository. It contains real-world financial and demographic data of credit card users, making it suitable for credit risk analysis and predictive modeling.
[Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download)

__Dataset Advantages__
- No missing values
- Fully numeric (no encoding required)
- Widely used in academic research and industry case studies
- Suitable for EDA, machine learning, and deep learning models

### Working:

The Credit Card Default Risk Analysis System follows a structured pipeline to ensure accurate prediction and interpretability.

- __Data Loading:__
The dataset is directly loaded from a CSV file using Pandas, ensuring simplicity and transparency in data handling.

- __Data Cleaning & Preprocessing:__
    - Removal of irrelevant identifiers (e.g., ID column)
    - Duplicate record elimination
    - Outlier handling using robust statistical clipping
    - Feature scaling for consistent model performance

- __Exploratory Data Analysis (EDA):__
    - Visual analysis of default vs non-default distribution
    - Correlation heatmap generation to identify influential features
    - Helps in understanding class imbalance and feature relationships

- __Feature Preparation:__
    - Separation of independent variables and target variable
    - Stratified train-test split to preserve class distribution

- __Model Training:__
    - Logistic Regression for baseline linear classification
    - Random Forest for capturing complex non-linear patterns
    - Neural Network for deep feature learning and comparison

- __Model Evaluation:__
    - Accuracy
    - ROC-AUC score
    - Confusion matrix
    - Classification report

- __Model Persistence:__
    - Trained models and scalers are saved for future prediction and deployment
    - Enables easy integration with GUI or real-time systems
 
### Results & Performance Comparison:

The performance of different predictive models was evaluated using standard classification metrics. Since credit default prediction is a class-imbalanced problem, greater emphasis was placed on ROC-AUC, precision, and recall rather than accuracy alone.

- __Evaluation Metrics Used__
- Accuracy: Overall correctness of predictions.
- Precision: Ability to correctly identify defaulters.
- Recall: Ability to capture actual defaulters.
- ROC-AUC: Model’s ability to distinguish between defaulters and non-defaulters.

### Observations:

- Logistic Regression provides stable and interpretable results, making it suitable for baseline risk assessment.
- Random Forest consistently outperforms other models by capturing complex feature interactions and handling class imbalance effectively.
- Neural Networks show competitive performance but require careful tuning and are more computationally expensive.
- ROC-AUC proved to be the most reliable metric for evaluating model effectiveness in this credit risk context.

### Conclusion:
This project successfully demonstrates the application of Exploratory Data Analysis, Machine Learning, and Deep Learning techniques to predict credit card default risk. Through systematic preprocessing and robust modeling approaches, the system is able to identify key behavioral and financial factors influencing default decisions.

The comparative analysis of Logistic Regression, Random Forest, and Neural Network models highlights the importance of model selection in financial risk prediction. The use of advanced evaluation metrics ensures reliable and unbiased performance assessment, especially in the presence of class imbalance.

Overall, the system provides a scalable, accurate, and practical solution for credit risk assessment, making it suitable for academic projects, real-world financial applications, and further extensions such as deployment through graphical interfaces or web-based platforms.
