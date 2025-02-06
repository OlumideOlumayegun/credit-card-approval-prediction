# Credit Card Approval Prediction Using Machine Learning

![credit card](images/credit_card_banner.png)

## Overview
This project is a **machine learning-based credit card approval predictor**. The goal is to automate the decision-making process of credit card applications by training a model to predict whether an application will be **approved** or **denied** based on various applicant features. This helps commercial banks efficiently process applications while minimising human error and effort.

For an in-depth exploration of the credit card approval prediction model, please consult the accompanying [Jupyter Notebook](notebook.ipynb). The notebook provides a comprehensive, reproducible workflow that demonstrates how I developed and validated the credit card approval prediction system using machine learning techniques.

## Dataset
The dataset used is the **Credit Card Approval dataset** from the **UCI Machine Learning Repository**. It contains a mix of **numerical and categorical features**, some of which are anonymised to protect applicant privacy.

## Project Structure
The project follows a structured approach:

1. **Data Inspection**: Understanding the dataset structure, types of features, and missing values.
2. **Data Preprocessing**:
   - Handling missing values using mean and mode imputation.
   - Encoding categorical variables into numerical format.
   - Feature scaling for consistency.
   - Feature selection by removing non-informative columns.
3. **Model Selection and Training**:
   - Splitting the dataset into training and testing sets.
   - Using **Logistic Regression** as the base model.
4. **Model Evaluation**:
   - Assessing performance using **classification accuracy** and the **confusion matrix**.
6. **Hyperparameter Tuning**:
   - Using **GridSearchCV** to optimise model parameters for better performance.
7. **Final Model and Conclusions**: Storing the best model parameters and summarising findings.

## Tools & Technologies
- **Python**
- **Jupyter Notebook**
- **pandas, numpy** (for data handling and preprocessing)
- **scikit-learn** (for machine learning modeling and evaluation)
- **GridSearchCV** (for hyperparameter tuning)

## Results
The final **Logistic Regression model** achieved an accuracy of **83%** on the test set, demonstrating its effectiveness in predicting credit card approvals. The model was fine-tuned using **GridSearchCV**, ensuring optimal hyperparameters.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/OlumideOlumayegun/credit-card-approval-prediction.git
   cd credit-card-approval-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and execute **notebook.ipynb** to reproduce the analysis and model training.

## Future Improvements
- Experimenting with more complex models like **Random Forests** or **XGBoost**.
- Addressing data imbalance using **SMOTE** or **cost-sensitive learning**.
- Implementing a web-based interface for real-time predictions.

## License
This project is licensed under the **MIT License**.

## Author
**Olumide Olumayegun**  
[LinkedIn](https://www.linkedin.com/in/olumide-olumayegun-phd-fhea-miet-r-eng-coren-5a844127/?originalSubdomain=uk) | [GitHub](https://github.com/OlumideOlumayegun)

---
Feel free to contribute and enhance this project!

