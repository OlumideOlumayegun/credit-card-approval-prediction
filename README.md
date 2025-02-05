# Credit Card Approval Prediction Using Machine Learning

![credit card](images\credit_card_banner.png)

## Overview
This project is a **machine learning-based credit card approval predictor**. The goal is to automate the decision-making process of credit card applications by training a model to predict whether an application will be **approved** or **denied** based on various applicant features. This helps commercial banks efficiently process applications while minimizing human error and effort.

## Dataset
The dataset used is the **Credit Card Approval dataset** from the **UCI Machine Learning Repository**. It contains a mix of **numerical and categorical features**, some of which are anonymized to protect applicant privacy.

## Project Structure
The project follows a structured approach:

1. **Data Inspection**: Understanding the dataset structure, types of features, and missing values.
2. **Data Preprocessing**:
   - Handling missing values using mean and mode imputation.
   - Encoding categorical variables into numerical format.
   - Feature scaling for consistency.
   - Feature selection by removing non-informative columns.
3. **Exploratory Data Analysis (EDA)**: Identifying patterns and relationships within the dataset.
4. **Model Selection and Training**:
   - Using **Logistic Regression** as the base model.
   - Splitting the dataset into training and testing sets.
5. **Model Evaluation**:
   - Assessing performance using **classification accuracy** and the **confusion matrix**.
6. **Hyperparameter Tuning**:
   - Using **GridSearchCV** to optimize model parameters for better performance.
7. **Final Model and Conclusions**: Storing the best model parameters and summarizing findings.

## Tools & Technologies
- **Python**
- **Jupyter Notebook**
- **pandas, numpy** (for data handling and preprocessing)
- **scikit-learn** (for machine learning modeling and evaluation)
- **GridSearchCV** (for hyperparameter tuning)

## Results
The final **Logistic Regression model** achieved an accuracy of **100%** on the test set, demonstrating its effectiveness in predicting credit card approvals. The model was fine-tuned using **GridSearchCV**, ensuring optimal hyperparameters.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-approval.git
   cd credit-card-approval
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and execute **credit_card_approval.ipynb** to reproduce the analysis and model training.

## Future Improvements
- Experimenting with more complex models like **Random Forests** or **XGBoost**.
- Addressing data imbalance using **SMOTE** or **cost-sensitive learning**.
- Implementing a web-based interface for real-time predictions.

## License
This project is licensed under the **MIT License**.

## Author
**Your Name**  
[LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/yourusername)

---
Feel free to contribute and enhance this project!

