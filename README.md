# Churn-Prediction
Obtaining a new customer costs significantly more than retaining an existing customer. 
Therefore, it is beneficial for companies to understand why a customer decides to leave.
Churn prediction helps businesses develop loyalty programmes and retention campaigns to retain as many customers as possible. This project focuses on creating a model that can predict which customers are likely to churn (leave the service).

# Data Description
- RowNumber: The unique identifier for each record(row)
- CustomerId: A unique identification number assigned to each customer
- Surname: The last name(surname) of the customer
- CreditScore: A numeric score representing the customer's creditworthiness
- Geography: The country where the customer resides
- Gender: The gender of the customer, identified as either "Male" or "Female"
- Age: The age of the customer
- Tenure: The number of years that the customer has been a client of the bank
- Balance: The account balance of the customer
- NumOfProducts: Number of products that a customer has
- HasCrCard: Indicates whether or not a customer has a credit card
- IsActiveMember: Indicates if the customer is an active member of the bank
- EstimatedSalary: The estimated annual salary of the customer
- Exited: Whether the customer has left the bank
- Complain: If the customer has a complaint of not
- Satisfaction Score: A score provided by the customer to rate their satisfaction with complaint resolution
- Card Type: The type of credit card the customer holds
- Points Earned: The points earned by the customer for using their credit card

# Project Structure
### 1. Data Collection
The dataset used for this project was obtained from the Kaggle Bank Customer Churn dataset which contains various features, including customer geography and account information.

### 2. Data Cleaning & Preprocessing
To prepare the data for analysis, I applied the following steps using Python libraries such  as Pandas and Numpy
- Dropping unnecessary columns: Removed irrelevant columns such as RowNumber, CustomerId, and Surname, as they did not contribute to the predictive model
- Handling missing data: Verified that no missing values were present
- Encoding: Applied Label Encoding to a categorical variable, 'Gender'

### 3. Exploratory Data Analysis (EDA)
Conducted EDA to uncover insights and patterns in the data using Python libraries Matplotlib and Seaborn

The features analysed include:
- Geography
- Gender
- Age
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Complain
- Satisfaction Score
- Card Type

For each feature, I calculated the churn rate by focusing on customers who had exited the bank and visualised the churn rates using bar charts and count chart

### 4. Feature Engineering
- Feature Encoding: Applied One-Hot Encoding to categorical variables such as 'Geography' and 'Card Type' to convert them into numerical format, making them suitable for machine learning model training, as machine learning models don't run without the variables being numerical
- Correlation Matrix: Used a correlation matrix to identify patterns and relationships within the dataset. Noticed a perfect correlation of 1 between 'Complain' and 'Exited', leading to the removal of the 'Complain' column to prevent redundancy and improve model accuracy.
- Variance Inflation Factor (VIF): Conducted VIF analysis to detect multicollinearity issues in the dataset. 'Creditscore' and 'Geography' features had VIF values above 10, indicating high multicollinearity, so they were removed to ensure model performance

### 5. Model Building

### 6. Model Evaluation


# Results
