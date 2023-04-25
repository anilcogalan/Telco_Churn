# Telco Churn Prediction

<p align="center">
  <img src="https://user-images.githubusercontent.com/61653147/234227901-d03114e6-493f-4389-9e83-f551d91b1b38.png" width=25% height=25%>
</p>

Telco Churn Prediction refers to the process of predicting which customers of a telecommunications company are likely to cancel their services in the near future. This can be achieved by analyzing various customer data points, such as demographics, usage patterns, and customer interactions, in order to identify factors that contribute to churn.

By leveraging machine learning algorithms, Telco Churn Prediction models can be developed to analyze customer data and make accurate predictions about which customers are most likely to churn. This information can then be used to proactively reach out to customers who are at risk of leaving and offer them incentives to stay, ultimately reducing customer churn and increasing customer retention rates.

The Telco Churn Prediction project typically involves:
* Data collection
* Data cleaning
* Exploratory data analysis 
* Feature engineering 
* Model selection
* Model evaluation

The final outcome of the project is a predictive model that can be used to identify at-risk customers and take proactive steps to retain them.

## Business Problem 

It is expected to develop a machine learning model that can predict customers who will leave the company.

## Dataset Story

| Column Name     | Description                                                                               |
| --------------- | ----------------------------------------------------------------------------------------- |
| CustomerId      | Customer ID                                                                               |
| Gender          | Gender of the customer                                                                    |
| SeniorCitizen   | Whether the customer is a senior citizen or not (1, 0)                                    |
| Partner         | Whether the customer has a partner or not (Yes, No)                                       |
| Dependents      | Whether the customer has dependents or not (Yes, No)                                      |
| tenure          | Number of months the customer has stayed with the company                                 |
| PhoneService    | Whether the customer has a phone service or not (Yes, No)                                  |
| MultipleLines   | Whether the customer has multiple lines or not (Yes, No, No phone service)                 |
| InternetService | Customer's internet service provider (DSL, Fiber optic, No)                                |
| OnlineSecurity  | Whether the customer has online security or not (Yes, No, No internet service)             |
| OnlineBackup    | Whether the customer has online backup or not (Yes, No, No internet service)               |
| DeviceProtection| Whether the customer has device protection or not (Yes, No, No internet service)           |
| TechSupport     | Whether the customer has tech support or not (Yes, No, No internet service)                |
| StreamingTV     | Whether the customer has streaming TV or not (Yes, No, No internet service)                |
| StreamingMovies | Whether the customer has streaming movies or not (Yes, No, No internet service)            |
| Contract        | The contract term of the customer (Month-to-month, One year, Two year)                     |
| PaperlessBilling| Whether the customer has paperless billing or not (Yes, No)                                |
| PaymentMethod   | The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card) |
| MonthlyCharges  | The amount charged to the customer monthly                                                |
| TotalCharges    | The total amount charged to the customer                                                  |
| Churn           | Whether the customer has churned or not (Yes or No) 


