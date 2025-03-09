# PHASE-3-PROJECT
MODELING AND CLASSIFICATION 
## Business Understanding
Churn is a critical business problem for companies, especially those  telecommunications and other industries with recurring customer interactions. The cost of acquiring new customers typically exceeds the cost of retaining existing ones.By predicting which customers are likely to churn, the company can focus on retaining them, improving customer satisfaction, and ultimately boosting long-term profitability

## Objective

 The goal of this churn rate prediction model is to identify customers who are at risk of leaving the service (churning), 
 enabling the business to take proactive actions such as targeted marketing or personalized offers to retain them.




## Key Metrics to Track:
 - **Churn Rate**: Percentage of customers who leave in a given period.

- **Customer Lifetime Value (CLV)**: How valuable a customer is over the long term.

 - **Retention Rate**: The percentage of customers retained over a given period.
### Import Libaries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize


df= pd.read_csv(r"C:\Users\thion\Downloads\CHURN RATE.csv")

df.head()
## Data Understanding & Cleaning
df.info()
5 columns  have categorical variables rest are numerical
df.describe()
df.isnull().mean()
No missing values thus no need to procced with handling missing data 
df[df.duplicated()].count()
No dupliacted values 
df_droped = df.copy()

df_droped.drop(columns=['phone number'], inplace=True)
df_droped.head()
Droped unnessecary columns 
plt.figure(figsize=(8, 6))
sns.countplot(data=df_droped, x='churn')  
plt.title('Distribution of Churn')
plt.xlabel('Churn (True: Churned, False: Not Churned)')
plt.ylabel('Count')
plt.show()
y_True = df["churn"][df["churn"] == True]
print ("Churn Percentage = "+str( (y_True.shape[0] / df["churn"].shape[0]) * 100 ))
grouped_table_state = df_droped.groupby(["state", "churn"]).size().unstack()

grouped_table_state["churn_percentage"] = (grouped_table_state[True] / (grouped_table_state[True] + grouped_table_state[False])) * 100

grouped_table_state = grouped_table_state.sort_values(by="churn_percentage", ascending=False)

grouped_table_state

grouped_table_state["churn_percentage"].plot(kind='bar', figsize=(18, 12))
plt.title('Churn Percentage by State')
# Churn rate by international plan
plt.figure(figsize=(8, 6))
sns.countplot(data=df_droped, x='international plan', hue='churn')
plt.title('Churn Rate by International Plan')
plt.xlabel('International Plan')
plt.ylabel('Count')
plt.show()

# Churn rate by voice mail plan
plt.figure(figsize=(8, 6))
sns.countplot(data=df_droped, x='voice mail plan', hue='churn')
plt.title('Churn Rate by Voice Mail Plan')
plt.xlabel('Voice Mail Plan')
plt.ylabel('Count')
plt.show()
## Churn corelation with other features 
churn_corr = df_droped.corr()['churn'].sort_values(ascending=False)

churn_corr
### Business Implications
1. Customer service calls are the biggest churn predictor.
2. Customers with high day-minute usage are more likely to churn
3. International call users are slightly more loyal.
4.  Voicemail users are less likely to churn.
plt.figure(figsize=(12, 8))
churn_corr = df_droped.corr()['churn'].sort_values(ascending=False)
sns.barplot(x=churn_corr.index, y=churn_corr.values)
plt.title('Correlation of Features with Churn')
plt.xlabel('Feature')
plt.ylabel('Correlation with Churn')
plt.xticks(rotation=90)
plt.show()
## Checking  for outilers and handling them 


z_scores = np.abs(df_droped.select_dtypes(include=['number']).apply(zscore))

outliers_zscore = (z_scores > 3).sum()
print("\nOutliers detected using Z-Score:\n", outliers_zscore)

#### Highly Skewed Features (Extreme Outliers)
Total Intl Calls (50 outliers)

Customer Service Calls (35 outliers)

Total Intl Minutes (22 outliers)

Total Intl Charge (22 outliers)
#### Features with extreme values 
Total Day Minutes, Total Eve Minutes, Total Night Minutes (9, 9, and 11 outliers)

Total Day Charge, Total Eve Charge, Total Night Charge (9, 9, and 11 outliers)

Total Calls (Day, Eve, Night) → 9, 7, 6 outliers

df_outiler = df_droped.copy()


 #Capping extreme values
df_outiler['total intl calls'] = winsorize(df_outiler['total intl calls'], limits=[0, 0.05])  
df_outiler['customer service calls'] = winsorize(df_outiler['customer service calls'], limits=[0, 0.05])

#Log Transformation skewed data
log_transform_cols = ['total intl minutes', 'total intl charge', 'total day minutes', 'total day charge',
                      'total eve minutes', 'total eve charge', 'total night minutes', 'total night charge']

for col in log_transform_cols:
    df_outiler[col] = np.log1p(df_outiler[col])  # log1p avoids log(0) issues

# Robust Scaling For features with high variance
scaler = RobustScaler()
scale_cols = ['total intl calls', 'customer service calls']

df_outiler[scale_cols] = scaler.fit_transform(df_outiler[scale_cols])


df_outiler.head()
## Feature Engineering and Encoding 
### Encoding 

df_engineered = df_outiler.copy()

df_engineered= pd.get_dummies(df_engineered, columns=['state'], drop_first=True)

df_engineered['international plan'] = df_engineered['international plan'].map({'yes': 1, 'no': 0})
df_engineered['voice mail plan'] = df_engineered['voice mail plan'].map({'yes': 1, 'no': 0})

df_engineered.head()
### Feature engineering 

# 1. International Usage Feature
df_engineered['intl usage'] = df_engineered['total intl minutes'] + df_engineered['total intl calls']

# 2. International to Total Minutes Ratio (Handling Zero Division)
df_engineered['intl to total minutes ratio'] = np.where(
    (df_engineered['total day minutes'] + df_engineered['total eve minutes'] + df_engineered['total night minutes'] + df_engineered['total intl minutes']) == 0,
    0,
    df_engineered['total intl minutes'] / (df_engineered['total day minutes'] + df_engineered['total eve minutes'] + df_engineered['total night minutes'] + df_engineered['total intl minutes'])
)

# 3. Average Call Duration (Handling Zero Division)
df_engineered['avg call duration'] = np.where(
    (df_engineered['total day calls'] + df_engineered['total eve calls'] + df_engineered['total night calls']) == 0,
    0,
    (df_engineered['total day minutes'] + df_engineered['total eve minutes'] + df_engineered['total night minutes']) /
    (df_engineered['total day calls'] + df_engineered['total eve calls'] + df_engineered['total night calls'])
)

# 4. Total Call Count
df_engineered['total calls'] = df_engineered['total day calls'] + df_engineered['total eve calls'] + df_engineered['total night calls']

# 5. Total Call Duration
df_engineered['total minutes'] = df_engineered['total day minutes'] + df_engineered['total eve minutes'] + df_engineered['total night minutes']

# 6. Day vs Night Call Duration Ratio (Handling Zero Division)
df_engineered['day night ratio'] = np.where(
    df_engineered['total night minutes'] == 0,
    0,
    df_engineered['total day minutes'] / df_engineered['total night minutes']
)

# 7. Voice Mail Usage
df_engineered['voice mail activity'] = df_engineered['number vmail messages']

# 8. Voice Mail Plan Usage (Binary Encoding)
df_engineered['vmail plan usage'] = df_engineered['voice mail plan'].apply(lambda x: 1 if x == 'yes' else 0)

# 9. Total Charge
df_engineered['total charge'] = df_engineered['total day charge'] + df_engineered['total eve charge'] + df_engineered['total night charge'] + df_engineered['total intl charge']

# 10. Charge-to-Call Ratio (Handling Zero Division)
df_engineered['charge per call'] = np.where(
    df_engineered['total calls'] == 0,
    0,
    df_engineered['total charge'] / df_engineered['total calls']
)

# 11. Average Charge per Minute (Handling Zero Division)
df_engineered['avg charge per minute'] = np.where(
    df_engineered['total minutes'] == 0,
    0,
    df_engineered['total charge'] / df_engineered['total minutes']
)

# 12. Customer Service Calls Behavior
df_engineered['service calls'] = df_engineered['customer service calls']

# 13. High Spender Feature (Using Median Charge as Threshold)
threshold = df_engineered['total charge'].median()
df_engineered['high spender'] = (df_engineered['total charge'] > threshold).astype(int)



df_engineered.head()

### Further EDA with featutred engineered

average_revenue_per_customer = df_engineered['total charge'].mean()

# Calculate the churn rate
churn_rate = df_engineered['churn'].mean()

clv = average_revenue_per_customer / churn_rate

print(f"Customer Lifetime Value (CLV): {clv}")
high_spenders = df_engineered[df_engineered['high spender'] == 1]
churn_rate_high_spenders = high_spenders['churn'].mean() * 100

print(f"Churn Rate Percentage for High Spenders: {churn_rate_high_spenders:.2f}%")
plt.figure(figsize=(12, 8))
sns.countplot(data=df_engineered, x='high spender', hue='churn')
plt.title('Churn Rate for High Spenders')
plt.xlabel('High Spender (1: Yes, 0: No)')
plt.ylabel('Count')
plt.show()
## Model Training
#convert churn column to numerical values
df_engineered['churn'] = df_engineered['churn'].astype(int)
X = df_engineered.drop(columns=['churn'])
y = df_engineered['churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Models trained 
Logistic model 
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

logreg_pred = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
print(f"Accuracy: {logreg_accuracy}")
conf_matrix_logreg = confusion_matrix(y_test, logreg_pred)
sns.heatmap(conf_matrix_logreg, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
Random Forest 
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_accuracy}")
conf_matrix_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix for Random Forest")
plt.show()
Support Vector Machine 
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"Accuracy: {svm_accuracy}")
conf_matrix_svm = confusion_matrix(y_test, svm_pred)
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix for SVM")
plt.show()
Decision Tree 
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Accuracy: {dt_accuracy}")
conf_matrix_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix for Decision Tree")
plt.show()
K- Nearest Neighbour 
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"Accuracy: {knn_accuracy}")
conf_matrix_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix for KNN")
plt.show()

## Evalution of perfomanace Model 

logreg_report = classification_report(y_test, logreg_pred, output_dict=True)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)
dt_report = classification_report(y_test, dt_pred, output_dict=True)
knn_report = classification_report(y_test, knn_pred, output_dict=True)

summary_table = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN'],

    'Accuracy': [logreg_accuracy, rf_accuracy, svm_accuracy, dt_accuracy, knn_accuracy],

    'Precision': [logreg_report['1']['precision'], rf_report['1']['precision'], svm_report['1']['precision'], dt_report['1']['precision'], knn_report['1']['precision']],

    'Recall': [logreg_report['1']['recall'], rf_report['1']['recall'], svm_report['1']['recall'], dt_report['1']['recall'], knn_report['1']['recall']],

    'F1-Score': [logreg_report['1']['f1-score'], rf_report['1']['f1-score'], svm_report['1']['f1-score'], dt_report['1']['f1-score'], knn_report['1']['f1-score']]
})

print(summary_table)

summary_table['Average'] = summary_table[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean(axis=1)
best_model_index = summary_table['Average'].idxmax()
best_model_name = summary_table.loc[best_model_index, 'Model']

print(f"The best model based on all metrics is: {best_model_name}")
### Predict high risk customer 
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

#set_threshold = 0.8

high_risk_threshold = 0.8
high_risk_customers = X_test[rf_probabilities > high_risk_threshold]


high_risk_customers_df = pd.DataFrame(high_risk_customers, columns=X.columns)


high_risk_customers_df['phone number'] = df.loc[high_risk_customers_df.index, 'phone number']
high_risk_customers_df['churn_probability'] = rf_probabilities[rf_probabilities > high_risk_threshold]


high_risk_customers_df[['phone number', 'churn_probability']].sort_values(by='churn_probability', ascending=False).head(50).reset_index(drop=True)
churn_probabilities = rf_model.predict_proba(X_test)[:, 1]

#thereshold for churn @ 0.5
churn_threshold = 0.5
churned_customers = X_test[churn_probabilities > churn_threshold]


churned_customers_df = pd.DataFrame(churned_customers, columns=X.columns)


churned_customers_df['phone number'] = df.loc[churned_customers_df.index, 'phone number']
churned_customers_df['churn_probability'] = churn_probabilities[churn_probabilities > churn_threshold]



churned_customers_df[['phone number', 'churn_probability']].sort_values(by='churn_probability', ascending=False).head(50).reset_index(drop=True)
### Conclusion:
1. **Model Performance**: Of the models examined, the Random Forest model performed the best, with an accuracy of 95.65%.   This demonstrates that the Random Forest model is quite effective in predicting customer attrition.

2. **Key Predictors**: The investigation found that customer service calls, total day minutes, and total charge are all important predictors of turnover.  Customers that have a high day-minute use and make numerous customer service calls are more likely to churn.

3. **Feature Engineering**: The additional features engineered, such as international usage, average call duration, and total charge, contributed to improving the model's performance.

### Recommendations:
1. **Focus on Customer Service**: Since customer service calls are a significant predictor of churn, improving customer service quality and reducing the number of calls can help in retaining customers. Implementing proactive customer service measures can address issues before they lead to churn.

2. **Targeted Marketing**: Use the churn prediction model to identify high-risk customers and implement targeted marketing strategies, such as personalized offers and discounts, to retain them.

3. **Monitor High Usage Customers**: Customers with high day-minute usage and total charges are more likely to churn. Monitoring these customers and providing them with tailored plans or incentives can help in reducing churn.

4. **Continuous Model Improvement**: Regularly update and retrain the model with new data to ensure its accuracy and effectiveness. Incorporate feedback from marketing and customer service teams to refine the model further.

5. **Customer Feedback**: Collect and analyze customer feedback to identify common pain points and address them promptly. This can help in improving customer satisfaction and reducing churn.
