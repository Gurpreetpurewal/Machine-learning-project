import pandas as pd
import numpy as np
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import silhouette_score

Data = pd.read_csv("brain.csv")

#First problem statement:
'''exploratory data analysis'''

# Checking If there is any null values in dataset

print(Data.describe())
print(type(Data))

#Checking if the row having missing values grater than 100
data_null_coiumns = Data[Data.isnull().sum(axis=1) > 100]
print(data_null_coiumns)

'''print(Data.dtypes)'''

x = Data.isnull().sum()
y = (Data.isnull().sum()/Data.shape[0])*100
z = {'Number of missing values':x,'percentage of missing values':y}
df = pd.DataFrame(z, columns=['Number of missing values','percentage of missing values'])
df.sort_values(by='percentage of missing values',ascending=False)
print(df)


# Apply boxplotting to check the outliers

plt.rcParams['figure.figsize']=(14,6)

plt.subplot(2,4,1)
sns.boxplot(x=Data['hypertension'],y=Data['age'])

plt.subplot(2,4,2)
sns.boxplot(x=Data['heart_disease'],y=Data['age'])

plt.subplot(2,4,3)
sns.scatterplot(y=Data['bmi'],x=Data['age'])

plt.subplot(2,4,4)
sns.scatterplot(x=Data['avg_glucose_level'],y=Data['age'])

plt.subplot(2,4,5)
sns.boxplot(x=Data['smoking_status'],y=Data['age'])
plt.suptitle("Checking the outliers")
plt.show()

#handling the missing values.

Data['age'] = Data['age'].fillna(Data['age'].mode()[0])
Data['hypertension'] = Data['hypertension'].fillna(Data['hypertension'].mode()[0])
Data['heart_disease'] = Data['heart_disease'].fillna(Data['heart_disease'].mode()[0])
Data['work_type'] = Data['work_type'].fillna(Data['work_type'].mode()[0])
Data['Residence_type'] = Data['Residence_type'].fillna(Data['Residence_type'].mode()[0])
Data['ever_married'] = Data['ever_married'].fillna(Data['ever_married'].mode()[0])
Data['avg_glucose_level'] = Data['avg_glucose_level'].fillna(Data['avg_glucose_level'].mode()[0])

print("Check if there is any missing value after the using mode fuction in columns")

x = Data.isnull().sum()
y = (Data.isnull().sum()/Data.shape[0])*100
z = {'Number of missing values':x,'percentage of missing values':y}
df = pd.DataFrame(z, columns=['Number of missing values','percentage of missing values'])
df.sort_values(by='percentage of missing values', ascending=False)
print(df)

'''
Data['heart_disease'].values[Data['heart_disease'].values>1]=1
print(Data[Data['heart_disease']>1])
'''

#Apply label encoding to convert categorical value to numerical value

le = LabelEncoder() 

Data['gender'] = le.fit_transform(Data['gender'])
Data['ever_married'] = le.fit_transform(Data['ever_married'])
Data['work_type'] = le.fit_transform(Data['work_type'])
Data['Residence_type'] = le.fit_transform(Data['Residence_type'])
Data['smoking_status'] = le.fit_transform(Data['smoking_status'])

print(Data.head) 



feature_sel = ['gender', 'age', 'hypertension', 'heart_disease',
                      'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level',
                      'bmi','smoking_status','stroke']

# Extract the selected features from the dataset
selected_df = Data[feature_sel]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

# Set labels and title
plt.title('Online Fraud Dataset - Correlation Matrix Heatmap')

# Show the plot
plt.show()


# CASE 1 Predicting stroke occurrence by examining the health condition using various health factors that help to identify the possible cause. Therefore, some classification models like GaussianNB model, decision tree, Naive Bayesian model and K-nearest Neighbor. However, I am using GaussianNB and confusion metrics for model accuracy and prediction. 

x = Data.iloc[:,:-1].values     # taking all values excepted last col or target value
y = Data.iloc[:,-1].values      # taking target values

#Divide the dataset  in x(train & test) and y(train & test)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=42)

print("x train",x_train.shape)
print("y train",y_train.shape)
print("x test",x_test.shape)
print("y test",y_test.shape)

#Using GaussianNB model to predict the model accuracy 

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test) 

print("Gaussian Naive Bayes model accuracy(in %):",metrics.accuracy_score(y_test,y_pred)*100)

#import scikit learn matrics module for accuracy calculation: how often is the clasifier correct?

cn = confusion_matrix(y_pred,y_test)
sns.heatmap(cn,annot=True, fmt=".2f", cmap="cividis")
plt.show()
print("Gaussion Classifiation result")
print(" ")
print(classification_report(y_test, y_pred))


# CASE 2 I will use K-means clustering on the dataset, making at least 2-3 clusters to check whether an individual has experienced strokes or not. Considering variables like age, hypertension, gender, work type, BMI, glucose level, and heart disease. 

'''Using K-means clustering to check weather person have stroke or not.'''
x1 = Data.iloc[:,[7,8,9,10]].values
y1 = Data.iloc[:,-1].values
wcss = []

for i in range(1, 11):
    # to check the params of KMeans and the meaning for each one
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)

#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.4, random_state=1)

kmeans_2 = KMeans(n_clusters = 2, init = 'k-means++')
kmeans_2.fit(x_train,y_train)
y_pred=kmeans_2.predict((x_test))

#confuse matrix
cn = confusion_matrix(y_pred,y_test)
sns.heatmap(cn,annot=True, fmt=".2f", cmap="cividis")

plt.xlabel('Predicted values by the classifier')
plt.ylabel('Represents the true values')
plt.show()

print(accuracy_score(y_pred, y_test)*100)
print(classification_report(y_pred,y_test))

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x1)

#generated cluster
print(y_kmeans)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D Scatter plot for each cluster
for cluster_label in np.unique(y_kmeans):
    ax.scatter(x1[y_kmeans == cluster_label, 0],
               x1[y_kmeans == cluster_label, 1],
               x1[y_kmeans == cluster_label, 2],
               s=10, label = f'Cluster {cluster_label + 1}')

ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
           s=100, c='black', marker='o', label='Centroids')

ax.set_title('3D Scatter Plot of Clusters and Centroids')

# Add legend
ax.legend()

# Show the plot
plt.show()

# Assuming 'X1' is your data and 'kmeans.labels_' are your cluster labels from K-means
silhouette_avg = silhouette_score(x1, kmeans.labels_)

print("The average silhouette_score is :", silhouette_avg)



# CASE 3 creating variable called heath index which combines the values of glucose level, BMI, and linear classification of heart disease.this help to understand the patterns and trends we can observe weather people with certain occupations or living in certain area tend to better or worse health.'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Feature scaling

numerical_features = ['avg_glucose_level', 'bmi']
scaler = StandardScaler()
Data[numerical_features] = scaler.fit_transform(Data[numerical_features])

# Split the data into features and target

X = Data[['avg_glucose_level', 'bmi']]
y = Data['heart_disease']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
Data['heart_disease_prediction'] = model.predict(Data[['avg_glucose_level', 'bmi']])

# Define weights for each feature in the health index

glucose_weight = 0.4
bmi_weight = 0.3
heart_disease_weight = 0.3

# Create the health index

Data['health_index'] = (
    glucose_weight * Data['avg_glucose_level'] +
    bmi_weight * Data['bmi'] +
    heart_disease_weight * Data['heart_disease_prediction'])

# Display the resulting DataFrame with the health index

print(Data[['avg_glucose_level', 'bmi', 'heart_disease_prediction', 'health_index']])


from sklearn.metrics import mean_squared_error
# Make predictions on the test set

y_pred = model.predict(X_test)

# Convert predicted values to binary labels (0 or 1)

y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Display the classification report

classification_rep = classification_report(y_test, y_pred_binary)
print("Classification Report:\n", classification_rep)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

# Evaluate the model using Mean Squared Error (MSE)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([0, 1], [0, 1], linestyle='-', color='red', linewidth=2)  # Diagonal line for reference
plt.xlabel('Actual Brain Stock')
plt.ylabel('Predicted Brain Stock')
plt.title('Linear Regression for Bain strokes detection - Scatter Plot')
plt.show()







