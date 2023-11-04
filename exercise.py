import streamlit as st
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data
X = wine.data.features  # Features
y = wine.data.targets  # Target variable

# Get a list of feature column names
feature_cols = list(X.columns.values)

# Verify the number of unique classes in the target variable
num_classes = len(np.unique(y))

# Define class names based on the number of unique classes
class_names = [str(i) for i in range(num_classes)]

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% training, 30% testing

# Streamlit app
st.title("Machine Learning App")

# Sidebar with options
model_option = st.sidebar.selectbox("Select a Model", ["Decision Tree", "SVM", "Random Forest"])
if model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 1, 100, 10)

# Create and evaluate the selected model
if model_option == "Decision Tree":
    st.header("Decision Tree")

    # Create a Decision Tree classifier and fit it to the data (TRAIN DATA)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)

    # Make predictions with Decision Tree
    y_pred = clf.predict(X_test)

    # Print Decision Tree accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Calculate the Decision Tree confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

if model_option == "SVM":
    st.header("Support Vector Machine (SVM)")

    # Create an SVM classifier and fit it to the data (TRAIN DATA)
    clf = SVC(kernel="linear")  # You can choose different kernels like "linear", "rbf", etc.
    clf = clf.fit(X_train, y_train)

    # Make predictions with SVM
    y_pred = clf.predict(X_test)

    # Print SVM accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Calculate the SVM confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

if model_option == "Random Forest":
    st.header("Random Forest")

    # Create a Random Forest classifier and fit it to the data (TRAIN DATA)
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(X_train, y_train)

    # Make predictions with Random Forest
    y_pred = clf.predict(X_test)

    # Print Random Forest accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Calculate the Random Forest confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
