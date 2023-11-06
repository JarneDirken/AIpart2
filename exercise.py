import streamlit as st
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data
X = wine.data.features  # Features
y = wine.data.targets  # Target variable

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% training, 30% testing

# Streamlit app
st.title("Machine Learning Model Comparison")

# Sidebar with options
model_option = st.sidebar.selectbox("Select a Model", ["Random Forest", "SVM", "K-Nearest Neighbors (KNN)"])
if model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 1, 100, 10)

# Create and evaluate the selected model
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

if model_option == "K-Nearest Neighbors (KNN)":
    st.header("K-Nearest Neighbors (KNN)")

    # Create a K-Nearest Neighbors (KNN) classifier and fit it to the data (TRAIN DATA)
    clf = KNeighborsClassifier(n_neighbors=5)  # You can specify the number of neighbors
    clf = clf.fit(X_train, y_train)

    # Make predictions with KNN
    y_pred = clf.predict(X_test)

    # Print KNN accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Calculate the KNN confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
