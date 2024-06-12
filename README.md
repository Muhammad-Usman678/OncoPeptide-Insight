# OncoPeptide Insight

OncoPeptide Insight is a machine learning project designed to analyze peptide sequences associated with breast and lung cancer. The project utilizes a RandomForest classifier to predict the activity of peptides and provide valuable insights into their potential roles in cancer research.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Peptides play a significant role in cancer biology. OncoPeptide Insight aims to classify peptides as either active or inactive using a machine learning approach. By leveraging the RandomForest algorithm, the project seeks to provide accurate predictions and enhance our understanding of peptide activity in cancer.

## Datasets
The project uses two primary datasets from Kaggle:
1. **Breast Cancer Dataset**: Contains peptide sequences and their activity status related to breast cancer.
2. **Lung Cancer Dataset**: Contains peptide sequences and their activity status related to lung cancer.


## Installation
To run this project, you need to have Python and the necessary libraries installed. Follow these steps to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/Muhammad-Usman678/OncoPeptide-Insight
    cd OncoPeptideInsight
    ```

2. Install the required libraries:
    ```bash
    pip install pandas==2.0.3
    pip install scikit-learn==1.2.2
    ```

## Usage
1. Load the datasets:
    ```python
    import pandas as pd

    breast_cancer_data = pd.read_csv('ACPs_Breast_cancer.csv')
    lung_cancer_data = pd.read_csv('ACPs_Lung_cancer.csv')
    ```

2. Train the RandomForest classifier:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Example code for training the classifier
    X = breast_cancer_data.drop(columns=['class'])
    y = breast_cancer_data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    ```

3. Evaluate the model on the lung cancer dataset:
    ```python
    # Example code for evaluation
    X_lung = lung_cancer_data.drop(columns=['class'])
    y_lung = lung_cancer_data['class']

    y_lung_pred = clf.predict(X_lung)
    print(classification_report(y_lung, y_lung_pred))
    ```

## Results
The project provides classification reports for both breast and lung cancer datasets, showcasing the performance of the RandomForest classifier. The reports include metrics such as precision, recall, and F1-score.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements or suggestions.

## License
This project is licensed under the MIT License. 
