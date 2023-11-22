import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import Tuple

# If you created custom transformers or helper functions, you can also add them to this file.
class LithoEstimator:
    def __init__(self, path:str='data/log_data.csv') -> None:
        # Load the data
        df = pd.read_csv(path)

        # Initialize the LabelEncoder
        le = LabelEncoder()

        # Encode the 'munsel_color' feature
        df['munsel_color'] = le.fit_transform(df['munsel_color'])

        # Split the data into features and target variable
        X = df.drop('lithology', axis=1)
        y = df['lithology']

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])

        # Train the model
        self.model.fit(self.X_train, self.y_train)

    # Method to calculate the F1 score
    def x_test_score(self) -> np.number:
        y_pred = self.model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='macro')

    # Method to get the training and test features
    def get_Xs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_train, self.X_test

    # Method to get the training and test target variable
    def get_ys(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.y_train, self.y_test

    # Method to make predictions on new data
    def predict(self, path_to_new_file:str='data/new_data.csv') -> np.array:
#         le = LabelEncoder()
        new_samples = pd.read_csv(path_to_new_file)
        new_samples['munsel_color'] = le.transform(new_samples['munsel_color'])
        return self.model.predict(new_samples)

    # Method to get the trained model
    def get_model(self) -> Pipeline:
        return self.model
