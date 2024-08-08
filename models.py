# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Model class in which all various classifier algo's be residing!


class Model:

    def __init__(self):  # constructor
        self.name = ''  # name of which model we will use!
        # path for dataset
        path = 'dataset/depressionDataset.csv'
        df = pd.read_csv(path)
        # dropping unnecessary columns
        df = df[['q1', 'q2', 'q3', 'q4', 'q5',
                 'q6', 'q7', 'q8', 'q9', 'q10', 'class']]

        # Handling Missing Data
        df['q1'] = df['q1'].fillna(df['q1'].mode()[0])
        df['q2'] = df['q2'].fillna(df['q2'].mode()[0])
        df['q3'] = df['q3'].fillna(df['q3'].mode()[0])
        df['q4'] = df['q4'].fillna(df['q4'].mode()[0])
        df['q5'] = df['q5'].fillna(df['q5'].mode()[0])
        df['q6'] = df['q6'].fillna(df['q6'].mode()[0])
        df['q7'] = df['q7'].fillna(df['q7'].mode()[0])
        df['q8'] = df['q8'].fillna(df['q8'].mode()[0])
        df['q9'] = df['q9'].fillna(df['q9'].mode()[0])
        df['q10'] = df['q10'].fillna(df['q10'].mode()[0])
        df['class'] = df['class'].fillna(df['class'].mode()[0])
        self.split_data(df)
    # this function will split our data

    def split_data(self, df):
        x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values
        y = df.iloc[:, 10].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=24)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    # support vector classifier

    def svm_classifier(self):
        self.name = 'Svm Classifier'
        classifier = SVC()
        return classifier.fit(self.x_train, self.y_train)
    # using confuion metrix to get accuracy

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        accuracy = (cm[0][0] + cm[1][1]) / \
            (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        print(f"{self.name} has accuracy of {accuracy *100} % ")

    # decision tree
    def decision_tree_classifier(self):
        self.name = 'Decision Tree'
        classifier = DecisionTreeClassifier()
        return classifier.fit(self.x_train, self.y_train)


if __name__ == '__main__':
    model = Model()
    model.accuracy(model.svm_classifier())
    model.accuracy(model.decision_tree_classifier())
