from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import os, pprint, pickle, shutil
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import tree
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

def statistics(method, data_base, trainer, x_treinamento, x_teste, y_treinamento, y_teste, previsoes, previsores):
    cm = ConfusionMatrix(trainer)
    cm.fit(x_treinamento, y_treinamento)

    print(f"\n{method}_{data_base}")
    ac_store = cm.score(x_teste, y_teste)*100
    print("accuracy_score: %.2f" % ac_store, "%")
    pprint.pprint(f"confusion_matrix: {confusion_matrix(y_teste, previsoes)}")
    print("classification_report: ")
    print(classification_report(y_teste, previsoes))

    try:
        names_class = list()
        for name_class in trainer.classes_:
            names_class.append(str(name_class))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        fig.savefig(f'{data_base}_{method}.png')
        shutil.move(f'{os.getcwd()}//{data_base}_{method}.png', f'{os.getcwd()}/data/imagens/{data_base}_{method}.png')
    except Exception as e:
        print(str(e), e.args)

class MachineLearning():
    def __init__(self):
        # if not f'{os.getcwd()}/data/my_progress/census.pkl' and not f'{os.getcwd()}/data/my_progress/credit.pkl':
        # load_csv_files
        base_credit = pd.read_csv(f'{os.getcwd()}/data/my_progress/credit_data.csv')
        base_census = pd.read_csv(f'{os.getcwd()}/data/my_progress/census.csv')

        # treatment_data_credit
        base_credit.loc[base_credit["age"] < 0, "age"] = base_credit["age"][base_credit["age"] > 0].mean()
        base_credit["age"].fillna(base_credit["age"].mean(), inplace=True)
        scaler_credit = StandardScaler()
        X_credit = scaler_credit.fit_transform(base_credit.iloc[:, 1:4].values)
        y_credit = base_credit.iloc[:, 4].values

        # treatment_data_cencus_LabelEncoder
        fit_transform = [1, 3, 5, 6, 7, 8, 9, 13]
        X_census = base_census.iloc[:, 0:14].values
        y_census = base_census.iloc[:, 14].values
        for index in range(14):
            if index in fit_transform:
                X_census[:, index] = LabelEncoder().fit_transform(X_census[:, index])
        # treatment_data_census_OneHotEncoder
        onehotencoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), fit_transform)],
                                                 remainder="passthrough")
        X_census = onehotencoder_census.fit_transform(X_census).toarray()
        # scaling
        X_census = StandardScaler().fit_transform(X_census)

        # training
        self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste = \
            train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)
        self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste = \
            train_test_split(X_census, y_census, test_size=0.15, random_state=0)

        # treatment_data_cencus_LabelEncoder
        base_risco_credito = pd.read_csv(f'{os.getcwd()}/data/my_progress/risco_credito.csv')

        self.X_risco_credito = base_risco_credito.iloc[:, 0:4].values
        self.y_risco_credito = base_risco_credito.iloc[:, 4].values
        for index in range(4):
            self.X_risco_credito[:, index] = LabelEncoder().fit_transform(self.X_risco_credito[:, index])

        with open('credit.pkl', mode="wb") as f:
            pickle.dump([self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste], f)
        shutil.move(f'{os.getcwd()}/credit.pkl', f'{os.getcwd()}/data/my_progress/credit.pkl')

        with open('census.pkl', mode="wb") as f:
            pickle.dump([self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste], f)
        shutil.move(f'{os.getcwd()}//census.pkl', f'{os.getcwd()}/data/my_progress/census.pkl')

        with open('risco_credito.pkl', mode="wb") as f:
            pickle.dump([self.X_risco_credito, self.y_risco_credito], f)
        shutil.move(f'{os.getcwd()}//risco_credito.pkl', f'{os.getcwd()}/data/my_progress/risco_credito.pkl')
        # else:
        #     with open(f'{os.getcwd()}/data/my_progress/credit.pkl', 'rb') as f:
        #         self.X_credit_treinamento, self.y_credit_treinamento, self.X_credit_teste, self.y_credit_teste = pickle.load(f)
        #
        #     with open(f'{os.getcwd()}/data/my_progress/credit.pkl', 'rb') as f:
        #         self.X_census_treinamento, self.y_census_treinamento, self.X_census_teste, self.y_census_teste = pickle.load(f)
        #
        #     with open(f'{os.getcwd()}/data/my_progress/risco_credito.pkl', 'rb') as f:
        #         self.X_risco_credito, self.y_risco_credito = pickle.load(f)

    def treinner(self):
        list_methods = {
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(criterion="entropy", random_state=0),
            "random_forest": RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0),
            "knn":  KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
            "logistic_regression": LogisticRegression(random_state=1),
            # "svm": "",
        }

        data_base = {
            "credit": {
                "treinamento": {
                    "X": self.X_credit_treinamento,
                    "y": self.y_credit_treinamento
                },
                "teste": {
                    "X": self.X_credit_teste,
                    "y": self.y_credit_teste
                },
            },
            "census": {
                "treinamento": {
                    "X": self.X_census_treinamento,
                    "y": self.y_census_treinamento
                },
                "teste": {
                    "X": self.X_census_teste,
                    "y": self.y_census_teste
                },
            }
        }

        previsores = {
            "credit": ["age", "workclass", "final-weight", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loos", "hour-per-week", "native-country"],
            "census": ['income', 'age', 'loan']
        }

        for method in list_methods:
            methodExecutable = list_methods[method]
            for db in data_base:
                methodExecutable.fit(data_base[db]["treinamento"]["X"], data_base[db]["treinamento"]["y"])
                previsoes = methodExecutable.predict(data_base[db]["teste"]["X"])
                statistics(method, db, methodExecutable, data_base[db]["treinamento"]["X"], data_base[db]["teste"]["X"],
                           data_base[db]["treinamento"]["y"], data_base[db]["teste"]["y"], previsoes, previsores[db])


def launcher():
    machine_learning = MachineLearning()
    machine_learning.treinner()

launcher()