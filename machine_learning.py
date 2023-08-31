from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import os, pprint, pickle, shutil
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import tree
from sklearn.svm import SVC
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
    # print(classification_report(y_teste, previsoes))

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

        files = ["credit.pkl", "census.pkl", "risco_credito.pkl"]

        for file in files:
            with open(file, mode="wb") as f:
                pickle.dump(
                    [self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste], f)
        shutil.move(f'{os.getcwd()}/{file}', f'{os.getcwd()}/data/my_progress/{file}')

        self.data_base = {
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

        self.list_methods = {
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(criterion="entropy", random_state=0),
            "random_forest": RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0),
            "knn": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
            "logistic_regression": LogisticRegression(random_state=1),
            "svm": SVC(kernel="linear", random_state=1, C=1.0),
            "rede_neural": MLPClassifier(max_iter=1500, verbose=True, tol=0.000010, solver='adam', activation='relu',
                                         hidden_layer_sizes=(2, 2))
            # ^ ((data_base[db]["teste"]["X"].shape+1)/2) ^
        }

    def treinner(self):
        previsores = {
            "credit": ["age", "workclass", "final-weight", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loos", "hour-per-week", "native-country"],
            "census": ['income', 'age', 'loan']
        }

        for method in self.list_methods:
            methodExecutable = self.list_methods[method]
            for db in self.data_base:
                methodExecutable.fit(self.data_base[db]["treinamento"]["X"], self.data_base[db]["treinamento"]["y"])
                previsoes = methodExecutable.predict(self.data_base[db]["teste"]["X"])
                print(f"\n{method}")
                statistics(method, db, methodExecutable, self.data_base[db]["treinamento"]["X"], self.data_base[db]["teste"]["X"],
                           self.data_base[db]["treinamento"]["y"], self.data_base[db]["teste"]["y"], previsoes, previsores[db])

    def parameter_settings(self):

        parameters_methods = {
            "decision_tree": {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 10]
            },
            "random_forest": {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [10,40,100,150],
                'min_samples_split': [2,5,10],
                'min_samples_leaf': [1,5,10]
            },
            "knn": {
                'n_neighbors': [3,5,10,20],
                'p': [1,2]
            },
            "logistic_regression": {
                'tol': [0.001, 0.0001, 0.00001],
                'C': [1.1, 1., 2.0],
                'solver': ['lbfgs', 'sag', 'saga']
            },
            "svm": {
                'tol': [0.001, 0.0001, 0.00001],
                'C': [1.1, 1., 2.0],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
            },
            "rede_neural": {
                'activation': ['relu', 'logistic', 'tahn'],
                'solver': ['adam', 'sgd'],
                'batch_size': [10, 56],
            }
        }

        for method in self.list_methods:
            for db in self.data_base:
                X_credit = np.concatenate((self.data_base[db]["treinamento"]["X"], self.data_base[db]["teste"]["X"]), axis=0)
                y_credit = np.concatenate((self.data_base[db]["treinamento"]["y"], self.data_base[db]["teste"]["y"]), axis=0)
                if method != "naive_bayes":
                    grid_search = GridSearchCV(estimator=self.list_methods[method], param_grid=parameters_methods[method])
                    grid_search.fit(X_credit, y_credit)
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    print(f"\n{method} {db}")
                    print(best_params, best_score)

    def cross_validation(self):
        results = {
            "decision_tree": list(),
            "random_forest": list(),
            "knn": list(),
            "logistic_regression": list(),
            "svm": list(),
            "rede_neural": list(),
        }
        for method in self.list_methods:
            for db in self.data_base:
                X_credit = np.concatenate((self.data_base[db]["treinamento"]["X"], self.data_base[db]["teste"]["X"]),axis=0)
                y_credit = np.concatenate((self.data_base[db]["treinamento"]["y"], self.data_base[db]["teste"]["y"]),axis=0)
                for i in range(30):
                    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
                    match method:
                        case "decision_tree":
                            treinner_method = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2,
                                                                     min_samples_split=2, splitter='best')
                        case "random_forest":
                            treinner_method = RandomForestClassifier(criterion='entropy', min_samples_leaf=1,
                                                                     min_samples_split=5, n_estimators=10)
                        case "knn":
                            treinner_method = KNeighborsClassifier()
                        case "logistic_regression":
                            treinner_method = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
                        case "svm":
                            treinner_method = SVC(kernel='rbf', C=2.0)
                        case "rede_neural":
                            treinner_method = MLPClassifier(activation='relu', batch_size=56, solver='adam')
                    scores = cross_val_score(treinner_method, X_credit, y_credit)
                    results[method].append(scores.mean())
                    print(i)
        resultados  = pd.DataFrame({
            'Decision Tree': results['decision_tree'],
            'Random Forest': results['random_forest'],
            'KNN': results['knn'],
            'Logistic Regression': results['logistic_regression'],
            'SVM': results['svm'],
            'Rede Neural': results['rede_neural']
        })
        print(resultados.describe())


def launcher():
    machine_learning = MachineLearning()
    # machine_learning.treinner()
    # machine_learning.parameter_settings()
    machine_learning.cross_validation()

launcher()