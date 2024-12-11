import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils import confidence_interval


class Test:
    def __init__(self) -> None:
        self.train_size = 0.8
        self.random_state = 100
        self.N = 100

        self.model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                   activation='tanh',
                                   alpha=0.1,
                                   max_iter=1000,
                                   random_state=self.random_state)
        # early_stopping reduces accuracy !?

    def run(self, X, y):
        result = []
        for _ in range(self.N):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
            self.model.fit(X_train, y_train)
            result.append(self.model.score(X_test, y_test))
        # print(result)
        # print(np.mean(result))
        print(confidence_interval(result))
