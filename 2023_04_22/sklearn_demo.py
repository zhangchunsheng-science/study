from math import floor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_LENGTH = 100


def load_data():
    np.random.seed(0)
    X = np.random.rand(DATA_LENGTH, 1) * 30 + 50
    noise = np.random.rand(DATA_LENGTH, 1) * 50
    y = X * 8 - 127
    y = y - noise
    return X, y


def visualization_data(X, y):
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.scatter(X, y, c='black')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.tight_layout()  # 调整子图间距
    plt.show()


def visualization_model(X, y, X_test, y_pre):
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.scatter(X, y, c='black')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    #
    plt.plot(X_test, y_pre, c='black')
    plt.tight_layout()  # 调整子图间距
    plt.show()


def train(X, y):
    test_len = floor(DATA_LENGTH * 0.2)
    index = -1 * test_len
    X_train = X[:index]
    X_test = X[index:]

    y_train = y[:index]
    y_test = y[index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    return X, y, X_test, y_pred


if __name__ == '__main__':
    x, y = load_data()
    visualization_data(x, y)
    x, y, X_test, y_pre = train(x, y)
    visualization_model(x, y, X_test, y_pre)
