import hashlib
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.interchange import column
from pandas.plotting import scatter_matrix
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5) ** 2 - 1
plt.scatter(plot_x[5], plot_y[5], color='r')
plt.plot(plot_x, plot_y)


def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')


def dJ(theta):
    return 2 * (theta - 2.5)


theta = 0.0
theta_history = [theta]
eta = 1.1
epsilon = 1e-8
i = 0
while i < 10:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    i += 1
    theta_history.append(theta)
    if (abs(J(theta) - J(last_theta)) < epsilon):
        break

plt.plot(plot_x, J(plot_x), color='r')
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='b', marker='x')

print(len(theta_history))

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()

housing = load_housing_data()

housing.head()

housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

housing.hist(bins=50, figsize=(20, 15))
plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train + ", len(test_set), "test")


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts() / len(housing)

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing['population'] / 100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

housing = housing.drop(columns=["ocean_proximity"])
corr_matrix = housing.corr()
corr_matrix

corr_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix