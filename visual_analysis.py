import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from mlwrap import get_test, get_train, push_test
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler

x_train = pd.read_csv("x_train.csv", delimiter=";", dtype=np.float32)
y_train = pd.read_csv("y_train.csv", delimiter=";", header=None,
                      names=["returned"], dtype=np.float32)
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.read_csv("x_test.csv", delimiter=";")

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

#x = np.random.normal(size=100)
#sns.distplot(x);
#sns.plt.show()

#sns.pairplot(train_data, vars=["maxPlayerLevel",
                               #"doReturnOnLowerLevels",
                               #"totalScore"], hue="returned", dropna=True)
#sns.plt.show()

#print(x_train[:3])
#print()
#print(y_train[:3])
#print()
#print(train_data[:3])

#print(train_data.groupby("maxPlayerLevel")["returned"].value_counts(normalize=True))

describe_fields = ["maxPlayerLevel", "numberOfAttemptedLevels", "attemptsOnTheHighestLevel",
                   "totalNumOfAttempts", "averageNumOfTurnsPerCompletedLevel"]

#print("===== train: males")
#print(train_data[describe_fields].describe())

parameters = ["maxPlayerLevel",
              "numberOfAttemptedLevels",
              "attemptsOnTheHighestLevel",
              "totalNumOfAttempts",
              "averageNumOfTurnsPerCompletedLevel",
              "doReturnOnLowerLevels",
              "numberOfBoostersUsed",
              "fractionOfUsefullBoosters",
              "totalScore",
              "totalBonusScore",
              "totalStarsCount",
              "numberOfDaysActuallyPlayed",
]

scaler = StandardScaler()
scaler.fit(train_data[parameters])
train_data_scaled = scaler.transform(train_data[parameters])
test_data_scaled = scaler.transform(test_data[parameters])


selector = SelectPercentile(f_classif)
selector.fit(np.array(train_data_scaled), np.array(train_data["returned"]))

print(train_data_scaled[:3])
print(train_data["returned"][:3])
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(parameters)), scores)
plt.xticks(range(len(parameters)), parameters, rotation='vertical')
#plt.show()