import kagglehub
import pandas as pd
import numpy as np
import os

path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

print("Path to dataset files:", path)


# Kaggle-version tiedostonimet ovat päinvastaiset suhteessa
# UNSW:n ilmoittamaan viralliseen train/test-jakoon
train_path = os.path.join(path, "UNSW_NB15_testing-set.csv")
test_path = os.path.join(path, "UNSW_NB15_training-set.csv" )

train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)



y_train = train_df["label"]
X_train = train_df.drop(columns=["label"])

y_test = test_df["label"]
X_test = test_df.drop(columns=["label"])


X_train = X_train.select_dtypes(include=[np.number])
X_train = X_train.replace([np.inf, -np.inf], np.nan)


X_test = X_test.select_dtypes(include=[np.number])
X_test = X_test.replace([np.inf, -np.inf], np.nan)

train_medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


print(X_test.columns)




print(X_train.columns)
print(y_train)

X_train.to_csv("X_train_clean.csv", index=False)
y_train.to_csv("y_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_test.to_csv("y_test_clean.csv", index=False)