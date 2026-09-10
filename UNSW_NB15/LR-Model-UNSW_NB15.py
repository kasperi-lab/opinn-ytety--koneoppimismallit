# Halutaanko käyttää parannettua mallia vai baseline mallia (parametrit)
identifier = input("baseline vai parannettu:").strip().lower()
if identifier != "baseline" and identifier != "parannettu":
    raise ValueError("Valitse baseline tai parannettu!")

# Halutaanko ajossa toteuttaa parametrihaku, eli parhaiden parametrien etsintä
parameter_search_state = input("Parametrihaku valinta (Y/N): ").strip().upper()
if parameter_search_state == "Y":
    parameter_search_state = True
elif parameter_search_state == "N":
    parameter_search_state = False
else:
    raise ValueError("Valitse parametrihaun käyttö: Y tai N.")


# Halutaanko poistaa määritellyt piirteet mallin parantamiseksi
from Piirteiden_poisto_lista import features_to_be_removed

feature_removal = input("Poistetaanko piirteitä? (Y/N): ").strip().upper()
if feature_removal not in ["Y", "N"]:
    raise ValueError("Valitse Y tai N.")




import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Luetaan aineistot.")

X_train = pd.read_csv("X_train_clean.csv")
y_train = pd.read_csv("y_train_clean.csv").squeeze()

X_test = pd.read_csv("X_test_clean.csv")
y_test = pd.read_csv("y_test_clean.csv").squeeze()

X_test = X_test[X_train.columns]

if feature_removal == "Y":
    X_train = X_train.drop(columns=[c for c in features_to_be_removed if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in features_to_be_removed if c in X_test.columns])

print("Skaalaus: ")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



print("Luodaan malli.")


if identifier == "baseline":
    lr = LogisticRegression(
        random_state=42,
        C=1,
        max_iter=2000,
        solver="lbfgs"
    )
else:
    lr = LogisticRegression(
        random_state=42,
        C=2.0,
        max_iter=2000,
        class_weight=None,
        penalty="l2",
        solver="lbfgs"
    )

print("Mallin opetus: ")



lr.fit(X_train_scaled, y_train)


# Ennuste
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)


# Mittarit
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)


TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

fn_rate = FN / (FN + TP)
recall = TP / (TP + FN)


print(f"\nLogistic Regression {identifier} results")
print("---------------------------")

print("Accuracy:", acc)
print("ROC-AUC:", roc)
print("PR-AUC:", pr_auc)

print("\nConfusion Matrix:")
print(cm)

print("\nFalse-negative ratio:", fn_rate)
print("Recall ratio:", recall)

print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Piirteiden poisto:", feature_removal)
print("Piirteiden määrä:", X_train.shape[1])

if parameter_search_state:
    print("logistic regression parametrihaku.")

    search_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    parameter_space = {
        "model__C": [0.1, 0.5, 1.0, 2.0],
        "model__class_weight": [None, "balanced"],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=parameter_space,
        n_iter=8,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2,
        refit=True
    )

    search.fit(X_train, y_train)

    print("\nParhaat parametrit:", search.best_params_)
    print("Paras keskimääräinen CV F1:", search.best_score_)

    search_y_pred = search.best_estimator_.predict(X_test)
    search_y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]

    print("\nParametrihaun valitseman LR-mallin testitulokset")
    print("Accuracy:", accuracy_score(y_test, search_y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, search_y_proba))
    print("AP:", average_precision_score(y_test, search_y_proba))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, search_y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, search_y_pred))
