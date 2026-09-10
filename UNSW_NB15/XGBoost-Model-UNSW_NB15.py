
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




import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score



print("Luetaan aineistot")

X_train = pd.read_csv("X_train_clean.csv")
y_train = pd.read_csv("y_train_clean.csv").squeeze()

X_test = pd.read_csv("X_test_clean.csv")
y_test = pd.read_csv("y_test_clean.csv").squeeze()

if feature_removal == "Y":
    X_train = X_train.drop(columns=[c for c in features_to_be_removed if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in features_to_be_removed if c in X_test.columns])


X_test = X_test[X_train.columns]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("Luodaan XGBoost malli")

if identifier == "baseline":
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
else:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=9,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.7,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1
    )

print("Mallin opetus")

xgb.fit(X_train, y_train)

# Mallin ennuste ja 1:n todennäköisyys
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]



# Mittarien lasku

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


print(f"\nXGBoost {identifier} results")
print("----------------")

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

# Parametrihaku, jos päällä, baseline parametrit aluksi käytössä
if parameter_search_state:
    print("\nAloitetaan XGBoost-parametrihaku.")

    search_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )

    parameter_space = {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=parameter_space,
        n_iter=24,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2,
        refit=True
    )

    search.fit(X_train, y_train)

    print("Parhaat parametrit:", search.best_params_)
    print("Paras keskimääräinen CV F1:", search.best_score_)

    search_y_pred = search.best_estimator_.predict(X_test)
    search_y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]

    print("Parametrihaun valitseman XGBoost-mallin testitulokset")
    print("Accuracy:", accuracy_score(y_test, search_y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, search_y_proba))
    print("AP:", average_precision_score(y_test, search_y_proba))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, search_y_pred))
    print("Classification report:")
    print(classification_report(y_test, search_y_pred))
