from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,roc_auc_score,average_precision_score)
from Piirteiden_poisto_lista import features_to_be_removed

# Halutaanko käyttää parannettua mallia vai baseline mallia (parametrit)
identifier = input("baseline vai parannettu:").strip().lower()

if identifier not in ["baseline", "parannettu"]:
    raise ValueError("Valitse baseline tai parannettu!")


# Halutaanko ajossa toteuttaa parametrihaku, eli parhaiden parametrien etsintä
parameter_search_state = input("Parametrihaku valinta (Y/N): ").strip().upper()

if parameter_search_state not in ["Y", "N"]:
    raise ValueError("Valitse parametrihaun käyttö: Y tai N.")


# Halutaanko poistaa määritellyt piirteet mallin parantamiseksi
feature_removal = input("Poistetaanko piirteitä? (Y/N): ").strip().upper()

if feature_removal not in ["Y", "N"]:
    raise ValueError("Valitse Y tai N.")

# piirretärkeyden näyttäminen
feature_importance = input("Näytetäänkö piirteiden tärkeys? (Y/N): ").strip().upper()

if feature_importance not in ["Y", "N"]:
    raise ValueError("Valitse Y tai N.")


# Luetaan aineisto dataframeihin
print("Luetaan aineisto")

X_train = pd.read_csv("X_train_clean.csv")
y_train = pd.read_csv("y_train_clean.csv").squeeze()

X_test = pd.read_csv("X_test_clean.csv")
y_test = pd.read_csv("y_test_clean.csv").squeeze()

# Poistetaan featuret jos valittu
if feature_removal == "Y":
    X_train = X_train.drop(columns=[c for c in features_to_be_removed if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in features_to_be_removed if c in X_test.columns])



# Varmistetaan että sarakkeet vastaavat toisiaan
X_test = X_test[X_train.columns]


print("Luodaan RF-malli")

# Baseline ja parannetun mallin parametrit
if identifier == "baseline":
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
else:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.5,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )

print("Mallin koulutus")

rf.fit(X_train, y_train)


# Ennuste ja 1:n todennäköisyys
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]


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


print(f"\nRandom Forest {identifier} results")
print("---------------------")

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

# Parametrihaun ollessa päällä
if parameter_search_state == "Y":
    print("\nAloitetaan Random Forest -parametrihaku.")

    search_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    )

    parameter_space = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"]
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

    print("Parametrihaun valitseman RF-mallin testitulokset")
    print("Accuracy:", accuracy_score(y_test, search_y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, search_y_proba))
    print("AP:", average_precision_score(y_test, search_y_proba))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, search_y_pred))
    print("Classification report:")
    print(classification_report(y_test, search_y_pred))


if feature_importance == "Y":

    importance_result = permutation_importance(
        rf,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=1
    )

    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance_result.importances_mean,
        "Std": importance_result.importances_std
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature importance")
    print()
    print(feature_importance.to_string(index=False))