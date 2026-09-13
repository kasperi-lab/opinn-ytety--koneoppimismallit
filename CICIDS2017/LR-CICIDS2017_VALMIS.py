from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import os 
import kagglehub 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import ( accuracy_score, confusion_matrix, roc_auc_score, classification_report, average_precision_score ) 
from sklearn.preprocessing import StandardScaler 
from sklearn.inspection import permutation_importance

# Mallin valinta
identifier = input("baseline vai parannettu:")

if identifier not in ["baseline", "parannettu"]:
    raise ValueError("Valitse baseline tai parannettu!")

# Feature importancen näyttäminen
feature_importance_state = input("Feature importance valinta (Y/N)").strip()

if feature_importance_state not in ["Y", "N"]:
    raise ValueError("Valitse feature importancen käyttö.")


 
# Aineiston lataus
path = kagglehub.dataset_download( 
    "aymenabb/ddos-evaluation-dataset-cic-ddos2019" 
) 
 
file_path = os.path.join( 
    path, 
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" 
) 

df = pd.read_csv(file_path) 
 
 
 
# Esikäsittelyosio
 
 
df.columns = df.columns.str.strip() 

# Tunnistetietojen ja metatietojen poisto
drop_metadata = [ 
    "Flow ID", 
    "Source IP", 
    "Destination IP", 
    "Timestamp" 
] 

df = df.drop( 
    columns=[c for c in drop_metadata if c in df.columns] 
) 
 
# Duplikaattien poisto 
df = df.drop_duplicates() 
 
# Ennustettava luokka 
y = (df["Label"] != "BENIGN").astype(int) 
 
# Piirteet X dataframelle 
X = df.drop(columns=["Label"]) 
 
# Valitaan vain numeeriset arvot
X = X.select_dtypes(include=[np.number]) 
 
# inf -> NaN 
X = X.replace([np.inf, -np.inf], np.nan) 
 

if identifier == "parannettu":

    features_to_remove = [
        "Destination Port",
        "Fwd Header Length",
        "RST Flag Count",
        "ECE Flag Count",
        "FIN Flag Count",
        "Bwd IAT Total",
        "Active Mean",
        "Bwd Packet Length Min",
        "Idle Min",
        "Bwd Packet Length Mean",
        "Idle Mean",
        "Avg Bwd Segment Size",
        "Max Packet Length",
        "Total Length of Fwd Packets"
    ]

    X = X.drop(columns=[c for c in features_to_remove if c in X.columns])
 

# TRAIN / TEST SPLIT 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
) 
 
 
 # Poistetaan muuttumattomat arvot
constant_columns = X.columns[ 
    X.nunique(dropna=False) <= 1 
] 
 
X = X.drop(columns=constant_columns) 
 
 
# Lasketaan mediaanit training datasta
train_medians = X_train.median() 
 
X_train = X_train.fillna(train_medians) 
X_test = X_test.fillna(train_medians) 
 
 
# LR:n skaalaus tehdään mallin Pipelinessä
 
 
 
 
# Baseline tai parametrihaulla valittu malli
if identifier == "baseline":
    # Luodaan ja koulutetaan baseline-malli
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
        C=1,
        max_iter=2000,
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        random_state=42
    ))
    ])
    lr.fit(X_train, y_train)

else:
    # Parannetussa ajossa haetaan parhaat parametrit
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

    # Käytetään haun valitsemaa, koko opetusaineistolla koulutettua mallia
    lr = search.best_estimator_

 
 
 
 # Ennustus
y_pred = lr.predict(X_test) 
 
# Attack luokan todennäköisyys havainnolle
y_proba = lr.predict_proba(X_test)[:, 1] 
 
 
# Mittareiden lasku
 
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
 
 
print(f'\nLogistic regression {identifier} tulokset:') 
print()
print("Accuracy:", acc) 
print("ROC-AUC:", roc) 
print("AP:", pr_auc) 
 
print("\nConfusion Matrix:") 
print(cm) 
 
print("\nFalse-negative ratio:", fn_rate) 
print("Recall ratio:", recall) 
 
print("\nClassification report:") 
print(classification_report(y_test, y_pred)) 

if feature_importance_state == "Y":

    importance_result = permutation_importance(
        lr,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
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


