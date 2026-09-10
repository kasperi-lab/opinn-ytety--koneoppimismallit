#XGBoost malli CICIDS2017 ainestossa


identifier = input("baseline vai parannettu:")
if identifier != "baseline" and identifier != "parannettu":
    raise ValueError("Valitse baseline tai parannettu!")

feature_importance_state = input("Feature importance valinta (Y/N)").strip()
if feature_importance_state == "Y":
    feature_importance_state = True
elif feature_importance_state == "N":
    feature_importance_state = False
else:
    raise ValueError("Valitse feature importancen käyttö.")


import numpy as np 
import pandas as pd 
import os  
import kagglehub 
 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score 
 
# Ladataan dataset
path = kagglehub.dataset_download("aymenabb/ddos-evaluation-dataset-cic-ddos2019") 
file_path = os.path.join(path, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") 
 
df = pd.read_csv(file_path) 
 

 
df.columns = df.columns.str.strip() 
 
# Poistetaan tunniste- ja metatiedot 
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
 
# Ennustettava arvo
y = (df["Label"] != "BENIGN").astype(int) 

# Pelkät piirteet X dataframeen 
X = df.drop(columns=["Label"]) 
 
# vain numeeriset arvot
X = X.select_dtypes(include=[np.number]) 
 
# Loputtomien arvojen vaihto NaN:iin 
X = X.replace([np.inf, -np.inf], np.nan) 

# Piirteiden poisto jos käytössä parannettu malli
if identifier == "parannettu":
    features_to_remove = [
        "Destination Port",
        "Total Length of Fwd Packets",
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
        "Max Packet Length"
    ]



    X = X.drop(columns=[c for c in features_to_remove if c in X.columns])

# Poistetaan piirteet joiden arvot eivät muutu 
constant_columns = X.columns[X.nunique(dropna=False) <= 1] 
X = X.drop(columns=constant_columns) 

#Jaetaan aineisto testi ja opetus- setteihin, stratify=y, jotta aineistot ovat tasaiset. 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size = 0.2, random_state = 42, stratify=y) 
 


scale_pos_weight = (y == 0).sum() / (y == 1).sum() 

# Mallin parametrit, parannettu
if identifier == "parannettu":
    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,

        subsample=0.9,
        colsample_bytree=0.8,

        min_child_weight=2,
        gamma=0.1,

        reg_alpha=0.01,
        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        eval_metric="logloss",

        n_jobs=-1,
        random_state=42
    )

# Mallin parametrit, baseline
elif identifier == "baseline":
    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

else:
    raise ValueError("baseline vai parannettu?")

#Mallin luonti
model.fit(X_train, y_train) 

# Mallin ennustukset
y_pred = model.predict(X_test) 

# Mallin todennäköisyys ennusteluokalle
y_proba = model.predict_proba(X_test)[:, 1] 
 

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
 

# Mallin tulosteet

print(f'\nXGBoost {identifier} tulokset:') 
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


# Jos halutaan ajaa feature importance mallille
if feature_importance_state:
    importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

#n_repeats = kuinka monta kertaa malli ajetaan sekoitetuilla featureilla

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
