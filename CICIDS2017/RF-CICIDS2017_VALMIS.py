# Identifier arvoa muuttamalla saadaan mallista joko baseline, tai parannettu ->
# -> ilman tarvetta kahdelle eri mallitiedostolle

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

import os
import kagglehub
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)



# Datasetin lataus kagglesta


print("Luetaan aineisto.")

path = kagglehub.dataset_download(
    "aymenabb/ddos-evaluation-dataset-cic-ddos2019"
)

file_path = os.path.join(
    path,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

df = pd.read_csv(file_path)


# Yhtenäinen esikäsittely malleille


df.columns = df.columns.str.strip()

# Poistetaan tunniste- ja metatietoja. Eivät vaikuta tuloksiin.
drop_metadata = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

df = df.drop(
    columns=[c for c in drop_metadata if c in df.columns]
)

# Poistetaan duplikaatit
df = df.drop_duplicates()

# Muutetaan luokat
# 0 = Normaali
# 1 = Hyökkäys
y = (df["Label"] != "BENIGN").astype(int)

# Piirteet, eli poistetaan ennustettava
X = df.drop(columns=["Label"])

# Pidetään vain numeeriset piirteet
X = X.select_dtypes(include=[np.number])

# Korvataan loputtomat arvot NaN:illa
X = X.replace([np.inf, -np.inf], np.nan)


# Aineistojako Train / Test. Opetusaineisto 80% aineistosta
# Stratifyllä yhtenäinen suhde normaali/hyökkäys testiin ja trainiin

if identifier == "parannettu":


# Parannetun mallin piirteiden poisto

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
        "Max Packet Length"
    ]

    X = X.drop(columns=[c for c in features_to_remove if c in X.columns])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Poistetaan arvot mitkä ei muutu, eli arvo joko pelkkä 0 tai 1 ->
# -> Jokaisessa havainnossa

constant_columns = X_train.columns[
    X_train.nunique(dropna=False) <= 1
]

X_train = X_train.drop(columns=constant_columns)
X_test = X_test.drop(columns=constant_columns)



# Puuttuvat arvot

# Lasketaan puuttuvien arvojen tilalle mediaaniarvot opetusdatasta
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# Random Forest-malli


print("Luodaan malli: ")


if identifier == "parannettu":
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

elif identifier == "baseline":
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

else:
    raise ValueError("Parannettu vai Baseline?")


print("Mallin opetus.")

rf.fit(X_train, y_train)


# Mallin ennusteet

y_pred = rf.predict(X_test)

# Mallin antama todennäköisyys sille, onko liikenne normaalia vai hyökkäys
y_proba = rf.predict_proba(X_test)[:, 1]


# Mittarien tulostus


acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

roc = roc_auc_score(y_test, y_proba)

average_precision = average_precision_score(
    y_test,
    y_proba
)


TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


fn_rate = FN / (FN + TP)
recall = TP / (TP + FN)

precision = TP / (TP + FP)

specificity = TN / (TN + FP)


print(f'{identifier}-RF tulokset')
print()

print("Accuracy:", acc)
print("ROC-AUC:", roc)
print("Average Precision:", average_precision)

print("\nConfusion Matrix:")
print(cm)

print("\nFalse-negative ratio:", fn_rate)
print("Recall:", recall)
print("Precision:", precision)
print("Specificity:", specificity)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"]
    )
)

if feature_importance_state:
    importance_result = permutation_importance(
        rf,
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
