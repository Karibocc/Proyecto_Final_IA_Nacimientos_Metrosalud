# -*- coding: utf-8 -*-
"""Entrena y guarda los modelos que usará la API (SVM de bajo peso + regresión del peso)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score, accuracy_score

RS = 42
d = pd.read_csv("../datos_limpios.csv")
d["BAJO_PESO"] = (d.PESO_G < 2500).astype(int)

# --- Modelo 1: SVM RBF balanceado (bajo peso) ---
cols_svm = ["TALLA", "SEXO_N", "CESAREA", "ESP_N", "UPS_N", "ANIO_N"]
X, y = d[cols_svm].values, d.BAJO_PESO.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
esc = MinMaxScaler().fit(Xtr)
svm = SVC(kernel="rbf", class_weight="balanced", random_state=RS).fit(esc.transform(Xtr), ytr)
p = svm.predict(esc.transform(Xte))
print(f"SVM bajo peso -> exactitud {accuracy_score(yte,p):.4f} | sensibilidad {recall_score(yte,p):.4f}")

# --- Modelo 2: regresión lineal múltiple (peso estimado en gramos) ---
cols_reg = ["TALLA", "SEXO_N", "CESAREA", "ESP_N"]
reg = LinearRegression().fit(d[cols_reg].values, d.PESO_G.values)

joblib.dump({"escalador": esc, "svm": svm, "regresion": reg,
             "cols_svm": cols_svm, "cols_reg": cols_reg}, "modelos.joblib")
print("Modelos guardados en modelos.joblib")
