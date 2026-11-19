# -*- coding: utf-8 -*-
"""Proyecto final IA - Metodología de un proyecto de aprendizaje automático.
Dataset: Nacimientos atendidos en la ESE Metrosalud (datos.gov.co).
Experimentos: (1) SVM clasificación tipo de parto, (2) Regresión lineal del peso, (3) K-means."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             mean_squared_error, r2_score, mean_absolute_error, silhouette_score)
RS = 42; R = {}
# ---------- 1. Carga y exploración ----------
df = pd.read_csv("datos_originales.csv")
df.columns = ['ANIO','FECHA','SEXO','ESPECIALIDAD','ESTADO','PESO','GRUPO_PESO','TALLA','CCO','UPS','TIPOPARTO','BARRIO_COD','BARRIO']
R['registros_originales'] = len(df)
R['nulos'] = df.isna().sum().to_dict()
# ---------- 2. Limpieza ----------
df['PESO_G'] = pd.to_numeric(df['PESO'].astype(str).str.replace(',', '', regex=False), errors='coerce')
df['FECHA_D'] = pd.to_datetime(df['FECHA'].str.replace(r' 12:00:00 AM', '', regex=True), format='%Y %b %d', errors='coerce')
d = df.dropna(subset=['PESO_G','TALLA','SEXO','TIPOPARTO','UPS']).copy()
d = d[d.TIPOPARTO.isin(['PARTO','CESAREA']) & d.SEXO.isin(['M','F']) & d.ESTADO.isin(['VIVO','MUERTO'])]
d = d[(d.PESO_G.between(500, 5500)) & (d.TALLA.between(25, 60))]
R['registros_limpios'] = len(d)
R['eliminados'] = R['registros_originales'] - len(d)
# ---------- 3. Preparación / normalización ----------
d['SEXO_N'] = (d.SEXO == 'M').astype(int)                 # F=0, M=1
d['ESTADO_N'] = (d.ESTADO == 'MUERTO').astype(int)        # VIVO=0, MUERTO=1
d['CESAREA'] = (d.TIPOPARTO == 'CESAREA').astype(int)     # PARTO=0, CESAREA=1
d['ESP_N'] = (d.ESPECIALIDAD == 'Medico(a) Especialista').astype(int)
top = ['UPSS MANRIQUE','UPSS NUEVO OCCIDENTE','UPSS BELEN']
d['UPS_N'] = d.UPS.map({u:i for i,u in enumerate(top)}).fillna(3).astype(int)
d['ANIO_N'] = d.ANIO
d.to_csv("datos_limpios.csv", index=False)
R['distribucion_tipo_parto'] = d.TIPOPARTO.value_counts().to_dict()
R['peso_describe'] = d.PESO_G.describe().round(1).to_dict()
R['talla_describe'] = d.TALLA.describe().round(1).to_dict()
R['por_anio'] = d.ANIO.value_counts().sort_index().to_dict()
R['cesarea_por_anio_pct'] = (d.groupby('ANIO').CESAREA.mean()*100).round(1).to_dict()
# Figuras exploratorias
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
sns.histplot(d.PESO_G, bins=40, ax=ax[0], color="#2a6f97"); ax[0].set_xlabel("Peso al nacer (g)")
sns.histplot(d.TALLA, bins=30, ax=ax[1], color="#52796f"); ax[1].set_xlabel("Talla (cm)")
d.TIPOPARTO.value_counts().plot.bar(ax=ax[2], color=["#2a6f97","#c8553d"]); ax[2].set_xlabel("Tipo de parto"); ax[2].tick_params(axis='x', rotation=0)
for a in ax: a.set_ylabel("Frecuencia")
plt.tight_layout(); plt.savefig("figuras/fig_exploracion.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(4.2, 3))
sns.heatmap(d[['PESO_G','TALLA','SEXO_N','ESTADO_N','CESAREA','ESP_N','ANIO_N']].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, annot_kws={"size":6}, cbar=False)
ax.tick_params(labelsize=6); plt.tight_layout(); plt.savefig("figuras/fig_correlacion.png", dpi=200); plt.close()
# ---------- 4. EXPERIMENTO 1: SVM (clasificación) ----------
def svm_exp(cols, kernel, cw=None, n=None):
    X = d[cols].values; y = d.CESAREA.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
    sc = MinMaxScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = SVC(kernel=kernel, class_weight=cw, random_state=RS).fit(Xtr, ytr)
    p = m.predict(Xte); rep = classification_report(yte, p, output_dict=True, zero_division=0)
    return dict(columnas=cols, kernel=kernel, class_weight=cw, accuracy=round(accuracy_score(yte,p),4),
        precision_cesarea=round(rep['1']['precision'],4), recall_cesarea=round(rep['1']['recall'],4), f1_cesarea=round(rep['1']['f1-score'],4),
        cm=confusion_matrix(yte,p).tolist(), n_train=len(ytr), n_test=len(yte))
base = ['PESO_G','TALLA','SEXO_N','ESTADO_N','UPS_N','ANIO_N']
R['svm'] = {
 'E1_lineal': svm_exp(base,'linear'),
 'E2_rbf': svm_exp(base,'rbf'),
 'E3_rbf_balanceado': svm_exp(base,'rbf','balanced'),
 'E4_rbf_balanceado_especialidad': svm_exp(base+['ESP_N'],'rbf','balanced'),
}
cm = np.array(R['svm']['E3_rbf_balanceado']['cm'])
fig, ax = plt.subplots(figsize=(3.2, 2.7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Parto","Cesárea"], yticklabels=["Parto","Cesárea"], ax=ax)
ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); plt.tight_layout(); plt.savefig("figuras/fig_svm_cm.png", dpi=200); plt.close()
# ---------- 4b. EXPERIMENTO 1b: SVM - bajo peso al nacer (< 2500 g) ----------
d['BAJO_PESO'] = (d.PESO_G < 2500).astype(int)
R['pct_bajo_peso'] = round(d.BAJO_PESO.mean()*100, 2)
def svm_bp(cols, kernel, cw=None):
    X = d[cols].values; y = d.BAJO_PESO.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
    sc = MinMaxScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    p = SVC(kernel=kernel, class_weight=cw, random_state=RS).fit(Xtr, ytr).predict(Xte)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    return dict(columnas=cols, kernel=kernel, class_weight=cw, accuracy=round(accuracy_score(yte,p),4),
        precision=round(rep['1']['precision'],4), recall=round(rep['1']['recall'],4), f1=round(rep['1']['f1-score'],4), cm=confusion_matrix(yte,p).tolist())
cb = ['TALLA','SEXO_N','CESAREA','ESP_N','UPS_N','ANIO_N']
R['svm_bp'] = {'B1_lineal': svm_bp(cb,'linear'), 'B2_rbf': svm_bp(cb,'rbf'), 'B3_rbf_balanceado': svm_bp(cb,'rbf','balanced'),
               'B4_rbf_balanceado_sin_talla': svm_bp([c for c in cb if c!='TALLA'],'rbf','balanced')}
cm = np.array(R['svm_bp']['B3_rbf_balanceado']['cm'])
fig, ax = plt.subplots(figsize=(3.2, 2.7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Normal","Bajo peso"], yticklabels=["Normal","Bajo peso"], ax=ax)
ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); plt.tight_layout(); plt.savefig("figuras/fig_svm_bp_cm.png", dpi=200); plt.close()
# ---------- 5. EXPERIMENTO 2: Regresión lineal (peso) ----------
yr = d.PESO_G.values
def reg_exp(cols):
    X = d[cols].values
    Xtr, Xte, ytr, yte = train_test_split(X, yr, test_size=0.20, random_state=RS)
    m = LinearRegression().fit(Xtr, ytr); p = m.predict(Xte)
    return dict(columnas=cols, r2=round(r2_score(yte,p),4), rmse=round(mean_squared_error(yte,p)**0.5,1), mae=round(mean_absolute_error(yte,p),1),
                coef=[round(c,2) for c in m.coef_], intercepto=round(float(m.intercept_),1)), (m, Xte, yte, p)
R['reg'] = {}
R['reg']['R1_simple'], (m1, Xte1, yte1, p1) = reg_exp(['TALLA'])
R['reg']['R2_multiple'], _ = reg_exp(['TALLA','SEXO_N','CESAREA','ESP_N'])
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.scatter(Xte1[:,0], yte1, s=3, alpha=.25, color="#2a6f97", label="Datos de prueba")
xs = np.linspace(d.TALLA.min(), d.TALLA.max(), 50).reshape(-1,1); ax.plot(xs, m1.predict(xs), color="#c8553d", lw=2, label="Recta de regresión")
ax.set_xlabel("Talla (cm)"); ax.set_ylabel("Peso (g)"); ax.legend(fontsize=6); plt.tight_layout(); plt.savefig("figuras/fig_regresion.png", dpi=200); plt.close()
# ---------- 6. EXPERIMENTO 3: K-means ----------
Xk = MinMaxScaler().fit_transform(d[['PESO_G','TALLA']].values)
ks = range(2, 9); inercia = []; sil = []
rng = np.random.RandomState(RS); idx = rng.choice(len(Xk), 5000, replace=False)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Xk)
    inercia.append(km.inertia_); sil.append(silhouette_score(Xk[idx], km.labels_[idx]))
R['kmeans_inercia'] = dict(zip(map(str,ks), np.round(inercia,1).tolist())); R['kmeans_silhouette'] = dict(zip(map(str,ks), np.round(sil,4).tolist()))
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax[0].plot(list(ks), inercia, "o-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia"); ax[0].set_title("Curva de codo", fontsize=8)
ax[1].plot(list(ks), sil, "o-", color="#c8553d"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silueta"); ax[1].set_title("Coeficiente de silueta", fontsize=8)
plt.tight_layout(); plt.savefig("figuras/fig_codo.png", dpi=200); plt.close()
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=RS).fit(Xk); d['CLUSTER'] = km.labels_
perf = d.groupby('CLUSTER').agg(n=('PESO_G','size'), peso_medio=('PESO_G','mean'), talla_media=('TALLA','mean'),
                                 pct_cesarea=('CESAREA','mean'), pct_muerto=('ESTADO_N','mean'))
perf['pct_cesarea'] *= 100; perf['pct_muerto'] *= 100
R['kmeans_perfiles'] = perf.round(2).reset_index().to_dict(orient='records')
fig, ax = plt.subplots(figsize=(3.6, 2.9))
sns.scatterplot(x=d.TALLA, y=d.PESO_G, hue=d.CLUSTER, palette="Set1", s=5, alpha=.5, ax=ax, legend=True)
ax.set_xlabel("Talla (cm)"); ax.set_ylabel("Peso (g)"); ax.legend(title="Grupo", fontsize=6, title_fontsize=7, markerscale=2); plt.tight_layout(); plt.savefig("figuras/fig_kmeans.png", dpi=200); plt.close()
d.to_csv("datos_con_clusters.csv", index=False)
json.dump(R, open("resultados.json","w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
