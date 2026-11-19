# -*- coding: utf-8 -*-
"""API REST (Flask) + página web para el prototipo. Ejecutar: python app.py  ->  http://127.0.0.1:5000"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import joblib, numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
M = joblib.load("modelos.joblib")

@app.get("/")
def inicio():
    return render_template("index.html")

@app.get("/salud")
def salud():
    return jsonify(estado="ok")

@app.post("/predecir")
def predecir():
    try:
        j = request.get_json(force=True)
        talla = float(j["talla"]); sexo = int(j["sexo"]); cesarea = int(j["cesarea"])
        esp = int(j["especialista"]); ups = int(j["sede"]); anio = int(j["anio"])
        if not (25 <= talla <= 60):
            return jsonify(error="La talla debe estar entre 25 y 60 cm."), 400
        if sexo not in (0, 1) or cesarea not in (0, 1) or esp not in (0, 1) or ups not in (0, 1, 2, 3):
            return jsonify(error="Valores de entrada no válidos."), 400
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Datos incompletos o con formato incorrecto."), 400
    x = np.array([[talla, sexo, cesarea, esp, ups, anio]])
    bajo = int(M["svm"].predict(M["escalador"].transform(x))[0])
    peso = float(M["regresion"].predict(np.array([[talla, sexo, cesarea, esp]]))[0])
    return jsonify(bajo_peso=bool(bajo), peso_estimado_g=round(peso))

if __name__ == "__main__":
    app.run(debug=False, port=5000)
