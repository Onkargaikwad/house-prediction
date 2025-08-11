# app.py (Flask)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # optional if UI same app se serve ho raha
import pandas as pd
import joblib, os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # dev me ok

# ---- Artifacts ----
PIPE_PATH  = os.getenv("PIPELINE_PATH", "housing_preprocessing.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "model_housing.pkl")
META_PATH  = os.getenv("META_PATH", "meta_columns.pkl")

pipeline = joblib.load(PIPE_PATH)
model    = joblib.load(MODEL_PATH)
meta     = joblib.load(META_PATH)
INPUT_COLS = meta["input_columns"]  # original order

# ---- Serve UI (index.html is in the SAME folder as this app.py) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def root():
    # serve ./index.html at "/"
    return send_from_directory(BASE_DIR, "index.html")

# Optional: API info
@app.route("/api-info")
def info():
    return jsonify({
        "message": "Housing Price Prediction API (Flask)",
        "expected_features": INPUT_COLS,
        "target": "price"
    })

# ---- Predict ----
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # basic validation
    missing = [c for c in INPUT_COLS if c not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 422

    # normalize yes/no fields to lowercase
    for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]:
        if k in data and isinstance(data[k], str):
            data[k] = data[k].strip().lower()

    df_in = pd.DataFrame([data])[INPUT_COLS]
    X = pipeline.transform(df_in)
    y_pred = model.predict(X)[0]
    return jsonify({"predicted_price": float(y_pred), "units": "INR", "input_echo": data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

































































# # app.py  (Flask)
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import pandas as pd
# import joblib, os

# app = Flask(__name__)
# # Agar UI ko isi Flask se serve kar rahe ho to CORS ki zarurat nahi.
# # Agar UI alag domain/port se hai to CORS enable:
# CORS(app, resources={r"/*": {"origins": "*"}})

# PIPE_PATH  = os.getenv("PIPELINE_PATH", "housing_preprocessing.pkl")
# MODEL_PATH = os.getenv("MODEL_PATH", "model_housing.pkl")
# META_PATH  = os.getenv("META_PATH", "meta_columns.pkl")

# pipeline = joblib.load(PIPE_PATH)
# model    = joblib.load(MODEL_PATH)
# meta     = joblib.load(META_PATH)
# INPUT_COLS = meta["input_columns"]  # original order

# # ---- Serve UI (put index.html in ./ui/index.html) ----
# @app.route("/")
# def root():
#     return send_from_directory(app.static_folder, "index.html")

# # Optional: API info
# @app.route("/api-info")
# def info():
#     return jsonify({
#         "message": "Housing Price Prediction API (Flask)",
#         "expected_features": INPUT_COLS,
#         "target": "price"
#     })

# # ---- Predict ----
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(force=True)

#     # basic validation
#     missing = [c for c in INPUT_COLS if c not in data]
#     if missing:
#         return jsonify({"error": f"Missing keys: {missing}"}), 422

#     # normalize yes/no fields to lowercase
#     for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]:
#         if k in data and isinstance(data[k], str):
#             data[k] = data[k].strip().lower()

#     df_in = pd.DataFrame([data])[INPUT_COLS]
#     X = pipeline.transform(df_in)
#     y_pred = model.predict(X)[0]
#     return jsonify({"predicted_price": float(y_pred), "units": "INR", "input_echo": data})

# if __name__ == "__main__":
#     # Flask default 5000. Change to 8000 if your HTML has 127.0.0.1:8000 preset.
#     app.run(host="0.0.0.0", port=8000, debug=True)

























# # # main.py (FastAPI)

# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import joblib
# # import pandas as pd

# # app = FastAPI()

# # # Load pipeline and model
# # pipeline = joblib.load("housing_preprocessing.pkl")
# # model = joblib.load("model_RFR.pkl")

# # # original input columns hi rakho
# # class InputData(BaseModel):
# #     area: int = Field(..., ge=0)
# #     bedrooms: int = Field(..., ge=0)
# #     bathrooms: int = Field(..., ge=0)
# #     stories: int = Field(..., ge=0)
# #     mainroad: Literal["yes", "no"]
# #     guestroom: Literal["yes", "no"]
# #     basement: Literal["yes", "no"]
# #     hotwaterheating: Literal["yes", "no"]
# #     airconditioning: Literal["yes", "no"]
# #     parking: int = Field(..., ge=0)
# #     prefarea: Literal["yes", "no"]
# #     furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]


# # @app.get("/")
# # def home():
# #     return {"message": "Asthma Prediction API"}

# # @app.post("/predict")
# # def predict(data: InputData):
# #     input_df = pd.DataFrame([data.dict()])

# #     # pipeline automatically scaling & encoding karega
# #     processed_input = pipeline.transform(input_df)

# #     # predict karo
# #     prediction = model.predict(processed_input)[0]
# #     probability = model.predict_proba(processed_input)[0][1]

# #     return {
# #         "prediction": int(prediction),
# #         "probability": float(probability)
# #     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# # main.py
# from fastapi.responses import FileResponse
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import Literal
# import pandas as pd
# import joblib
# import os

# app = FastAPI(title="Housing Price API", version="1.0")

# PIPE_PATH  = os.getenv("PIPELINE_PATH", "housing_preprocessing.pkl")
# MODEL_PATH = os.getenv("MODEL_PATH", "model_housing.pkl")
# META_PATH  = os.getenv("META_PATH", "meta_columns.pkl")

# try:
#     pipeline = joblib.load(PIPE_PATH)
#     model = joblib.load(MODEL_PATH)
#     meta = joblib.load(META_PATH)
#     INPUT_COLS = meta["input_columns"]
# except Exception as e:
#     raise RuntimeError(f"Failed to load artifacts: {e}")

# # --- Pydantic schema (original columns from Housing.csv minus 'price') ---
# class InputData(BaseModel):
#     area: int = Field(..., ge=0)
#     bedrooms: int = Field(..., ge=0)
#     bathrooms: int = Field(..., ge=0)
#     stories: int = Field(..., ge=0)
#     mainroad: Literal["yes", "no"]
#     guestroom: Literal["yes", "no"]
#     basement: Literal["yes", "no"]
#     hotwaterheating: Literal["yes", "no"]
#     airconditioning: Literal["yes", "no"]
#     parking: int = Field(..., ge=0)
#     prefarea: Literal["yes", "no"]
#     furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]

# @app.get("/", response_class=FileResponse)
# def serve_ui():
#     return FileResponse("index.html")

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         # DataFrame with exact original order
#         df_in = pd.DataFrame([data.dict()])[INPUT_COLS]

#         # preprocess
#         X = pipeline.transform(df_in)

#         # predict
#         y_pred = model.predict(X)[0]
#         return {"predicted_price": float(y_pred), "units": "INR"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
