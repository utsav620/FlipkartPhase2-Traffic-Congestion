from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from startup import predict_event
import pandas as pd
from startup import generate_hotspot_map
from fastapi.responses import FileResponse

pluh = pd.read_csv("dataset.csv")

app = FastAPI()

@app.get("/hotspot-map")
def hotspot_map():
    return FileResponse("hotspot_map.html")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Traffic Impact Prediction API Running"
    }

@app.post("/predict")
def predict(data: dict):
    return predict_event(data)

@app.get("/dashboard")
def dashboard():

    active_incidents = len(
        pluh[
            pluh["status"].isin(
                ["open", "assigned", "in_progress"]
            )
        ]
    )

    highest_risk_station = (
        pluh.groupby("police_station")["impact_score"]
        .mean()
        .idxmax()
    )

    return {
        "active_incidents": int(active_incidents),
        "highest_risk_station": highest_risk_station
    }