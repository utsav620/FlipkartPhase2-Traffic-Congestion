import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
pluh = pd.read_csv("dataset.csv")

pluh["start_datetime"] = pd.to_datetime(
    pluh["start_datetime"],
    utc=True,
    format="mixed",
    errors="coerce"
)

pluh = (
    pluh
    .sort_values("start_datetime")
    .reset_index(drop=True)
)

def haversine(lat1, lon1, lat2, lon2):

    R = 6371000

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


pluh["month"] = pluh["start_datetime"].dt.month
pluh["hour"] = pluh["start_datetime"].dt.hour
pluh["day_of_week"] = pluh["start_datetime"].dt.dayofweek

pluh["is_weekend"] = (
    pluh["day_of_week"].isin([5, 6])
).astype(int)

month_counts = pluh["month"].value_counts()
hour_counts = pluh["hour"].value_counts()

weekend_counts = (
    pluh["is_weekend"]
    .value_counts()
)

weekend_map = {
    0: weekend_counts[0] / weekend_counts.max(),
    1: weekend_counts[1] / weekend_counts.max()
}

severity = {
    "vehicle_breakdown": 0.10,
    "pot_holes": 0.10,
    "road_conditions": 0.15,
    "water_logging": 0.30,
    "accident": 0.40,
    "tree_fall": 0.50,
    "congestion": 0.60,
    "construction": 0.70,
    "public_event": 0.80,
    "procession": 0.90,
    "vip_movement": 0.95,
    "protest": 1.00,
    "others": 0.20,
    "Debris": 0.20,
    "debris": 0.20,
    "Fog / Low Visibility": 0.30,
    "test_demo": 0.10
}

def historical_density(lat, lon, timestamp):

    historical = pluh[
        pluh["start_datetime"]
        >= timestamp - pd.Timedelta(days=90)
    ]

    distances = haversine(
        lat,
        lon,
        historical["latitude"].values,
        historical["longitude"].values
    )

    return int(np.sum(distances <= 500))


densities = []

for idx, row in pluh.iterrows():

    current_time = row["start_datetime"]

    historical = pluh.iloc[:idx]

    historical = historical[
        historical["start_datetime"]
        >= current_time - pd.Timedelta(days=90)
    ]

    if len(historical) == 0:
        densities.append(0)
        continue

    distances = haversine(
        row["latitude"],
        row["longitude"],
        historical["latitude"].values,
        historical["longitude"].values
    )

    densities.append(
        int(np.sum(distances <= 500))
    )

pluh["historical_density_500m"] = densities

MAX_DENSITY = (
    pluh["historical_density_500m"]
    .max()
)

police_df = pd.read_csv("police_station.csv")

def nearest_police_station(lat, lon):
    distances = haversine(
        lat,
        lon,
        police_df["latitude"].values,
        police_df["longitude"].values
    )

    nearest_idx = np.argmin(distances)

    return {
        "police_station":
            police_df.iloc[nearest_idx]["police_station"],

        "distance_meters":
            round(float(distances[nearest_idx]), 2),

        "incident_count":
            int(
                police_df.iloc[nearest_idx]["incident_count"]
            )
    }

def diversion_level(score):

    if score <= DIV_Q25:
        return "No Diversion Required"

    elif score <= DIV_Q50:
        return "Diversion Recommended"

    elif score <= DIV_Q75:
        return "Diversion Required"

    return "Immediate Diversion Required"


temp = pluh.copy()

temp["spatial_risk"] = (
    temp["historical_density_500m"]
    / MAX_DENSITY
)

temp["month_risk"] = (
    temp["month"]
    .map(month_counts)
    / month_counts.max()
)

temp["hour_risk"] = (
    temp["hour"]
    .map(hour_counts)
    / hour_counts.max()
)

temp["weekend_risk"] = (
    temp["is_weekend"]
    .map(weekend_map)
)

temp["event_risk"] = (
    temp["event_cause"]
    .map(severity)
    .fillna(0.20)
)

temp["priority_risk"] = (
    temp["priority"]
    .map({
        "Low": 0.3,
        "High": 0.7
    })
    .fillna(0.3)
)

temp["planned_risk"] = (
    temp["event_type"]
    .map({
        "planned": 1.0,
        "unplanned": 0.0
    })
)

temp["road_closure_risk"] = (
    temp["requires_road_closure"]
    .astype(int)
)

temp["impact_score"] = (
    0.45 * temp["spatial_risk"]
    + 0.25 * temp["priority_risk"]
    + 0.15 * temp["event_risk"]
    + 0.10 * (
        0.5 * temp["hour_risk"]
        + 0.3 * temp["weekend_risk"]
        + 0.2 * temp["month_risk"]
    )
    + 0.05 * temp["planned_risk"]
)

temp["personnel_score"] = (
    0.40 * temp["impact_score"]
    + 0.35 * temp["priority_risk"]
    + 0.15 * temp["planned_risk"]
    + 0.10 * temp["event_risk"]
)

temp["barricading_score"] = (
    0.10 * temp["impact_score"]
    + 0.30 * temp["priority_risk"]
    + 0.20 * temp["planned_risk"]
    + 0.40 * temp["event_risk"]
)

temp["diversion_score"] = (
    0.40 * temp["barricading_score"]
    + 0.30 * temp["impact_score"]
    + 0.20 * temp["personnel_score"]
    + 0.10 * temp["road_closure_risk"]
)

DIV_Q25 = temp["diversion_score"].quantile(0.25)
DIV_Q50 = temp["diversion_score"].quantile(0.50)
DIV_Q75 = temp["diversion_score"].quantile(0.75)

personnel_q25 = temp["personnel_score"].quantile(0.25)
personnel_q50 = temp["personnel_score"].quantile(0.50)
personnel_q75 = temp["personnel_score"].quantile(0.75)

barricade_q25 = temp["barricading_score"].quantile(0.25)
barricade_q50 = temp["barricading_score"].quantile(0.50)
barricade_q75 = temp["barricading_score"].quantile(0.75)

def officers_required(score):

    if score <= personnel_q25:
        return 2

    elif score <= personnel_q50:
        return 5

    elif score <= personnel_q75:
        return 10
    
    else:
        return 20
    

def barricades_required(score):

    if score <= barricade_q25:
        return 0

    elif score <= barricade_q50:
        return 2

    elif score <= barricade_q75:
        return 5

    else:
        return 15

def predict_event(data):

    timestamp = pd.to_datetime(
        data["start_datetime"],
        utc=True
    )

    month = timestamp.month
    hour = timestamp.hour
    weekend = int(
        timestamp.dayofweek >= 5
    )

    density = historical_density(
        data["latitude"],
        data["longitude"],
        timestamp
    )

    spatial_risk = density / MAX_DENSITY

    month_risk = (
        month_counts.get(month, 0)
        / month_counts.max()
    )

    hour_risk = (
        hour_counts.get(hour, 0)
        / hour_counts.max()
    )

    weekend_risk = (
        weekend_map[weekend]
    )

    event_risk = severity.get(
        data["event_cause"],
        0.20
    )

    priority_risk = (
        0.7
        if data["priority"] == "High"
        else 0.3
    )

    planned_risk = (
        1.0
        if data["event_type"] == "planned"
        else 0.0
    )

    road_closure_risk = (
        1
        if data["requires_road_closure"]
        else 0
    )

    impact_score = (
        0.45 * spatial_risk
        + 0.25 * priority_risk
        + 0.15 * event_risk
        + 0.10 * (
            0.5 * hour_risk
            + 0.3 * weekend_risk
            + 0.2 * month_risk
        )
        + 0.05 * planned_risk
    )

    personnel_score = (
        0.40 * impact_score
        + 0.35 * priority_risk
        + 0.15 * planned_risk
        + 0.10 * event_risk
    )

    barricading_score = (
        0.10 * impact_score
        + 0.30 * priority_risk
        + 0.20 * planned_risk
        + 0.40 * event_risk
    )

    diversion_score = (
        0.40 * barricading_score
        + 0.30 * impact_score
        + 0.20 * personnel_score
        + 0.10 * road_closure_risk
    )
    nearest_station = nearest_police_station(
    data["latitude"],
    data["longitude"]
    )

    return {
        "nearest_police_station":nearest_station["police_station"],
        "distance_to_police_station_m":round(nearest_station["distance_meters"]),
        "police_station_incidents":round(nearest_station["incident_count"]),
        "congestion_risk_score": round(impact_score*100, 3),
        "personnel_number": round(officers_required(personnel_score), 3),
        "barricading_number": round(barricades_required(barricading_score), 3),
        "diversion_score": round(diversion_score, 3),
        "historical_density_500m": density,
        "diversion_recommendation": diversion_level(diversion_score)
    }


def generate_hotspot_map(pluh):

    center_lat = pluh["latitude"].mean()
    center_lon = pluh["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11
    )

    heat_data = pluh[
        ["latitude", "longitude", "impact_score"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        max_zoom=13
    ).add_to(m)

    m.save("hotspot_map.html")

    return "hotspot_map.html"



if __name__ == "__main__":

    sample_event = {
        "event_type": "planned",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "event_cause": "vip_movement",
        "requires_road_closure": True,
        "priority": "High",
        "start_datetime": "2024-02-21T18:00:00Z"
    }

    result = predict_event(sample_event)

    print("\nPrediction:\n")

    for key, value in result.items():
        print(f"{key}: {value}")