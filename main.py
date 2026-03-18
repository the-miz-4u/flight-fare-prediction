import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# ================= 1. LOAD DATA =================
data = pd.read_excel("C:/Users/HP/Flight Fare Prediction/Data_Train.xlsx")
data.dropna(inplace=True)

# ================= 2. DATE FEATURES =================
data["Journey_day"] = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.day
data["Journey_month"] = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.month
data.drop("Date_of_Journey", axis=1, inplace=True)

# ================= 3. DEP & ARRIVAL TIME =================
data["Dep_hour"] = pd.to_datetime(data["Dep_Time"]).dt.hour
data["Dep_min"] = pd.to_datetime(data["Dep_Time"]).dt.minute
data.drop("Dep_Time", axis=1, inplace=True)

data["Arrival_hour"] = pd.to_datetime(data["Arrival_Time"]).dt.hour
data["Arrival_min"] = pd.to_datetime(data["Arrival_Time"]).dt.minute
data.drop("Arrival_Time", axis=1, inplace=True)

# ================= 4. DURATION =================
duration = list(data["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] += " 0m"
        else:
            duration[i] = "0h " + duration[i]

data["Duration_hours"] = [int(x.split("h")[0]) for x in duration]
data["Duration_mins"] = [int(x.split("m")[0].split()[-1]) for x in duration]
data["Total_Duration_Min"] = data["Duration_hours"]*60 + data["Duration_mins"]
data.drop("Duration", axis=1, inplace=True)

# ================= 5. TOTAL STOPS =================
data["Total_Stops"] = data["Total_Stops"].replace({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}).astype(int)

# ================= 6. WEEKEND FEATURE =================
data["Is_Weekend"] = data["Journey_day"].apply(lambda x: 1 if x in [6, 7] else 0)

# ================= 7. CATEGORICAL ENCODING =================
data = pd.concat([
    data,
    pd.get_dummies(data["Airline"], drop_first=True),
    pd.get_dummies(data["Source"], drop_first=True),
    pd.get_dummies(data["Destination"], drop_first=True)
], axis=1)

data.drop(["Airline", "Source", "Destination", "Route", "Additional_Info"], axis=1, inplace=True)

# ================= 8. OUTLIER REMOVAL =================
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1
data = data[(data["Price"] >= Q1 - 1.5*IQR) & (data["Price"] <= Q3 + 1.5*IQR)]

# ================= 9. TRAIN TEST SPLIT =================
X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= 10. TUNED RANDOM FOREST =================
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


rf.fit(X_train, y_train)

# ================= 11. PREDICTION =================
y_pred = rf.predict(X_test)

print("Test Accuracy (R2):", r2_score(y_test, y_pred) * 100)

# ================= 12. CROSS VALIDATION =================
cv_score = cross_val_score(rf, X, y, cv=5, scoring="r2")
print("Cross Validation Accuracy:", cv_score.mean() * 100)

import pickle

pickle.dump(rf, open("flight_rf_model.pkl", "wb"))

# Save feature names
feature_names = X.columns
import pickle
pickle.dump(feature_names, open("model_features.pkl", "wb"))

# Save model
pickle.dump(rf, open("flight_rf_model.pkl", "wb"))
