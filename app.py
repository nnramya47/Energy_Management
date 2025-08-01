import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Household Energy Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("household_energy.csv", parse_dates=['timestamp'])
df['hour'] = df['timestamp'].dt.hour

st.title(" Household Energy Consumption Dashboard")

# Show basic info
st.subheader("Dataset Preview")
st.dataframe(df.head())


# Daily Consumption Plot
st.subheader("Daily Energy Consumption")
daily_consumption = df.groupby(df['timestamp'].dt.date)['energy_consumption'].sum()
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(daily_consumption.index, daily_consumption.values, marker='o')
ax1.set_xlabel("Date")
ax1.set_ylabel("Energy (kWh)")
ax1.set_title("Daily Household Energy Consumption")
ax1.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Daily Consumption Bar Plot
st.subheader("Daily Energy Consumption (Bar Graph)")
daily_consumption_bar = df.groupby(df['timestamp'].dt.date)['energy_consumption'].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(daily_consumption_bar['timestamp'], daily_consumption_bar['energy_consumption'], color='skyblue')
ax2.set_xlabel("Date")
ax2.set_ylabel("Energy (kWh)")
ax2.set_title("Daily Household Energy Consumption (Bar Plot)")
ax2.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Prediction Section
st.subheader("Energy Consumption Prediction")
import joblib
model = joblib.load("forecast_model.pkl")
temperature = st.number_input("Temperature", value=25.0)
outside_temperature = st.number_input("Outside Temperature", value=30.0)
device_usage = st.number_input("Device Usage", value=1)
hour = st.number_input("Hour", value=12)
weekday = st.number_input("Weekday (0=Mon, 6=Sun)", value=0)
if st.button("Predict"):
    input_df = pd.DataFrame([[temperature, outside_temperature, device_usage, hour, weekday]],
                            columns=['temperature', 'outside_temperature', 'device_usage', 'hour', 'weekday'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")