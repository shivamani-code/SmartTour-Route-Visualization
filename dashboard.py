import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

st.set_page_config(page_title="SmartTour Analytics Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("SmartTourRoutePlanner.csv")

df = load_data()

# -----------------------------
# HERO SECTION
# -----------------------------
st.title("SmartTour Route Intelligence Dashboard")

st.markdown("""
This dashboard analyzes **tourism routes, traveler preferences, costs, and traffic patterns**
across major Indian destinations.  
Use the filters in the sidebar to explore travel behavior and demand patterns.
""")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Dashboard Filters")

season_filter = st.sidebar.multiselect(
    "Select Season",
    options=df["season"].unique(),
    default=df["season"].unique()
)

transport_filter = st.sidebar.multiselect(
    "Transport Mode",
    options=df["transport_mode"].unique(),
    default=df["transport_mode"].unique()
)

day_filter = st.sidebar.multiselect(
    "Day Type",
    options=df["day_type"].unique(),
    default=df["day_type"].unique()
)

filtered_df = df[
    (df["season"].isin(season_filter)) &
    (df["transport_mode"].isin(transport_filter)) &
    (df["day_type"].isin(day_filter))
]

# -----------------------------
# HERO KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Routes", len(filtered_df))
col2.metric("Avg Satisfaction", round(filtered_df["satisfaction_rating"].mean(),2))
col3.metric("Avg Travel Time (hrs)", round(filtered_df["estimated_travel_time_hr"].mean(),2))
col4.metric("Avg Budget", round(filtered_df["user_budget"].mean(),2))

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Traffic Analysis",
    "Demand Patterns",
    "Budget & Preferences",
    "Cost Breakdown",
    "India Route Map"
])

# -----------------------------
# TRAFFIC ANALYSIS
# -----------------------------
with tab1:

    st.subheader("Traffic Density vs Travel Time")

    st.markdown("""
**Columns used**

• `traffic_density`  
• `estimated_travel_time_hr`

**Insight**

Shows how increasing traffic congestion impacts overall travel time.  
Higher traffic density typically results in longer travel durations.
""")

    filtered_df["traffic_density_bin"] = pd.cut(
        filtered_df["traffic_density"],
        bins=[0,0.2,0.4,0.6,0.8,1],
        labels=["Very Low","Low","Medium","High","Very High"]
    )

    traffic_time = (
        filtered_df.groupby("traffic_density_bin")["estimated_travel_time_hr"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        traffic_time,
        x="traffic_density_bin",
        y="estimated_travel_time_hr",
        markers=True,
        title="Impact of Traffic Density on Travel Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Traffic Density Distribution")

    st.markdown("""
**Columns used**

• `day_type`  
• `traffic_density`

**Insight**

Compares traffic congestion patterns between **weekdays and weekends**.
""")

    fig2 = px.violin(
        filtered_df,
        x="day_type",
        y="traffic_density",
        color="day_type",
        box=True,
        points="all"
    )

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# DEMAND ANALYSIS
# -----------------------------
with tab2:

    st.subheader("Travel Demand Heatmap")

    st.markdown("""
**Columns used**

• `season`  
• `day_type`  
• `popularity_score`

**Insight**

Shows when travel demand is highest across seasons and days.
""")

    fig3 = px.density_heatmap(
        filtered_df,
        x="season",
        y="day_type",
        z="popularity_score",
        histfunc="avg"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Travel Preference Hierarchy")

    st.markdown("""
**Columns used**

• `season`  
• `transport_mode`  
• `destination_type`

**Insight**

Displays how travel preferences change depending on season and transport type.
""")

    sunburst_data = (
        filtered_df.groupby(["season","transport_mode","destination_type"])["popularity_score"]
        .sum()
        .reset_index()
    )

    fig4 = px.sunburst(
        sunburst_data,
        path=["season","transport_mode","destination_type"],
        values="popularity_score"
    )

    st.plotly_chart(fig4, use_container_width=True)


# -----------------------------
# BUDGET ANALYSIS
# -----------------------------
with tab3:

    st.subheader("Budget vs Satisfaction")

    st.markdown("""
**Columns used**

• `user_budget`  
• `satisfaction_rating`  
• `transport_mode`

**Insight**

Explores whether higher travel budgets result in better traveler satisfaction.
""")

    fig5 = px.scatter(
        filtered_df,
        x="user_budget",
        y="satisfaction_rating",
        color="transport_mode"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Transport Mode Preference Radar")

    st.markdown("""
**Columns used**

• `user_budget`
• `user_time_constraint_hr`
• `popularity_score`
• `traffic_density`
• `satisfaction_rating`

**Insight**

Compares travel modes across multiple traveler priorities.
""")

    radar = filtered_df.groupby("transport_mode")[
        ["user_budget","user_time_constraint_hr","popularity_score","traffic_density","satisfaction_rating"]
    ].mean()

    scaler = MinMaxScaler()
    radar_scaled = scaler.fit_transform(radar)

    categories = ["Budget","Time","Popularity","Traffic","Satisfaction"]

    fig6 = go.Figure()

    for i, mode in enumerate(radar.index):

        values = radar_scaled[i].tolist()
        values += values[:1]

        fig6.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=mode
        ))

    fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])))

    st.plotly_chart(fig6, use_container_width=True)


# -----------------------------
# COST BREAKDOWN
# -----------------------------
with tab4:

    st.subheader("Travel Cost Distribution")

    st.markdown("""
**Columns used**

• `entry_fee`
• `accommodation_cost`
• `food_cost`

**Insight**

Shows how total travel expenses are distributed across key cost components.
""")

    entry = filtered_df["entry_fee"].mean()
    accom = filtered_df["accommodation_cost"].mean()
    food = filtered_df["food_cost"].mean()

    fig7 = go.Figure(data=[go.Sankey(
        node=dict(label=["Total Cost","Entry Fee","Accommodation","Food"]),
        link=dict(
            source=[0,0,0],
            target=[1,2,3],
            value=[entry,accom,food]
        )
    )])

    st.plotly_chart(fig7, use_container_width=True)


# -----------------------------
# ROUTE MAP
# -----------------------------
with tab5:

    st.subheader("SmartTour Routes Across India")

    st.markdown("""
Shows travel routes between major Indian cities.  
Line colors represent **transport modes**.
""")

    city_coords = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Kolkata": (22.5726, 88.3639),
        "Agra": (27.1767, 78.0081),
        "Goa": (15.2993, 74.1240),
        "Shimla": (31.1048, 77.1734),
        "Ooty": (11.4064, 76.6932),
        "Mahabalipuram": (12.6208, 80.1937)
    }

    filtered_df["start_lat"]=filtered_df["start_location"].map(lambda x: city_coords.get(x,(None,None))[0])
    filtered_df["start_lon"]=filtered_df["start_location"].map(lambda x: city_coords.get(x,(None,None))[1])

    filtered_df["end_lat"]=filtered_df["end_location"].map(lambda x: city_coords.get(x,(None,None))[0])
    filtered_df["end_lon"]=filtered_df["end_location"].map(lambda x: city_coords.get(x,(None,None))[1])

    m = folium.Map(location=[22.5,80], zoom_start=5, tiles="cartodbpositron")

    mode_colors={
        "car":"blue",
        "train":"green",
        "bike":"orange",
        "walk":"purple",
        "bus":"brown"
    }

    for _,row in filtered_df.iterrows():

        if pd.notnull(row["start_lat"]) and pd.notnull(row["end_lat"]):

            start=(row["start_lat"],row["start_lon"])
            end=(row["end_lat"],row["end_lon"])

            color=mode_colors.get(str(row["transport_mode"]).lower(),"black")

            folium.PolyLine(
                locations=[start,end],
                color=color,
                weight=3,
                opacity=0.8
            ).add_to(m)

            folium.CircleMarker(
                location=start,
                radius=max(3,row["popularity_score"]/20),
                color="blue",
                fill=True
            ).add_to(m)

            folium.CircleMarker(
                location=end,
                radius=max(3,row["popularity_score"]/20),
                color="orange",
                fill=True
            ).add_to(m)

    st_folium(m, width=900, height=600)
