import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

st.set_page_config(page_title="SmartTour Analytics", layout="wide")

# -----------------------------
# CUSTOM CSS (Aesthetic UI)
# -----------------------------
st.markdown("""
<style>

.hero {
background: linear-gradient(120deg,#0f172a,#1e293b,#334155);
padding: 40px;
border-radius: 18px;
color:white;
margin-bottom:20px;
transition: transform 0.3s ease;
}

.hero:hover{
transform: scale(1.01);
}

.hero-title{
font-size:38px;
font-weight:700;
}

.hero-sub{
opacity:0.8;
font-size:16px;
}

.metric-card{
background:#f8fafc;
padding:20px;
border-radius:12px;
text-align:center;
box-shadow:0 2px 10px rgba(0,0,0,0.05);
}

.section{
background:white;
padding:25px;
border-radius:16px;
margin-bottom:25px;
box-shadow:0 2px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="hero">', unsafe_allow_html=True)

col1,col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="hero-title">SmartTour Route Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
    '<div class="hero-sub">Explore travel demand, cost patterns, and transportation preferences across India.</div>',
    unsafe_allow_html=True)

with col2:
    st.markdown("### Filters")

    season_filter = st.multiselect(
        "Season",
        df["season"].unique(),
        default=df["season"].unique()
    )

    transport_filter = st.multiselect(
        "Transport",
        df["transport_mode"].unique(),
        default=df["transport_mode"].unique()
    )

    day_filter = st.multiselect(
        "Day Type",
        df["day_type"].unique(),
        default=df["day_type"].unique()
    )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FILTER DATA
# -----------------------------
filtered_df = df[
    (df["season"].isin(season_filter)) &
    (df["transport_mode"].isin(transport_filter)) &
    (df["day_type"].isin(day_filter))
]

# -----------------------------
# KPI METRICS
# -----------------------------
m1,m2,m3,m4 = st.columns(4)

m1.metric("Routes", len(filtered_df))
m2.metric("Avg Satisfaction", round(filtered_df["satisfaction_rating"].mean(),2))
m3.metric("Avg Budget", round(filtered_df["user_budget"].mean(),2))
m4.metric("Avg Travel Time", round(filtered_df["estimated_travel_time_hr"].mean(),2))

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1,tab2,tab3,tab4,tab5 = st.tabs([
"Traffic",
"Demand",
"Budget",
"Costs",
"India Map"
])

# -----------------------------
# TRAFFIC
# -----------------------------
with tab1:

    st.markdown("### Traffic Density vs Travel Time")

    st.caption("Uses **traffic_density** and **estimated_travel_time_hr** to show how congestion increases travel duration.")

    filtered_df["traffic_density_bin"]=pd.cut(
        filtered_df["traffic_density"],
        bins=[0,0.2,0.4,0.6,0.8,1],
        labels=["Very Low","Low","Medium","High","Very High"]
    )

    traffic_time=filtered_df.groupby("traffic_density_bin")["estimated_travel_time_hr"].mean().reset_index()

    fig=px.line(
        traffic_time,
        x="traffic_density_bin",
        y="estimated_travel_time_hr",
        markers=True
    )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Traffic Distribution")

    st.caption("Compares weekday vs weekend traffic using **traffic_density**.")

    fig2=px.violin(
        filtered_df,
        x="day_type",
        y="traffic_density",
        color="day_type",
        box=True
    )

    st.plotly_chart(fig2,use_container_width=True)


# -----------------------------
# DEMAND
# -----------------------------
with tab2:

    st.markdown("### Travel Demand Heatmap")

    st.caption("Uses **season**, **day_type**, and **popularity_score** to identify peak tourism periods.")

    fig3=px.density_heatmap(
        filtered_df,
        x="season",
        y="day_type",
        z="popularity_score",
        histfunc="avg"
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.markdown("### Travel Preference Hierarchy")

    st.caption("Shows relationships between **season**, **transport_mode**, and **destination_type**.")

    sunburst_data=filtered_df.groupby(["season","transport_mode","destination_type"])["popularity_score"].sum().reset_index()

    fig4=px.sunburst(
        sunburst_data,
        path=["season","transport_mode","destination_type"],
        values="popularity_score"
    )

    st.plotly_chart(fig4,use_container_width=True)


# -----------------------------
# BUDGET
# -----------------------------
with tab3:

    st.markdown("### Budget vs Satisfaction")

    st.caption("Explores how **user_budget** affects **satisfaction_rating** across transport types.")

    fig5=px.scatter(
        filtered_df,
        x="user_budget",
        y="satisfaction_rating",
        color="transport_mode"
    )

    st.plotly_chart(fig5,use_container_width=True)

    st.markdown("### Traveler Preference Radar")

    st.caption("Normalizes travel factors to compare transport modes.")

    radar=filtered_df.groupby("transport_mode")[[
        "user_budget",
        "user_time_constraint_hr",
        "popularity_score",
        "traffic_density",
        "satisfaction_rating"
    ]].mean()

    scaler=MinMaxScaler()
    radar_scaled=scaler.fit_transform(radar)

    categories=["Budget","Time","Popularity","Traffic","Satisfaction"]

    fig6=go.Figure()

    for i,mode in enumerate(radar.index):

        values=radar_scaled[i].tolist()
        values+=values[:1]

        fig6.add_trace(go.Scatterpolar(
            r=values,
            theta=categories+[categories[0]],
            fill="toself",
            name=mode
        ))

    fig6.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])))

    st.plotly_chart(fig6,use_container_width=True)


# -----------------------------
# COSTS
# -----------------------------
with tab4:

    st.markdown("### Travel Cost Breakdown")

    st.caption("Average values from **entry_fee**, **accommodation_cost**, and **food_cost**.")

    entry=filtered_df["entry_fee"].mean()
    accom=filtered_df["accommodation_cost"].mean()
    food=filtered_df["food_cost"].mean()

    fig7=go.Figure(data=[go.Sankey(
        node=dict(label=["Total","Entry","Accommodation","Food"]),
        link=dict(source=[0,0,0],target=[1,2,3],value=[entry,accom,food])
    )])

    st.plotly_chart(fig7,use_container_width=True)


# -----------------------------
# INDIA MAP
# -----------------------------
with tab5:

    st.markdown("### India Travel Routes")

    st.caption("Shows routes between cities colored by transport mode.")

    city_coords={
    "Delhi":(28.6139,77.2090),
    "Mumbai":(19.0760,72.8777),
    "Bangalore":(12.9716,77.5946),
    "Chennai":(13.0827,80.2707),
    "Kolkata":(22.5726,88.3639),
    "Agra":(27.1767,78.0081),
    "Goa":(15.2993,74.1240),
    "Shimla":(31.1048,77.1734),
    "Ooty":(11.4064,76.6932),
    "Mahabalipuram":(12.6208,80.1937)
    }

    filtered_df["start_lat"]=filtered_df["start_location"].map(lambda x:city_coords.get(x,(None,None))[0])
    filtered_df["start_lon"]=filtered_df["start_location"].map(lambda x:city_coords.get(x,(None,None))[1])
    filtered_df["end_lat"]=filtered_df["end_location"].map(lambda x:city_coords.get(x,(None,None))[0])
    filtered_df["end_lon"]=filtered_df["end_location"].map(lambda x:city_coords.get(x,(None,None))[1])

    m=folium.Map(location=[22.5,80],zoom_start=5,tiles="cartodbpositron")

    colors={
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

            color=colors.get(str(row["transport_mode"]).lower(),"black")

            folium.PolyLine(locations=[start,end],color=color,weight=3).add_to(m)

    st_folium(m,width=900,height=600)

    st.markdown("""
**Transport Legend**

Blue → Car  
Green → Train  
Orange → Bike  
Purple → Walk  
Brown → Bus
""")
