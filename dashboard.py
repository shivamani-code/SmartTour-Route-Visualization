import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html


def load_data(csv_path: str = "SmartTourRoutePlanner.csv") -> pd.DataFrame:
    """Load the dataset used in the notebook."""
    df = pd.read_csv(csv_path)
    return df


def build_traffic_time_line(df: pd.DataFrame) -> go.Figure:
    df = df.copy()

    df["traffic_density_bin"] = pd.cut(
        df["traffic_density"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

    traffic_time = (
        df.groupby("traffic_density_bin")["estimated_travel_time_hr"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        traffic_time,
        x="traffic_density_bin",
        y="estimated_travel_time_hr",
        markers=True,
        title="Impact of Traffic Density on Estimated Travel Time",
        labels={
            "traffic_density_bin": "Traffic Density Level",
            "estimated_travel_time_hr": "Average Travel Time (hours)",
        },
    )

    fig.update_layout(title_x=0.5)
    return fig


def build_demand_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.density_heatmap(
        df,
        x="season",
        y="day_type",
        z="popularity_score",
        histfunc="avg",
        animation_frame="transport_mode",
        text_auto=True,
        title="Heatmap of Travel Demand (Day Type vs Season)",
    )

    fig["layout"].pop("updatemenus", None)

    fig.update_layout(template="plotly_white", title_x=0.5)

    return fig


def build_budget_satisfaction_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="user_budget",
        y="satisfaction_rating",
        color="transport_mode",
        symbol="season",
        title="User Budget vs Satisfaction Rating by Transport Mode",
        hover_data=[
            "start_location",
            "end_location",
            "season",
            "transport_mode",
            "satisfaction_rating",
        ],
    )

    fig.update_layout(title_x=0.5)
    return fig


def build_cost_sankey(df: pd.DataFrame) -> go.Figure:
    entry_fee = df["entry_fee"].mean()
    accommodation = df["accommodation_cost"].mean()
    food = df["food_cost"].mean()

    labels = [
        "Total Travel Cost",
        f"Entry Fee ({entry_fee:.2f})",
        f"Accommodation ({accommodation:.2f})",
        f"Food ({food:.2f})",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=25,
                    thickness=30,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=[0, 0, 0],
                    target=[1, 2, 3],
                    value=[entry_fee, accommodation, food],
                ),
            )
        ]
    )

    fig.update_layout(title="Travel Cost Flow Distribution")
    return fig


def build_travel_preference_sunburst(df: pd.DataFrame) -> go.Figure:
    sunburst_data = (
        df.groupby(["season", "transport_mode", "destination_type"])["popularity_score"]
        .sum()
        .reset_index()
    )

    fig = px.sunburst(
        sunburst_data,
        path=["season", "transport_mode", "destination_type"],
        values="popularity_score",
        title="Travel Preference Hierarchy",
    )

    fig.update_layout(title_x=0.5)
    return fig


def build_transport_radar(df: pd.DataFrame) -> go.Figure:
    radar = df.groupby("transport_mode")[
        [
            "user_budget",
            "user_time_constraint_hr",
            "popularity_score",
            "traffic_density",
            "satisfaction_rating",
        ]
    ].mean()

    scaler = MinMaxScaler()

    radar_scaled = pd.DataFrame(
        scaler.fit_transform(radar),
        columns=radar.columns,
        index=radar.index,
    )

    categories = ["Budget", "Travel Time", "Popularity", "Traffic", "Satisfaction"]

    fig = go.Figure()

    for mode in radar_scaled.index:
        values = radar_scaled.loc[mode].tolist()
        values += values[:1]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=mode,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Traveler Preference Comparison",
    )

    return fig


def build_traffic_violin(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="day_type",
        y="traffic_density",
        color="day_type",
        box=True,
        points="all",
    )

    fig.update_layout(
        title="Traffic Density Distribution: Weekdays vs Weekends",
        title_x=0.5,
    )

    return fig


def create_dashboard(df: pd.DataFrame) -> Dash:
    app = Dash(__name__)

    app.layout = html.Div(
        style={"fontFamily": "Arial", "padding": "16px"},
        children=[
            html.H1(
                "SmartTour Route Visualization Dashboard",
                style={"textAlign": "center"},
            ),
            dcc.Tabs(
                children=[
                    dcc.Tab(
                        label="Traffic & Time",
                        children=[
                            dcc.Graph(figure=build_traffic_time_line(df)),
                            dcc.Graph(figure=build_traffic_violin(df)),
                        ],
                    ),
                    dcc.Tab(
                        label="Demand & Preferences",
                        children=[
                            dcc.Graph(figure=build_demand_heatmap(df)),
                            dcc.Graph(figure=build_travel_preference_sunburst(df)),
                        ],
                    ),
                    dcc.Tab(
                        label="Budget & Satisfaction",
                        children=[
                            dcc.Graph(figure=build_budget_satisfaction_scatter(df)),
                            dcc.Graph(figure=build_transport_radar(df)),
                        ],
                    ),
                    dcc.Tab(
                        label="Cost Breakdown",
                        children=[
                            dcc.Graph(figure=build_cost_sankey(df)),
                        ],
                    ),
                ]
            ),
        ],
    )

    return app


def main():
    df = load_data()
    app = create_dashboard(df)

    # FIXED FOR DASH 3
    app.run(debug=False)


if __name__ == "__main__":
    main()
