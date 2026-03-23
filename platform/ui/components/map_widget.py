"""Interactive Folium map component with click-to-select."""

import streamlit as st
import folium
from streamlit_folium import st_folium
from typing import Optional


def render_site_map(
    center_lat: float = 35.0,
    center_lon: float = -119.9,
    zoom: int = 6,
    markers: list[dict] | None = None,
    height: int = 500,
    key: str = "site_map",
) -> tuple[Optional[float], Optional[float]]:
    """Render an interactive map. Returns (lat, lon) if the user clicked.

    Args:
        center_lat: Map center latitude.
        center_lon: Map center longitude.
        zoom: Initial zoom level.
        markers: List of dicts with keys: lat, lon, popup, color (optional).
        height: Map height in pixels.
        key: Streamlit widget key (must be unique per page).

    Returns:
        (clicked_lat, clicked_lon) or (None, None).
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    # Add a crosshair marker for the currently selected location
    folium.Marker(
        [center_lat, center_lon],
        popup=f"Selected: {center_lat:.4f}, {center_lon:.4f}",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    # Add any extra markers (e.g. EIA generators)
    if markers:
        for marker in markers:
            folium.CircleMarker(
                location=[marker["lat"], marker["lon"]],
                radius=5,
                popup=marker.get("popup", ""),
                color=marker.get("color", "blue"),
                fill=True,
                fill_opacity=0.7,
            ).add_to(m)

    map_data = st_folium(
        m,
        width=None,  # full width
        height=height,
        returned_objects=["last_clicked"],
        key=key,
    )

    if map_data and map_data.get("last_clicked"):
        return (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"],
        )
    return None, None
