import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import xmltodict as xtd
import os
from collections import defaultdict
import heapq

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodNav – Relief Route Finder",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0f1e;
    color: #e0eaff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a35 0%, #0a0f1e 100%);
    border-right: 1px solid #1e3a6e;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7;
}

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #0a1628 40%, #0d1f3c 100%);
    border: 1px solid #1e4080;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(79,195,247,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero h1 span { color: #4fc3f7; }
.hero p {
    color: #8ab4d4;
    font-size: 1rem;
    margin: 0;
    font-family: 'Space Mono', monospace;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 2px;
}
.badge-clear   { background: #0d3320; color: #4caf82; border: 1px solid #2d6b48; }
.badge-light   { background: #2d2a00; color: #ffd54f; border: 1px solid #6b5e00; }
.badge-heavy   { background: #2d1200; color: #ff8a65; border: 1px solid #8b3a00; }
.badge-blocked { background: #2d0000; color: #ef5350; border: 1px solid #8b0000; }

/* Metric cards */
.metric-card {
    background: #0d1a35;
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #4fc3f7;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #6a8fae;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* Alert box */
.alert-box {
    background: #1a0a00;
    border-left: 4px solid #ff6b35;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ffb385;
}
.alert-box.info {
    background: #001a2d;
    border-left-color: #4fc3f7;
    color: #90caf9;
}
.alert-box.success {
    background: #001a0d;
    border-left-color: #4caf82;
    color: #a5d6a7;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4fc3f7;
    border-bottom: 1px solid #1e3a6e;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important;
    border: 1px solid #1e88e5 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    box-shadow: 0 0 20px rgba(79,195,247,0.3) !important;
}

/* Sliders & inputs */
[data-testid="stSlider"] .stSlider { color: #4fc3f7 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #8ab4d4 !important; font-size: 0.85rem !important; }

/* Rain level colors on selectbox */
</style>
""", unsafe_allow_html=True)

# ─── OSM Parsing ──────────────────────────────────────────────────────────────
ROAD_VALS = {
    'highway', 'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'road', 'residential', 'living_street',
    'service', 'services', 'motorway_junction'
}

FLOOD_PENALTY = {
    "🟢 Clear":        1.0,
    "🟡 Light Rain":   2.5,
    "🟠 Heavy Rain":   6.0,
    "🔴 Flooded/Blocked": float('inf'),
}

FLOOD_COLOR = {
    "🟢 Clear":        "#4caf82",
    "🟡 Light Rain":   "#ffd54f",
    "🟠 Heavy Rain":   "#ff8a65",
    "🔴 Flooded/Blocked": "#ef5350",
}

@st.cache_data(show_spinner=False)
def parse_osm(filepath):
    with open(filepath, "rb") as f:
        map_osm = xtd.parse(f)['osm']

    bounds = [
        map_osm['bounds']['@minlon'],
        map_osm['bounds']['@maxlon'],
        map_osm['bounds']['@minlat'],
        map_osm['bounds']['@maxlat'],
    ]

    Node = map_osm['node']
    Nnodes = len(Node)
    node_ids_list = [float(Node[i]['@id']) for i in range(Nnodes)]
    xy = [[float(Node[i]['@lat']), float(Node[i]['@lon'])] for i in range(Nnodes)]
    node_id_to_idx = {node_ids_list[i]: i for i in range(Nnodes)}

    Way = map_osm['way']
    Nways = len(Way)
    ways = []
    for i in range(Nways):
        w = Way[i]
        nd_refs = [float(nd['@ref']) for nd in w['nd']]
        tags = []
        if 'tag' in w:
            tags = w['tag'] if isinstance(w['tag'], list) else [w['tag']]
        ways.append({'id': float(w['@id']), 'nodes': nd_refs, 'tags': tags})

    # Build adjacency: {(i,j): way_index}
    adj = defaultdict(dict)  # adj[u][v] = way_idx
    for wi, w in enumerate(ways):
        is_road = any(t['@k'] in ROAD_VALS for t in w['tags'])
        if not is_road:
            continue
        nodeset = w['nodes']
        for a in range(len(nodeset)):
            ia = node_id_to_idx.get(nodeset[a], -1)
            if ia == -1: continue
            for b in range(a + 1, len(nodeset)):
                ib = node_id_to_idx.get(nodeset[b], -1)
                if ib == -1: continue
                if ib not in adj[ia]:
                    adj[ia][ib] = wi
                    adj[ib][ia] = wi

    return {
        'bounds': bounds,
        'Nnodes': Nnodes,
        'xy': xy,
        'node_id_to_idx': node_id_to_idx,
        'node_ids_list': node_ids_list,
        'ways': ways,
        'adj': dict(adj),
    }


def dijkstra_flood(source, dest, osm, flood_levels):
    """Penalized Dijkstra. flood_levels: {way_idx: penalty_multiplier}"""
    adj = osm['adj']
    Nnodes = osm['Nnodes']

    dist = [float('inf')] * Nnodes
    prev = [-1] * Nnodes
    dist[source] = 0

    pq = [(0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dest:
            break

        for v, way_idx in adj.get(u, {}).items():
            penalty = flood_levels.get(way_idx, 1.0)
            if penalty == float('inf'):
                continue  # blocked
            cost = d + penalty
            if cost < dist[v]:
                dist[v] = cost
                prev[v] = u
                heapq.heappush(pq, (cost, v))

    # Reconstruct path
    if dist[dest] == float('inf'):
        return None, float('inf')

    path = []
    cur = dest
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[dest]


def build_base_map(osm, flood_levels, selected_ways=None):
    bounds = osm['bounds']
    cx = (float(bounds[2]) + float(bounds[3])) / 2
    cy = (float(bounds[0]) + float(bounds[1])) / 2
    m = folium.Map(location=[cx, cy], zoom_start=15, tiles="CartoDB dark_matter")

    # Draw all road ways with flood color
    ways = osm['ways']
    adj = osm['adj']
    xy = osm['xy']

    # Draw flood-colored edges
    drawn_pairs = set()
    for u, neighbors in adj.items():
        for v, wi in neighbors.items():
            pair = (min(u, v), max(u, v))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            penalty = flood_levels.get(wi, 1.0)
            if penalty == float('inf'):
                color = FLOOD_COLOR["🔴 Flooded/Blocked"]
                weight = 3
                dash = "5,5"
            elif penalty >= 6.0:
                color = FLOOD_COLOR["🟠 Heavy Rain"]
                weight = 3
                dash = None
            elif penalty >= 2.5:
                color = FLOOD_COLOR["🟡 Light Rain"]
                weight = 2
                dash = None
            else:
                color = "#2a4a6e"
                weight = 1.5
                dash = None

            coords = [xy[u], xy[v]]
            folium.PolyLine(
                coords, color=color, weight=weight, opacity=0.75,
                dash_array=dash
            ).add_to(m)

    # Draw node markers for all road nodes only (subsample for performance)
    shown = set()
    for u, neighbors in adj.items():
        shown.add(u)
        for v in neighbors:
            shown.add(v)

    return m


def draw_path_on_map(m, path, osm, flood_levels):
    xy = osm['xy']
    coords = [xy[n] for n in path]

    # Draw colored segments
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        wi = osm['adj'].get(u, {}).get(v, -1)
        penalty = flood_levels.get(wi, 1.0)
        if penalty >= 6.0:
            seg_color = "#ff8a65"
        elif penalty >= 2.5:
            seg_color = "#ffd54f"
        else:
            seg_color = "#00e5ff"
        folium.PolyLine(
            [xy[u], xy[v]], color=seg_color, weight=6, opacity=0.95
        ).add_to(m)

    # Source marker
    folium.CircleMarker(
        xy[path[0]], radius=10, color="#4fc3f7", fill=True,
        fill_color="#4fc3f7", fill_opacity=1,
        popup=f"<b>SOURCE</b> Node {path[0]}"
    ).add_to(m)

    # Dest marker
    folium.CircleMarker(
        xy[path[-1]], radius=10, color="#ff4081", fill=True,
        fill_color="#ff4081", fill_opacity=1,
        popup=f"<b>DESTINATION</b> Node {path[-1]}"
    ).add_to(m)

    return m


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloodNav")
    st.markdown("<div class='section-header'>OSM Map File</div>", unsafe_allow_html=True)
    osm_file = st.text_input("Path to .osm file", value="Maps/mapHSR.osm",
                              help="Relative or absolute path to your .osm file")

    osm_data = None
    if os.path.exists(osm_file):
        with st.spinner("Parsing map…"):
            osm_data = parse_osm(osm_file)
        st.markdown("<div class='alert-box success'>✓ Map loaded — {} nodes, {} ways</div>".format(
            osm_data['Nnodes'], len(osm_data['ways'])), unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='alert-box'>⚠ OSM file not found. Update the path above.</div>",
            unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Flood Conditions</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.8rem; color:#6a8fae; margin-bottom:0.8rem; font-family:Space Mono,monospace;'>
    Assign flood levels to ways by their index range.<br>
    Use the table below to mark affected road segments.
    </div>
    """, unsafe_allow_html=True)

    # Global default
    default_level = st.selectbox(
        "Default road condition",
        list(FLOOD_PENALTY.keys()), index=0, key="default_flood"
    )

    st.markdown("<div class='section-header'>Add Flood Zones</div>", unsafe_allow_html=True)
    st.caption("Enter way index ranges and their flood level:")

    if 'flood_zones' not in st.session_state:
        st.session_state.flood_zones = []

    with st.expander("➕ Add a flood zone", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            fz_start = st.number_input("Way idx start", min_value=0, value=0, key="fz_start")
        with col2:
            max_ways = len(osm_data['ways']) - 1 if osm_data else 1000
            fz_end = st.number_input("Way idx end", min_value=0, value=min(50, max_ways), key="fz_end")
        fz_level = st.selectbox("Flood level", list(FLOOD_PENALTY.keys()), key="fz_level")
        if st.button("Add Zone"):
            st.session_state.flood_zones.append((int(fz_start), int(fz_end), fz_level))
            st.rerun()

    if st.session_state.flood_zones:
        st.markdown("**Active Flood Zones:**")
        for idx, (s, e, lvl) in enumerate(st.session_state.flood_zones):
            c1, c2 = st.columns([3, 1])
            with c1:
                badge_cls = {
                    "🟢 Clear": "badge-clear", "🟡 Light Rain": "badge-light",
                    "🟠 Heavy Rain": "badge-heavy", "🔴 Flooded/Blocked": "badge-blocked"
                }.get(lvl, "badge-clear")
                st.markdown(
                    f"<span class='badge {badge_cls}'>{lvl}</span> ways {s}–{e}",
                    unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"del_{idx}"):
                    st.session_state.flood_zones.pop(idx)
                    st.rerun()

    st.markdown("<div class='section-header'>Route Selection</div>", unsafe_allow_html=True)
    if osm_data:
        max_n = osm_data['Nnodes'] - 1
        source_node = st.number_input("Source Node Index", 0, max_n, 0, key="src")
        dest_node = st.number_input("Destination Node Index", 0, max_n, min(100, max_n), key="dst")
        find_route = st.button("🔍 Find Safe Route", use_container_width=True)
    else:
        find_route = False

# ─── Main Panel ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🌊 Flood<span>Nav</span></h1>
  <p>// Disaster-aware shortest path routing · Penalized Dijkstra on OSM data</p>
</div>
""", unsafe_allow_html=True)

if not osm_data:
    st.markdown("""
    <div class='alert-box'>
    🗺 Load an <b>.osm</b> file using the sidebar to get started.<br>
    Export one from <a href='https://www.openstreetmap.org/export' target='_blank' style='color:#4fc3f7'>openstreetmap.org/export</a> for your city.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build flood_levels dict
flood_levels = {}
default_penalty = FLOOD_PENALTY[default_level]
for wi in range(len(osm_data['ways'])):
    flood_levels[wi] = default_penalty

for (s, e, lvl) in st.session_state.flood_zones:
    pen = FLOOD_PENALTY[lvl]
    for wi in range(s, min(e + 1, len(osm_data['ways']))):
        flood_levels[wi] = pen

# Stats
n_clear   = sum(1 for p in flood_levels.values() if p == 1.0)
n_light   = sum(1 for p in flood_levels.values() if p == 2.5)
n_heavy   = sum(1 for p in flood_levels.values() if p == 6.0)
n_blocked = sum(1 for p in flood_levels.values() if p == float('inf'))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#4caf82'>{n_clear}</div><div class='metric-lbl'>Clear Roads</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ffd54f'>{n_light}</div><div class='metric-lbl'>Light Rain</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ff8a65'>{n_heavy}</div><div class='metric-lbl'>Heavy Rain</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ef5350'>{n_blocked}</div><div class='metric-lbl'>Blocked</div></div>", unsafe_allow_html=True)

# Legend
st.markdown("""
<div style='margin: 0.5rem 0 1rem 0; display:flex; gap:8px; flex-wrap:wrap;'>
  <span class='badge badge-clear'>● Clear (×1)</span>
  <span class='badge badge-light'>● Light Rain (×2.5)</span>
  <span class='badge badge-heavy'>● Heavy Rain (×6)</span>
  <span class='badge badge-blocked'>● Flooded/Blocked (∞)</span>
  <span class='badge' style='background:#001a2d;color:#00e5ff;border:1px solid #006080;'>— Safe Route</span>
</div>
""", unsafe_allow_html=True)

# Map
map_col, info_col = st.columns([3, 1])

with map_col:
    with st.spinner("Rendering flood map…"):
        base_map = build_base_map(osm_data, flood_levels)

    result_path = None
    result_cost = None

    if find_route:
        with st.spinner("Running penalized Dijkstra…"):
            path, cost = dijkstra_flood(
                int(source_node), int(dest_node), osm_data, flood_levels
            )
        if path is None:
            st.markdown("<div class='alert-box'>❌ No passable route found — all paths may be flooded/blocked.</div>", unsafe_allow_html=True)
        else:
            draw_path_on_map(base_map, path, osm_data, flood_levels)
            result_path = path
            result_cost = cost

    map_output = st_folium(base_map, width=None, height=520, returned_objects=[])

with info_col:
    st.markdown("<div class='section-header'>Route Info</div>", unsafe_allow_html=True)

    if result_path:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='font-size:1.4rem'>{result_cost:.1f}</div>
          <div class='metric-lbl'>Weighted Cost</div>
        </div>
        <div class='metric-card'>
          <div class='metric-val' style='font-size:1.4rem'>{len(result_path)}</div>
          <div class='metric-lbl'>Nodes in Path</div>
        </div>
        """, unsafe_allow_html=True)

        # Segment breakdown
        flood_counts = {"🟢 Clear": 0, "🟡 Light Rain": 0, "🟠 Heavy Rain": 0}
        for i in range(len(result_path) - 1):
            u, v = result_path[i], result_path[i+1]
            wi = osm_data['adj'].get(u, {}).get(v, -1)
            p = flood_levels.get(wi, 1.0)
            if p >= 6.0:
                flood_counts["🟠 Heavy Rain"] += 1
            elif p >= 2.5:
                flood_counts["🟡 Light Rain"] += 1
            else:
                flood_counts["🟢 Clear"] += 1

        st.markdown("<div class='section-header'>Segment Breakdown</div>", unsafe_allow_html=True)
        for label, cnt in flood_counts.items():
            if cnt > 0:
                badge_cls = {
                    "🟢 Clear": "badge-clear",
                    "🟡 Light Rain": "badge-light",
                    "🟠 Heavy Rain": "badge-heavy"
                }[label]
                st.markdown(
                    f"<span class='badge {badge_cls}'>{label}</span> — {cnt} segments",
                    unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Path Nodes</div>", unsafe_allow_html=True)
        st.code(" → ".join(str(n) for n in result_path[:10]) +
                (" …" if len(result_path) > 10 else ""),
                language=None)
    else:
        st.markdown("""
        <div class='alert-box info'>
        Set source & destination in the sidebar, then click<br>
        <b>Find Safe Route</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#4a6a8a; font-size:0.8rem; font-family:Space Mono,monospace; margin-top:1rem;'>
        HOW IT WORKS:<br><br>
        Roads are weighted by flood level.<br><br>
        × 1.0 → Clear<br>
        × 2.5 → Light rain<br>
        × 6.0 → Heavy rain<br>
        × ∞  → Blocked<br><br>
        Dijkstra finds minimum-cost path avoiding flooded segments.
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#2a4a6e; font-size:0.75rem;
     font-family:Space Mono,monospace; margin-top:2rem; padding-top:1rem;
     border-top:1px solid #111e38;'>
FloodNav · Penalized Dijkstra on OpenStreetMap · Built for disaster relief routing
</div>
""", unsafe_allow_html=True)