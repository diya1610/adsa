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

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a35 0%, #0a0f1e 100%);
    border-right: 1px solid #1e3a6e;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7;
}

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
.badge-mst     { background: #0d2d1a; color: #69f0ae; border: 1px solid #1b5e40; }

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
.alert-box.mst {
    background: #001a0d;
    border-left-color: #69f0ae;
    color: #69f0ae;
}

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

[data-testid="stSlider"] .stSlider { color: #4fc3f7 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #8ab4d4 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
ROAD_VALS = {
    'highway', 'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'road', 'residential', 'living_street',
    'service', 'services', 'motorway_junction'
}

FLOOD_PENALTY = {
    "🟢 Clear":               1.0,
    "🟡 Light Rain":          2.5,
    "🟠 Heavy Rain":          6.0,
    "🔴 Flooded/Blocked":     float('inf'),
}

FLOOD_COLOR = {
    "🟢 Clear":               "#4caf82",
    "🟡 Light Rain":          "#ffd54f",
    "🟠 Heavy Rain":          "#ff8a65",
    "🔴 Flooded/Blocked":     "#ef5350",
}

NODE_ICONS = {
    "🏥 Hospital":   ("red",    "plus"),
    "🏭 Depot":      ("orange", "home"),
    "🏠 Shelter":    ("blue",   "info-sign"),
    "📍 Checkpoint": ("purple", "map-marker"),
}

# ─── OSM Parsing ──────────────────────────────────────────────────────────────
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

    adj = defaultdict(dict)
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


# ─── Dijkstra ─────────────────────────────────────────────────────────────────
def dijkstra_flood(source, dest, osm, flood_levels):
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
                continue
            cost = d + penalty
            if cost < dist[v]:
                dist[v] = cost
                prev[v] = u
                heapq.heappush(pq, (cost, v))

    if dist[dest] == float('inf'):
        return None, float('inf')

    path = []
    cur = dest
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[dest]


# ─── Union-Find (for Kruskal's) ───────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ─── Kruskal's MST on key nodes (via pairwise Dijkstra) ───────────────────────
def kruskal_relief_network(key_nodes, osm, flood_levels):
    """
    Build a minimum-cost spanning tree connecting all key_nodes.

    Strategy:
      1. Run Dijkstra from each key node to all others → pairwise costs & paths.
      2. Treat key_nodes as vertices of a complete graph with flood-penalized weights.
      3. Run Kruskal's on that complete graph to get the MST.
      4. Return the MST edges (as actual road-node paths) and total cost.
    """
    n = len(key_nodes)
    if n < 2:
        return [], 0.0, "Need at least 2 key nodes."

    # Step 1 — pairwise shortest paths between key nodes
    pair_costs = {}   # (i,j) -> cost
    pair_paths = {}   # (i,j) -> list of node indices

    for i in range(n):
        for j in range(i + 1, n):
            path, cost = dijkstra_flood(key_nodes[i], key_nodes[j], osm, flood_levels)
            if path is None:
                pair_costs[(i, j)] = float('inf')
                pair_paths[(i, j)] = []
            else:
                pair_costs[(i, j)] = cost
                pair_paths[(i, j)] = path

    # Step 2 — sort all edges by cost
    edges = sorted([(pair_costs[(i, j)], i, j) for i in range(n) for j in range(i+1, n)])

    # Step 3 — Kruskal's
    uf = UnionFind(n)
    mst_edges   = []   # list of (cost, i, j, path)
    mst_cost    = 0.0
    blocked_any = False

    for cost, i, j in edges:
        if cost == float('inf'):
            blocked_any = True
            continue
        if uf.union(i, j):
            mst_edges.append((cost, i, j, pair_paths[(i, j)]))
            mst_cost += cost
            if len(mst_edges) == n - 1:
                break

    # Check connectivity
    roots = {uf.find(i) for i in range(n)}
    if len(roots) > 1:
        return mst_edges, mst_cost, "⚠ Network is disconnected — some nodes are unreachable."

    msg = "✓ Relief network MST computed."
    if blocked_any:
        msg += " Some pairs were blocked and skipped."
    return mst_edges, mst_cost, msg


# ─── Map builders ─────────────────────────────────────────────────────────────
def build_base_map(osm, flood_levels):
    bounds = osm['bounds']
    cx = (float(bounds[2]) + float(bounds[3])) / 2
    cy = (float(bounds[0]) + float(bounds[1])) / 2
    m = folium.Map(location=[cx, cy], zoom_start=15, tiles="CartoDB dark_matter")

    adj = osm['adj']
    xy  = osm['xy']
    drawn_pairs = set()

    for u, neighbors in adj.items():
        for v, wi in neighbors.items():
            pair = (min(u, v), max(u, v))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            penalty = flood_levels.get(wi, 1.0)
            if penalty == float('inf'):
                color, weight, dash = FLOOD_COLOR["🔴 Flooded/Blocked"], 3, "5,5"
            elif penalty >= 6.0:
                color, weight, dash = FLOOD_COLOR["🟠 Heavy Rain"], 3, None
            elif penalty >= 2.5:
                color, weight, dash = FLOOD_COLOR["🟡 Light Rain"], 2, None
            else:
                color, weight, dash = "#2a4a6e", 1.5, None

            folium.PolyLine(
                [xy[u], xy[v]], color=color, weight=weight,
                opacity=0.75, dash_array=dash
            ).add_to(m)

    return m


def draw_dijkstra_path(m, path, osm, flood_levels):
    xy = osm['xy']
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        wi = osm['adj'].get(u, {}).get(v, -1)
        penalty = flood_levels.get(wi, 1.0)
        seg_color = "#ff8a65" if penalty >= 6.0 else "#ffd54f" if penalty >= 2.5 else "#00e5ff"
        folium.PolyLine([xy[u], xy[v]], color=seg_color, weight=6, opacity=0.95).add_to(m)

    folium.CircleMarker(xy[path[0]],  radius=10, color="#4fc3f7", fill=True,
                        fill_color="#4fc3f7", fill_opacity=1,
                        popup="<b>SOURCE</b>").add_to(m)
    folium.CircleMarker(xy[path[-1]], radius=10, color="#ff4081", fill=True,
                        fill_color="#ff4081", fill_opacity=1,
                        popup="<b>DESTINATION</b>").add_to(m)
    return m


def draw_mst_on_map(m, mst_edges, key_nodes, key_node_meta, osm):
    """Draw the MST relief network in green with labeled key-node markers."""
    xy = osm['xy']

    # Draw each MST edge (the actual road path between the two key nodes)
    for cost, i, j, path in mst_edges:
        for k in range(len(path) - 1):
            u, v = path[k], path[k+1]
            folium.PolyLine(
                [xy[u], xy[v]],
                color="#69f0ae",
                weight=5,
                opacity=0.9,
                tooltip=f"MST edge cost: {cost:.1f}"
            ).add_to(m)

    # Draw key node markers
    for idx, node_idx in enumerate(key_nodes):
        meta = key_node_meta[idx]
        label    = meta.get("label", f"Node {node_idx}")
        node_type = meta.get("type", "📍 Checkpoint")
        marker_color, icon_name = NODE_ICONS.get(node_type, ("purple", "map-marker"))

        folium.Marker(
            location=xy[node_idx],
            popup=folium.Popup(f"<b>{label}</b><br>{node_type}<br>Node idx: {node_idx}", max_width=200),
            tooltip=label,
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="glyphicon")
        ).add_to(m)

    return m


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloodNav")

    # ── OSM file ──
    st.markdown("<div class='section-header'>OSM Map File</div>", unsafe_allow_html=True)
    osm_file = st.text_input("Path to .osm file", value="Maps/mapHSR.osm")

    osm_data = None
    if os.path.exists(osm_file):
        with st.spinner("Parsing map…"):
            osm_data = parse_osm(osm_file)
        st.markdown("<div class='alert-box success'>✓ Map loaded — {} nodes, {} ways</div>".format(
            osm_data['Nnodes'], len(osm_data['ways'])), unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-box'>⚠ OSM file not found.</div>", unsafe_allow_html=True)

    # ── Flood zones ──
    st.markdown("<div class='section-header'>Flood Conditions</div>", unsafe_allow_html=True)
    default_level = st.selectbox("Default road condition", list(FLOOD_PENALTY.keys()), index=0)

    st.markdown("<div class='section-header'>Add Flood Zones</div>", unsafe_allow_html=True)
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
                badge_cls = {"🟢 Clear": "badge-clear", "🟡 Light Rain": "badge-light",
                             "🟠 Heavy Rain": "badge-heavy", "🔴 Flooded/Blocked": "badge-blocked"}.get(lvl, "badge-clear")
                st.markdown(f"<span class='badge {badge_cls}'>{lvl}</span> ways {s}–{e}", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"del_{idx}"):
                    st.session_state.flood_zones.pop(idx)
                    st.rerun()

    # ── Mode selector ──
    st.markdown("<div class='section-header'>Mode</div>", unsafe_allow_html=True)
    mode = st.radio("Algorithm", ["🔵 Dijkstra – Single Route", "🟢 Kruskal – Relief Network"], index=0)

    # ── Dijkstra controls ──
    if osm_data and "Dijkstra" in mode:
        st.markdown("<div class='section-header'>Route Selection</div>", unsafe_allow_html=True)
        max_n = osm_data['Nnodes'] - 1
        source_node = st.number_input("Source Node Index", 0, max_n, 0, key="src")
        dest_node   = st.number_input("Destination Node Index", 0, max_n, min(100, max_n), key="dst")
        find_route  = st.button("🔍 Find Safe Route", use_container_width=True)
        run_kruskal = False
    # ── Kruskal controls ──
    elif osm_data and "Kruskal" in mode:
        st.markdown("<div class='section-header'>Key Relief Nodes</div>", unsafe_allow_html=True)
        st.caption("Add hospitals, depots, shelters and checkpoints to connect.")

        if 'key_nodes' not in st.session_state:
            st.session_state.key_nodes      = []   # list of node indices
            st.session_state.key_node_meta  = []   # list of {label, type}

        with st.expander("➕ Add a key node", expanded=True):
            max_n    = osm_data['Nnodes'] - 1
            kn_idx   = st.number_input("Node Index", 0, max_n, 0, key="kn_idx")
            kn_label = st.text_input("Label (e.g. City Hospital)", value="Node", key="kn_label")
            kn_type  = st.selectbox("Type", list(NODE_ICONS.keys()), key="kn_type")
            if st.button("Add Node", use_container_width=True):
                st.session_state.key_nodes.append(int(kn_idx))
                st.session_state.key_node_meta.append({"label": kn_label, "type": kn_type})
                st.rerun()

        if st.session_state.key_nodes:
            st.markdown("**Key Nodes:**")
            for idx, (ni, meta) in enumerate(zip(st.session_state.key_nodes, st.session_state.key_node_meta)):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f"<span class='badge badge-mst'>{meta['type']}</span> **{meta['label']}** (#{ni})",
                        unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"kn_del_{idx}"):
                        st.session_state.key_nodes.pop(idx)
                        st.session_state.key_node_meta.pop(idx)
                        st.rerun()

            if st.button("🌐 Build Relief Network", use_container_width=True):
                run_kruskal = True
            else:
                run_kruskal = False
        else:
            st.markdown("<div class='alert-box info'>Add at least 2 key nodes to build the network.</div>",
                        unsafe_allow_html=True)
            run_kruskal = False

        find_route = False
    else:
        find_route = run_kruskal = False


# ─── Main Panel ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🌊 Flood<span>Nav</span></h1>
  <p>// Disaster-aware routing · Penalized Dijkstra · Kruskal's Relief Network MST</p>
</div>
""", unsafe_allow_html=True)

if not osm_data:
    st.markdown("""
    <div class='alert-box'>
    🗺 Load an <b>.osm</b> file using the sidebar to get started.<br>
    Export one from <a href='https://www.openstreetmap.org/export' target='_blank' style='color:#4fc3f7'>openstreetmap.org/export</a>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build flood_levels dict
flood_levels = {wi: FLOOD_PENALTY[default_level] for wi in range(len(osm_data['ways']))}
for (s, e, lvl) in st.session_state.flood_zones:
    for wi in range(s, min(e + 1, len(osm_data['ways']))):
        flood_levels[wi] = FLOOD_PENALTY[lvl]

# Stats row
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
if "Dijkstra" in mode:
    st.markdown("""
    <div style='margin:0.5rem 0 1rem 0;display:flex;gap:8px;flex-wrap:wrap;'>
      <span class='badge badge-clear'>● Clear (×1)</span>
      <span class='badge badge-light'>● Light Rain (×2.5)</span>
      <span class='badge badge-heavy'>● Heavy Rain (×6)</span>
      <span class='badge badge-blocked'>● Flooded (∞)</span>
      <span class='badge' style='background:#001a2d;color:#00e5ff;border:1px solid #006080;'>— Safe Route</span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='margin:0.5rem 0 1rem 0;display:flex;gap:8px;flex-wrap:wrap;'>
      <span class='badge badge-clear'>● Clear (×1)</span>
      <span class='badge badge-light'>● Light Rain (×2.5)</span>
      <span class='badge badge-heavy'>● Heavy Rain (×6)</span>
      <span class='badge badge-blocked'>● Flooded (∞)</span>
      <span class='badge badge-mst'>— MST Relief Network</span>
      <span class='badge' style='background:#0d2d1a;color:#ff4081;border:1px solid #880e4f;'>● Hospital</span>
      <span class='badge' style='background:#0d2d1a;color:#ffd54f;border:1px solid #6b5e00;'>● Depot</span>
      <span class='badge' style='background:#0d2d1a;color:#4fc3f7;border:1px solid #006080;'>● Shelter</span>
    </div>""", unsafe_allow_html=True)

# ─── Map + Info layout ────────────────────────────────────────────────────────
map_col, info_col = st.columns([3, 1])

with map_col:
    with st.spinner("Rendering flood map…"):
        base_map = build_base_map(osm_data, flood_levels)

    result_path  = None
    result_cost  = None
    mst_result   = None

    # ── Dijkstra run ──
    if find_route:
        with st.spinner("Running penalized Dijkstra…"):
            path, cost = dijkstra_flood(int(source_node), int(dest_node), osm_data, flood_levels)
        if path is None:
            st.markdown("<div class='alert-box'>❌ No passable route found.</div>", unsafe_allow_html=True)
        else:
            draw_dijkstra_path(base_map, path, osm_data, flood_levels)
            result_path = path
            result_cost = cost

    # ── Kruskal run ──
    if run_kruskal:
        if len(st.session_state.key_nodes) < 2:
            st.markdown("<div class='alert-box'>⚠ Add at least 2 key nodes first.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Running Kruskal's MST on relief network…"):
                mst_edges, mst_cost, mst_msg = kruskal_relief_network(
                    st.session_state.key_nodes,
                    osm_data,
                    flood_levels
                )
            alert_cls = "mst" if "✓" in mst_msg else "alert-box"
            st.markdown(f"<div class='alert-box {alert_cls}'>{mst_msg}</div>", unsafe_allow_html=True)

            if mst_edges:
                draw_mst_on_map(base_map, mst_edges,
                                st.session_state.key_nodes,
                                st.session_state.key_node_meta,
                                osm_data)
                mst_result = (mst_edges, mst_cost)

    st_folium(base_map, width=None, height=520, returned_objects=[])

# ─── Info panel ───────────────────────────────────────────────────────────────
with info_col:
    # ── Dijkstra info ──
    if "Dijkstra" in mode:
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
            </div>""", unsafe_allow_html=True)

            flood_counts = {"🟢 Clear": 0, "🟡 Light Rain": 0, "🟠 Heavy Rain": 0}
            for i in range(len(result_path) - 1):
                u, v = result_path[i], result_path[i+1]
                wi = osm_data['adj'].get(u, {}).get(v, -1)
                p = flood_levels.get(wi, 1.0)
                if p >= 6.0:   flood_counts["🟠 Heavy Rain"] += 1
                elif p >= 2.5: flood_counts["🟡 Light Rain"] += 1
                else:          flood_counts["🟢 Clear"] += 1

            st.markdown("<div class='section-header'>Segment Breakdown</div>", unsafe_allow_html=True)
            for label, cnt in flood_counts.items():
                if cnt > 0:
                    badge_cls = {"🟢 Clear": "badge-clear", "🟡 Light Rain": "badge-light",
                                 "🟠 Heavy Rain": "badge-heavy"}[label]
                    st.markdown(f"<span class='badge {badge_cls}'>{label}</span> — {cnt} segments",
                                unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Path Nodes</div>", unsafe_allow_html=True)
            st.code(" → ".join(str(n) for n in result_path[:10]) + (" …" if len(result_path) > 10 else ""))
        else:
            st.markdown("""
            <div class='alert-box info'>Set source & destination,<br>then click <b>Find Safe Route</b>.</div>
            <div style='color:#4a6a8a;font-size:0.8rem;font-family:Space Mono,monospace;margin-top:1rem;'>
            HOW IT WORKS:<br><br>Roads weighted by flood level.<br><br>
            × 1.0 → Clear<br>× 2.5 → Light rain<br>× 6.0 → Heavy rain<br>× ∞  → Blocked<br><br>
            Dijkstra finds minimum-cost path.
            </div>""", unsafe_allow_html=True)

    # ── Kruskal info ──
    else:
        st.markdown("<div class='section-header'>Relief Network Info</div>", unsafe_allow_html=True)
        if mst_result:
            mst_edges, mst_cost = mst_result
            n_key = len(st.session_state.key_nodes)
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-val' style='font-size:1.4rem;color:#69f0ae'>{mst_cost:.1f}</div>
              <div class='metric-lbl'>Total MST Cost</div>
            </div>
            <div class='metric-card'>
              <div class='metric-val' style='font-size:1.4rem;color:#69f0ae'>{len(mst_edges)}</div>
              <div class='metric-lbl'>MST Edges</div>
            </div>
            <div class='metric-card'>
              <div class='metric-val' style='font-size:1.4rem;color:#69f0ae'>{n_key}</div>
              <div class='metric-lbl'>Key Nodes</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>MST Edge Details</div>", unsafe_allow_html=True)
            for cost, i, j, path in mst_edges:
                mi = st.session_state.key_node_meta[i]
                mj = st.session_state.key_node_meta[j]
                st.markdown(
                    f"<span class='badge badge-mst'>{mi['label']}</span> ↔ "
                    f"<span class='badge badge-mst'>{mj['label']}</span><br>"
                    f"<span style='font-family:Space Mono,monospace;font-size:0.75rem;color:#4a8a6a;'>"
                    f"cost {cost:.1f} · {len(path)} road nodes</span>",
                    unsafe_allow_html=True)
                st.markdown("")
        else:
            st.markdown("""
            <div class='alert-box info'>Add key nodes (hospitals, depots, shelters)<br>
            then click <b>Build Relief Network</b>.</div>
            <div style='color:#4a6a8a;font-size:0.8rem;font-family:Space Mono,monospace;margin-top:1rem;'>
            HOW IT WORKS:<br><br>
            1. Dijkstra finds flood-penalized shortest path between every pair of key nodes.<br><br>
            2. Kruskal's MST picks the minimum-cost set of connections that links ALL nodes.<br><br>
            Result: the cheapest road network to keep all relief sites connected.
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#2a4a6e;font-size:0.75rem;
     font-family:Space Mono,monospace;margin-top:2rem;padding-top:1rem;
     border-top:1px solid #111e38;'>
FloodNav · Dijkstra + Kruskal MST · OpenStreetMap · Built for disaster relief routing
</div>""", unsafe_allow_html=True)