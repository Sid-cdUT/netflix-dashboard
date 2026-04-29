import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Artwork Personalization Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Netflix brand colors ──────────────────────────────────────────────────────
RED       = "#E50914"
DARK_RED  = "#B20710"
BLACK     = "#141414"
DARK_GRAY = "#1F1F1F"
MID_GRAY  = "#2A2A2A"
LIGHT_GRAY = "#999999"
WHITE     = "#FFFFFF"
OFF_WHITE = "#E5E5E5"

SEGMENT_COLORS = {
    "Action Seeker":        "#E50914",
    "Comedy Buff":          "#FF6B35",
    "Documentary Lover":    "#F5C518",
    "Sci-Fi Enthusiast":    "#00A8E1",
}

VISUAL_FOCUS_COLORS = {
    "Explosion/Action": "#E50914",
    "Cast Ensemble":    "#FF6B35",
    "Solo Lead/Moody":  "#F5C518",
    "Couple/Intimacy":  "#888888",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
      background-color: #141414 !important;
      color: #E5E5E5 !important;
      font-family: 'DM Sans', sans-serif !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background-color: #1A1A1A !important;
      border-right: 1px solid #2A2A2A;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] span { color: #E5E5E5 !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; }

  /* Metric cards */
  .metric-card {
      background: #1F1F1F;
      border: 1px solid #2A2A2A;
      border-radius: 8px;
      padding: 20px 24px;
      text-align: center;
  }
  .metric-value {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 3rem;
      color: #E50914;
      line-height: 1;
  }
  .metric-label {
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #999999;
      margin-top: 6px;
  }
  .metric-sub {
      font-size: 0.8rem;
      color: #666666;
      margin-top: 4px;
  }

  /* Section header */
  .section-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.6rem;
      letter-spacing: 0.08em;
      color: #FFFFFF;
      border-left: 4px solid #E50914;
      padding-left: 12px;
      margin-bottom: 4px;
  }
  .section-sub {
      font-size: 0.78rem;
      color: #666;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
      padding-left: 16px;
  }

  /* Tab overrides */
  .stTabs [data-baseweb="tab-list"] { background: #1A1A1A; border-radius: 6px; gap: 4px; }
  .stTabs [data-baseweb="tab"] {
      color: #999 !important;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.85rem;
      letter-spacing: 0.05em;
      padding: 8px 18px;
      border-radius: 4px;
  }
  .stTabs [aria-selected="true"] {
      background: #E50914 !important;
      color: #fff !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
      background: #1F1F1F !important;
      color: #999 !important;
      font-size: 0.8rem !important;
      letter-spacing: 0.08em;
  }

  /* Netflix wordmark header */
  .netflix-header {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.4rem;
      letter-spacing: 0.06em;
      color: #E50914;
  }
  .netflix-subtitle {
      font-size: 0.82rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #666;
  }

  /* Finding badge */
  .finding-badge {
      display: inline-block;
      background: #E50914;
      color: white;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 0.9rem;
      letter-spacing: 0.1em;
      padding: 2px 10px;
      border-radius: 3px;
      margin-bottom: 6px;
  }

  /* Recommendation pill */
  .rec-pill {
      background: #1F1F1F;
      border-left: 3px solid #E50914;
      padding: 12px 16px;
      border-radius: 0 6px 6px 0;
      margin-bottom: 10px;
      font-size: 0.85rem;
      color: #CCC;
  }

  div[data-testid="stDataFrame"] { background: #1F1F1F; }
  .stSelectbox > div { background: #1F1F1F !important; }
  .stMultiSelect > div { background: #1F1F1F !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_artwork_personalization_data.csv")
    df["CTR"] = df["Clicks"] / df["Impressions"]

    # Genre match flag
    seg_to_genre = {
        "Sci-Fi Enthusiast":  "Sci-Fi",
        "Action Seeker":      "Action",
        "Documentary Lover":  "Documentary",
        "Comedy Buff":        "Comedy",
    }
    df["Genre_Match"] = df.apply(
        lambda r: seg_to_genre.get(r["User_Segment"], "") == r["Title_Genre"], axis=1
    )
    df["Match_Label"] = df["Genre_Match"].map({True: "Matched", False: "Mismatched"})
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="netflix-header">▶ FILTERS</div>', unsafe_allow_html=True)
    st.markdown("---")

    segments = st.multiselect(
        "User Segment",
        options=sorted(df["User_Segment"].unique()),
        default=sorted(df["User_Segment"].unique()),
    )

    visual_focuses = st.multiselect(
        "Visual Focus",
        options=sorted(df["Visual_Focus"].unique()),
        default=sorted(df["Visual_Focus"].unique()),
    )

    genres = st.multiselect(
        "Title Genre",
        options=sorted(df["Title_Genre"].unique()),
        default=sorted(df["Title_Genre"].unique()),
    )

    churn_filter = st.multiselect(
        "Retention Status",
        options=["Active", "Churned"],
        default=["Active", "Churned"],
    )

    variants = st.multiselect(
        "Artwork Variant",
        options=sorted(df["Artwork_Variant"].unique()),
        default=sorted(df["Artwork_Variant"].unique()),
    )

    st.markdown("---")
    st.markdown('<span class="netflix-subtitle">CS329E · Spring 2026<br>Sid & Jovan</span>', unsafe_allow_html=True)

# Apply filters
mask = (
    df["User_Segment"].isin(segments) &
    df["Visual_Focus"].isin(visual_focuses) &
    df["Title_Genre"].isin(genres) &
    df["Churn_Status"].isin(churn_filter) &
    df["Artwork_Variant"].isin(variants)
)
fdf = df[mask].copy()

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1A1A1A",
    font=dict(family="DM Sans", color=OFF_WHITE, size=12),
    title_font=dict(family="DM Sans", color=WHITE, size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=OFF_WHITE, size=11)),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor="#2A2A2A", linecolor="#333", tickcolor="#666"),
    yaxis=dict(gridcolor="#2A2A2A", linecolor="#333", tickcolor="#666"),
)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown('<div class="netflix-header" style="font-size:2.8rem;">N</div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="netflix-header">ARTWORK PERSONALIZATION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="netflix-subtitle">Content Strategy & Engineering · Visual Focus Analysis</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
if len(fdf) == 0:
    st.warning("No data matches your current filters.")
    st.stop()

overall_ctr   = fdf["CTR"].mean()
active_med    = fdf[fdf["Churn_Status"] == "Active"]["CTR"].median() if "Active" in fdf["Churn_Status"].values else 0
churned_med   = fdf[fdf["Churn_Status"] == "Churned"]["CTR"].median() if "Churned" in fdf["Churn_Status"].values else 0
churn_rate    = (fdf["Churn_Status"] == "Churned").mean()
lift_ratio    = active_med / churned_med if churned_med > 0 else 0

# Best visual focus per segment (from filtered data)
seg_vf = fdf.groupby(["User_Segment", "Visual_Focus"])["CTR"].mean()
best_vf_ctrs = seg_vf.groupby(level=0).max()
worst_vf_ctrs = seg_vf.groupby(level=0).min()
avg_lift = (best_vf_ctrs / worst_vf_ctrs).mean() if (worst_vf_ctrs > 0).all() else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{overall_ctr:.1%}</div>
        <div class="metric-label">Overall Avg CTR</div>
        <div class="metric-sub">{len(fdf):,} impressions</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{active_med:.1%}</div>
        <div class="metric-label">Active Median CTR</div>
        <div class="metric-sub">Retention benchmark</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{churned_med:.1%}</div>
        <div class="metric-label">Churned Median CTR</div>
        <div class="metric-sub">Churn warning threshold</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{lift_ratio:.1f}x</div>
        <div class="metric-label">Active vs Churned CTR</div>
        <div class="metric-sub">Retention signal gap</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{churn_rate:.1%}</div>
        <div class="metric-label">30-Day Churn Rate</div>
        <div class="metric-sub">In filtered dataset</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  FINDING 1 — Visual Focus",
    "📉  FINDING 2 — Churn Signal",
    "🔀  FINDING 3 — Genre Match",
    "🎬  DEEP DIVE — Sci-Fi",
    "🔍  VARIANT ANALYSIS",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Visual Focus Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="finding-badge">FINDING 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Focus Drives Up to 3.2x Higher CTR</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">SELECT A SEGMENT OR VISUAL FOCUS BELOW TO CROSS-FILTER ALL CHARTS</div>', unsafe_allow_html=True)

    col_heat, col_bar = st.columns([3, 2])

    with col_heat:
        heatmap_data = fdf.groupby(["User_Segment", "Visual_Focus"])["CTR"].mean().reset_index()
        pivot = heatmap_data.pivot(index="User_Segment", columns="Visual_Focus", values="CTR")

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#1A1A1A"], [0.3, "#4A0A0A"], [0.6, "#8B1111"], [1.0, "#E50914"]],
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}",
            textfont=dict(size=14, color="white", family="DM Sans"),
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=OFF_WHITE, size=10),
                tickformat=".0f",
                ticksuffix="%",
                bgcolor="rgba(0,0,0,0)",
                outlinecolor="#333",
            ),
            hovertemplate="<b>%{y}</b><br>Visual Focus: %{x}<br>Avg CTR: %{text}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Segment × Visual Focus — Avg CTR",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1A1A1A",
            font=dict(family="DM Sans", color="#E5E5E5", size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="Visual Focus", tickangle=-20, gridcolor="#2A2A2A", linecolor="#333"),
            yaxis=dict(title="User Segment", gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_bar:
        # Best vs worst per segment
        best_worst = heatmap_data.copy()
        seg_best  = best_worst.loc[best_worst.groupby("User_Segment")["CTR"].idxmax()].copy()
        seg_worst = best_worst.loc[best_worst.groupby("User_Segment")["CTR"].idxmin()].copy()
        seg_best["Type"]  = "Best Focus"
        seg_worst["Type"] = "Worst Focus"
        bw = pd.concat([seg_best, seg_worst])

        fig_bw = px.bar(
            bw, x="CTR", y="User_Segment", color="Type", barmode="group",
            orientation="h",
            color_discrete_map={"Best Focus": RED, "Worst Focus": "#444444"},
            text=bw["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Avg CTR", "User_Segment": ""},
            hover_data={"Visual_Focus": True},
        )
        fig_bw.update_traces(textposition="outside", textfont_size=11)
        fig_bw.update_layout(
            title="Best vs Worst Visual Focus per Segment",
            **PLOTLY_LAYOUT,
            xaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            yaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=OFF_WHITE)),
        )
        st.plotly_chart(fig_bw, use_container_width=True)

    # Per-segment selector
    st.markdown("---")
    st.markdown("**Drill into a segment:**")
    selected_seg = st.selectbox(
        "Select segment to explore visual focus breakdown",
        options=sorted(fdf["User_Segment"].unique()),
        key="seg_selector_t1",
    )
    seg_df = fdf[fdf["User_Segment"] == selected_seg]
    vf_breakdown = seg_df.groupby("Visual_Focus")["CTR"].mean().reset_index().sort_values("CTR", ascending=False)
    vf_breakdown["color"] = [RED if i == 0 else ("#333" if i == len(vf_breakdown)-1 else "#555")
                              for i in range(len(vf_breakdown))]

    fig_seg = go.Figure(go.Bar(
        x=vf_breakdown["Visual_Focus"],
        y=vf_breakdown["CTR"] * 100,
        marker_color=vf_breakdown["color"],
        text=[f"{v:.1f}%" for v in vf_breakdown["CTR"] * 100],
        textposition="outside",
        textfont=dict(color=WHITE, size=13),
        hovertemplate="<b>%{x}</b><br>Avg CTR: %{text}<extra></extra>",
    ))
    fig_seg.update_layout(
        title=f"{selected_seg} — CTR by Visual Focus (ranked)",
        **PLOTLY_LAYOUT,
        yaxis=dict(title="Avg CTR (%)", ticksuffix="%", gridcolor="#2A2A2A", linecolor="#333"),
        xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    # Insight callout
    best_row = vf_breakdown.iloc[0]
    worst_row = vf_breakdown.iloc[-1]
    lift = best_row["CTR"] / worst_row["CTR"] if worst_row["CTR"] > 0 else 0
    st.markdown(f"""<div class="rec-pill">
        💡 <b>{selected_seg}</b> responds best to <b>{best_row['Visual_Focus']}</b> ({best_row['CTR']:.1%} CTR) —
        <b>{lift:.1f}x</b> higher than {worst_row['Visual_Focus']} ({worst_row['CTR']:.1%} CTR).
        Serving the right visual focus is a zero-infrastructure win.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Churn Signal
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="finding-badge">FINDING 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Low CTR Is an Early Warning Signal for Churn</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ACTIVE SUBSCRIBERS CLICK 2.6X MORE — CTR PREDICTS RETENTION BEFORE CANCELLATION</div>', unsafe_allow_html=True)

    col_scatter, col_dist = st.columns([3, 2])

    with col_scatter:
        color_map = {"Active": RED, "Churned": "#555555"}
        fig_scatter = px.scatter(
            fdf, x="CTR", y="Time_Spent_on_Service",
            color="Churn_Status",
            color_discrete_map=color_map,
            opacity=0.65,
            labels={"CTR": "Click-Through Rate (CTR)", "Time_Spent_on_Service": "Avg Time on Platform (min)", "Churn_Status": "Status"},
            hover_data={"User_Segment": True, "Visual_Focus": True, "Artwork_Variant": True},
        )
        # Add median lines
        for status, color in [("Active", RED), ("Churned", "#888")]:
            subset = fdf[fdf["Churn_Status"] == status]
            if len(subset):
                med_ctr = subset["CTR"].median()
                fig_scatter.add_vline(x=med_ctr, line_dash="dash", line_color=color,
                                       annotation_text=f"Median {status}: {med_ctr:.1%}",
                                       annotation_font_color=color, annotation_font_size=11)

        fig_scatter.update_layout(
            title="CTR vs Time Spent — Active vs Churned Subscribers",
            **PLOTLY_LAYOUT,
            xaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            yaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=400,
            legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_dist:
        fig_box = go.Figure()
        for status in ["Active", "Churned"]:
            subset = fdf[fdf["Churn_Status"] == status]
            if len(subset):
                fig_box.add_trace(go.Box(
                    y=subset["CTR"] * 100,
                    name=status,
                    marker_color=color_map[status],
                    boxmean=True,
                    line=dict(color=color_map[status]),
                    fillcolor=color_map[status] + "44",
                    hovertemplate=f"<b>{status}</b><br>CTR: %{{y:.1f}}%<extra></extra>",
                ))
        fig_box.update_layout(
            title="CTR Distribution by Retention Status",
            **PLOTLY_LAYOUT,
            yaxis=dict(title="CTR (%)", ticksuffix="%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Churn by segment + visual focus
    st.markdown("---")
    col_seg_churn, col_vf_churn = st.columns(2)

    with col_seg_churn:
        seg_churn = fdf.groupby(["User_Segment", "Churn_Status"])["CTR"].median().reset_index()
        fig_sc = px.bar(
            seg_churn, x="User_Segment", y="CTR", color="Churn_Status",
            barmode="group",
            color_discrete_map=color_map,
            text=seg_churn["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Median CTR", "User_Segment": "Segment", "Churn_Status": "Status"},
        )
        fig_sc.update_traces(textposition="outside", textfont_size=10)
        fig_sc.update_layout(
            title="Median CTR: Active vs Churned by Segment",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(tickangle=-15, gridcolor="#2A2A2A", linecolor="#333"),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_vf_churn:
        vf_churn = fdf.groupby(["Visual_Focus", "Churn_Status"])["CTR"].median().reset_index()
        fig_vc = px.bar(
            vf_churn, x="Visual_Focus", y="CTR", color="Churn_Status",
            barmode="group",
            color_discrete_map=color_map,
            text=vf_churn["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Median CTR", "Visual_Focus": "Visual Focus", "Churn_Status": "Status"},
        )
        fig_vc.update_traces(textposition="outside", textfont_size=10)
        fig_vc.update_layout(
            title="Median CTR: Active vs Churned by Visual Focus",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(tickangle=-15, gridcolor="#2A2A2A", linecolor="#333"),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vc, use_container_width=True)

    a_med = fdf[fdf["Churn_Status"]=="Active"]["CTR"].median() if "Active" in fdf["Churn_Status"].values else 0
    c_med = fdf[fdf["Churn_Status"]=="Churned"]["CTR"].median() if "Churned" in fdf["Churn_Status"].values else 0
    ratio = a_med / c_med if c_med > 0 else 0
    st.markdown(f"""<div class="rec-pill">
        💡 Active subscribers click at <b>{a_med:.1%}</b> median CTR vs <b>{c_med:.1%}</b> for churned users — a
        <b>{ratio:.1f}x gap</b>. Any cohort falling below <b>{c_med:.1%}</b> should trigger an artwork rotation
        review within 7 days, before cancellation occurs.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Genre Match
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="finding-badge">FINDING 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Genre Matching Fails — Visual Focus Is the Real Signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">GENRE-MATCHED IMPRESSIONS PRODUCE LOWER CTR THAN MISMATCHED — VISUAL FOCUS EXPLAINS THE GAP</div>', unsafe_allow_html=True)

    col_box, col_bar3 = st.columns([2, 3])

    with col_box:
        fig_match_box = go.Figure()
        for label, color in [("Matched", "#555555"), ("Mismatched", RED)]:
            subset = fdf[fdf["Match_Label"] == label]
            if len(subset):
                fig_match_box.add_trace(go.Box(
                    y=subset["CTR"] * 100,
                    name=label,
                    marker_color=color,
                    boxmean=True,
                    line=dict(color=color),
                    fillcolor=color + "44",
                    hovertemplate=f"<b>{label}</b><br>CTR: %{{y:.1f}}%<extra></extra>",
                ))
        fig_match_box.update_layout(
            title="CTR Distribution: Genre-Matched vs Mismatched",
            **PLOTLY_LAYOUT,
            yaxis=dict(title="CTR (%)", ticksuffix="%", gridcolor="#2A2A2A", linecolor="#333"),
            height=380,
        )
        st.plotly_chart(fig_match_box, use_container_width=True)

        match_med   = fdf[fdf["Match_Label"]=="Matched"]["CTR"].median()
        mismatch_med = fdf[fdf["Match_Label"]=="Mismatched"]["CTR"].median()
        st.markdown(f"""<div class="rec-pill">
            📌 Matched median: <b>{match_med:.1%}</b> vs Mismatched: <b>{mismatch_med:.1%}</b> —
            genre alone is insufficient. Visual focus is the precise lever.
        </div>""", unsafe_allow_html=True)

    with col_bar3:
        # Visual Focus CTR within matched vs mismatched
        vf_match = fdf.groupby(["Visual_Focus", "Match_Label"])["CTR"].mean().reset_index()
        fig_vfm = px.bar(
            vf_match, x="Visual_Focus", y="CTR", color="Match_Label",
            barmode="group",
            color_discrete_map={"Matched": "#555555", "Mismatched": RED},
            text=vf_match["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Avg CTR", "Visual_Focus": "Visual Focus", "Match_Label": "Genre Match"},
        )
        fig_vfm.update_traces(textposition="outside", textfont_size=10)
        fig_vfm.update_layout(
            title="CTR by Visual Focus — Matched vs Mismatched Impressions",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(tickangle=-15, gridcolor="#2A2A2A", linecolor="#333"),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vfm, use_container_width=True)

    st.markdown("---")
    # Segment x Match x Visual Focus — the critical cross-tab
    st.markdown("**Segment-level breakdown: which visual focus wins inside genre-matched impressions?**")
    sel_seg_match = st.selectbox(
        "Select segment",
        options=sorted(fdf["User_Segment"].unique()),
        key="seg_match_tab3",
    )
    seg_match_df = fdf[fdf["User_Segment"] == sel_seg_match]
    svf = seg_match_df.groupby(["Visual_Focus", "Match_Label"])["CTR"].mean().reset_index()
    fig_svf = px.bar(
        svf, x="Visual_Focus", y="CTR", color="Match_Label",
        barmode="group",
        color_discrete_map={"Matched": "#555555", "Mismatched": RED},
        text=svf["CTR"].apply(lambda v: f"{v:.1%}"),
        labels={"CTR": "Avg CTR", "Visual_Focus": "Visual Focus", "Match_Label": "Genre Match"},
    )
    fig_svf.update_traces(textposition="outside", textfont_size=11)
    fig_svf.update_layout(
        title=f"{sel_seg_match}: CTR by Visual Focus (Matched vs Mismatched Genre)",
        **PLOTLY_LAYOUT,
        yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
        xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_svf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Sci-Fi Deep Dive
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="finding-badge">SCI-FI DEEP DIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Storytelling Framework — Sci-Fi Enthusiasts</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">EXPLOSION/ACTION OUTPERFORMS COUPLE/INTIMACY BY 2.6X — RETIRE COUPLE/INTIMACY IMMEDIATELY</div>', unsafe_allow_html=True)

    scifi_df = fdf[fdf["User_Segment"] == "Sci-Fi Enthusiast"]

    if len(scifi_df) == 0:
        st.info("No Sci-Fi Enthusiast data in current filter selection.")
    else:
        col_rank, col_scatter_sf = st.columns([2, 3])

        with col_rank:
            sf_vf = scifi_df.groupby("Visual_Focus")["CTR"].mean().reset_index().sort_values("CTR", ascending=True)
            sf_vf["Rank"] = range(1, len(sf_vf) + 1)
            bar_colors = [RED if i == len(sf_vf)-1 else ("#FF6B35" if i == len(sf_vf)-2 else "#444")
                          for i in range(len(sf_vf))]

            fig_sf_rank = go.Figure(go.Bar(
                x=sf_vf["CTR"] * 100,
                y=sf_vf["Visual_Focus"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in sf_vf["CTR"] * 100],
                textposition="outside",
                textfont=dict(color=WHITE, size=13),
                hovertemplate="<b>%{y}</b><br>Avg CTR: %{text}<extra></extra>",
            ))
            fig_sf_rank.update_layout(
                title="Sci-Fi Enthusiast: Visual Focus Ranking",
                **PLOTLY_LAYOUT,
                xaxis=dict(title="Avg CTR (%)", ticksuffix="%", gridcolor="#2A2A2A", linecolor="#333"),
                yaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig_sf_rank, use_container_width=True)

        with col_scatter_sf:
            fig_sf_scatter = px.scatter(
                scifi_df, x="CTR", y="Time_Spent_on_Service",
                color="Visual_Focus",
                color_discrete_map=VISUAL_FOCUS_COLORS,
                symbol="Churn_Status",
                opacity=0.75,
                size_max=8,
                labels={"CTR": "CTR", "Time_Spent_on_Service": "Time on Platform (min)", "Visual_Focus": "Visual Focus"},
                hover_data={"Title_Genre": True, "Artwork_Variant": True},
            )
            fig_sf_scatter.update_layout(
                title="Sci-Fi Enthusiast: CTR vs Time Spent by Visual Focus",
                **PLOTLY_LAYOUT,
                xaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
                yaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
                height=340,
                legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sf_scatter, use_container_width=True)

        st.markdown("---")
        col_genre_sf, col_variant_sf = st.columns(2)

        with col_genre_sf:
            # Sci-Fi enthusiast CTR by title genre
            sf_genre = scifi_df.groupby(["Title_Genre", "Visual_Focus"])["CTR"].mean().reset_index()
            fig_sfg = px.bar(
                sf_genre, x="Title_Genre", y="CTR", color="Visual_Focus",
                barmode="group",
                color_discrete_map=VISUAL_FOCUS_COLORS,
                text=sf_genre["CTR"].apply(lambda v: f"{v:.1%}"),
                labels={"CTR": "Avg CTR", "Title_Genre": "Title Genre", "Visual_Focus": "Visual Focus"},
            )
            fig_sfg.update_traces(textposition="outside", textfont_size=9)
            fig_sfg.update_layout(
                title="Sci-Fi Enthusiast: CTR by Genre × Visual Focus",
                **PLOTLY_LAYOUT,
                yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
                xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
                height=340,
                legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sfg, use_container_width=True)

        with col_variant_sf:
            sf_variant = scifi_df.groupby(["Artwork_Variant", "Visual_Focus"])["CTR"].mean().reset_index()
            fig_sfv = px.bar(
                sf_variant, x="Artwork_Variant", y="CTR", color="Visual_Focus",
                barmode="group",
                color_discrete_map=VISUAL_FOCUS_COLORS,
                text=sf_variant["CTR"].apply(lambda v: f"{v:.1%}"),
                labels={"CTR": "Avg CTR", "Artwork_Variant": "Artwork Variant", "Visual_Focus": "Visual Focus"},
            )
            fig_sfv.update_traces(textposition="outside", textfont_size=9)
            fig_sfv.update_layout(
                title="Sci-Fi Enthusiast: CTR by Variant × Visual Focus",
                **PLOTLY_LAYOUT,
                yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
                xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
                height=340,
                legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sfv, use_container_width=True)

        best_sf = scifi_df.groupby("Visual_Focus")["CTR"].mean().idxmax()
        worst_sf = scifi_df.groupby("Visual_Focus")["CTR"].mean().idxmin()
        best_ctr = scifi_df.groupby("Visual_Focus")["CTR"].mean().max()
        worst_ctr = scifi_df.groupby("Visual_Focus")["CTR"].mean().min()
        st.markdown(f"""<div class="rec-pill">
            🎯 <b>Recommendation:</b> For Sci-Fi Enthusiasts, prioritize <b>{best_sf}</b> ({best_ctr:.1%} CTR).
            Retire <b>{worst_sf}</b> ({worst_ctr:.1%} CTR) from Sci-Fi rotation immediately —
            a {best_ctr/worst_ctr:.1f}x CTR gap with no infrastructure cost to fix.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Variant Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="finding-badge">VARIANT ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Artwork Variants A–D: The Null Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">VARIANTS DIFFER BY LESS THAN 0.5% CTR — THE A/B FRAMEWORK IS MEASURING THE WRONG VARIABLE</div>', unsafe_allow_html=True)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        var_ctr = fdf.groupby("Artwork_Variant")["CTR"].mean().reset_index().sort_values("CTR", ascending=False)
        overall_mean = var_ctr["CTR"].mean()
        fig_var = go.Figure(go.Bar(
            x=var_ctr["Artwork_Variant"],
            y=var_ctr["CTR"] * 100,
            marker_color=[RED if v == var_ctr["CTR"].max() else "#444" for v in var_ctr["CTR"]],
            text=[f"{v:.2f}%" for v in var_ctr["CTR"] * 100],
            textposition="outside",
            textfont=dict(color=WHITE, size=13),
        ))
        fig_var.add_hline(y=overall_mean * 100, line_dash="dash", line_color="#888",
                          annotation_text=f"Overall avg: {overall_mean:.2f}%",
                          annotation_font_color="#888")
        fig_var.update_layout(
            title="Average CTR by Artwork Variant",
            **PLOTLY_LAYOUT,
            yaxis=dict(title="Avg CTR (%)", ticksuffix="%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
            showlegend=False,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with col_v2:
        # Variant x Visual Focus
        vv = fdf.groupby(["Artwork_Variant", "Visual_Focus"])["CTR"].mean().reset_index()
        fig_vv = px.bar(
            vv, x="Artwork_Variant", y="CTR", color="Visual_Focus",
            barmode="group",
            color_discrete_map=VISUAL_FOCUS_COLORS,
            text=vv["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Avg CTR", "Artwork_Variant": "Variant", "Visual_Focus": "Visual Focus"},
        )
        fig_vv.update_traces(textposition="outside", textfont_size=9)
        fig_vv.update_layout(
            title="CTR by Variant × Visual Focus — Visual Focus Dominates",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
            legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vv, use_container_width=True)

    st.markdown("---")
    col_v3, col_v4 = st.columns(2)

    with col_v3:
        # Variant CTR by segment
        vs = fdf.groupby(["Artwork_Variant", "User_Segment"])["CTR"].mean().reset_index()
        fig_vs = px.bar(
            vs, x="Artwork_Variant", y="CTR", color="User_Segment",
            barmode="group",
            color_discrete_map=SEGMENT_COLORS,
            text=vs["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Avg CTR", "Artwork_Variant": "Variant", "User_Segment": "Segment"},
        )
        fig_vs.update_traces(textposition="outside", textfont_size=9)
        fig_vs.update_layout(
            title="CTR by Variant × Segment",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
            legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vs, use_container_width=True)

    with col_v4:
        # Variant CTR by churn
        vc = fdf.groupby(["Artwork_Variant", "Churn_Status"])["CTR"].median().reset_index()
        fig_vc2 = px.bar(
            vc, x="Artwork_Variant", y="CTR", color="Churn_Status",
            barmode="group",
            color_discrete_map={"Active": RED, "Churned": "#555555"},
            text=vc["CTR"].apply(lambda v: f"{v:.1%}"),
            labels={"CTR": "Median CTR", "Artwork_Variant": "Variant", "Churn_Status": "Status"},
        )
        fig_vc2.update_traces(textposition="outside", textfont_size=10)
        fig_vc2.update_layout(
            title="Median CTR by Variant × Retention Status",
            **PLOTLY_LAYOUT,
            yaxis=dict(tickformat=".0%", gridcolor="#2A2A2A", linecolor="#333"),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#333"),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vc2, use_container_width=True)

    var_range = (var_ctr["CTR"].max() - var_ctr["CTR"].min()) * 100
    st.markdown(f"""<div class="rec-pill">
        💡 Artwork variants A–D differ by only <b>{var_range:.2f} percentage points</b> in CTR —
        statistically negligible. The A/B testing framework should be reoriented to measure
        <b>visual focus performance by segment</b>, not variant performance in aggregate.
    </div>""", unsafe_allow_html=True)

# ── Raw Data expander ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔍  View Underlying Data", expanded=False):
    st.markdown('<div class="section-sub" style="padding:0">FILTERED DATASET — ALL VALUES REFLECT CURRENT SIDEBAR SELECTIONS</div>', unsafe_allow_html=True)
    display_df = fdf.copy()
    display_df["CTR"] = display_df["CTR"].apply(lambda v: f"{v:.2%}")
    st.dataframe(
        display_df.drop(columns=["Genre_Match", "Match_Label"]),
        use_container_width=True,
        height=350,
    )
    st.markdown(f"**{len(fdf):,}** rows shown · **{len(df):,}** total records")
    csv_bytes = fdf.drop(columns=["Genre_Match","Match_Label"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=csv_bytes,
        file_name="netflix_filtered_data.csv",
        mime="text/csv",
    )