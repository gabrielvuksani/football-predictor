"""
Footy Predictor — Responsive Streamlit UI  (v2)

Mobile-first responsive design that works on any device.

Tabs:
  1. Upcoming — match cards with predictions + BTTS/O2.5 badges
  2. Match Detail — full-page detail with odds comparison, H2H, form, AI
  3. AI Insights — value bets, BTTS/O2.5, accumulators, form tables, accuracy
  4. Training — retraining status, drift check, model history
  5. Database — summary stats
"""

from __future__ import annotations

import json
from datetime import datetime
from math import exp

import pandas as pd
import streamlit as st

from footy.db import connect
from footy.config import settings
from footy.normalize import canonical_team_name

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Footy Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Responsive CSS
# ---------------------------------------------------------------------------
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    /* ---- base ---- */
    .stApp { max-width: 100vw; overflow-x: hidden; }
    [data-testid="stMetricValue"] { font-size: clamp(18px, 4vw, 28px); font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: clamp(11px, 2.5vw, 14px); }

    /* ---- match card ---- */
    .match-card {
        padding: 12px 16px; border-radius: 10px;
        border-left: 4px solid #1f77b4;
        background: linear-gradient(135deg, #f8f9fa, #fff);
        margin-bottom: 8px;
    }
    .prob-badge {
        display: inline-block; padding: 6px 10px;
        border-radius: 6px; font-weight: 600; text-align: center;
        min-width: 52px; font-size: clamp(13px, 3vw, 16px);
    }
    .prob-home { background: #e3f2fd; color: #1565c0; }
    .prob-draw { background: #f3e5f5; color: #7b1fa2; }
    .prob-away { background: #fce4ec; color: #c62828; }
    .badge-btts { background: #e8f5e9; color: #2e7d32; padding: 3px 8px;
                  border-radius: 4px; font-weight: 600; font-size: 12px; }
    .badge-o25  { background: #fff3e0; color: #e65100; padding: 3px 8px;
                  border-radius: 4px; font-weight: 600; font-size: 12px; }
    .edge-badge { background: #c8e6c9; color: #2e7d32; padding: 3px 8px;
                  border-radius: 4px; font-weight: 600; }

    /* ---- odds table ---- */
    .odds-grid { width: 100%; border-collapse: collapse; font-size: 14px; }
    .odds-grid th { text-align: center; padding: 6px 4px; background: #f5f5f5;
                    border-bottom: 2px solid #ddd; font-weight: 600; }
    .odds-grid td { text-align: center; padding: 5px 4px; border-bottom: 1px solid #eee; }
    .odds-grid tr:hover { background: #f9f9f9; }
    .odds-grid .row-label { text-align: left; font-weight: 500; }
    .odds-best { font-weight: 700; color: #2e7d32; }

    /* ---- acca card ---- */
    .acca-card { padding: 10px 14px; border-radius: 8px; border: 1px solid #ddd;
                 margin-bottom: 10px; background: #fafafa; }
    .acca-header { font-weight: 700; font-size: 16px; margin-bottom: 6px; }
    .acca-odds { font-weight: 700; color: #1565c0; font-size: 18px; }

    /* ---- mobile ---- */
    @media (max-width: 640px) {
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding: 0.5rem 0.75rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
@st.cache_resource
def db():
    return connect()

con = db()
s = settings()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_pct(x):
    return "—" if x is None else f"{float(x):.0%}"

def fmt_pct1(x):
    return "—" if x is None else f"{float(x):.1%}"

def fmt_f(x, nd=2):
    return "—" if x is None else f"{float(x):.{nd}f}"

def to_float(x):
    try:
        if x is None: return None
        v = float(x)
        return v if pd.notna(v) else None
    except Exception:
        return None

def implied_1x2(h, d, a):
    h, d, a = to_float(h), to_float(d), to_float(a)
    if not h or not d or not a: return None
    ih, id_, ia = 1/h, 1/d, 1/a
    s = ih + id_ + ia
    return (ih/s, id_/s, ia/s, s - 1)

def est_btts(egh, ega):
    """Estimate BTTS probability from expected goals."""
    try:
        lh, la = float(egh), float(ega)
        return (1 - exp(-lh)) * (1 - exp(-la))
    except Exception:
        return None

def est_o25(egh, ega):
    """Estimate Over 2.5 probability from expected goals."""
    try:
        t = float(egh) + float(ega)
        return 1 - exp(-t) * (1 + t + t*t/2)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_form(team):
    try:
        from footy.llm.insights import analyze_team_form
        return analyze_team_form(team)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_news(team):
    try:
        from footy.llm.insights import extract_team_news_signal
        return extract_team_news_signal(team)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_explanation(mid, h, a, ph, pd_, pa, mv):
    try:
        from footy.llm.insights import explain_match
        return explain_match(mid, h, a, ph, pd_, pa, mv)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    try:
        mv_df = con.execute("SELECT DISTINCT model_version FROM predictions ORDER BY model_version").df()
        versions = mv_df["model_version"].tolist() if not mv_df.empty else ["v8_council"]
    except Exception:
        versions = ["v8_council"]
    default = "v8_council" if "v8_council" in versions else versions[-1]
    model_version = st.selectbox("Model", versions, index=versions.index(default) if default in versions else 0)
    lookahead = st.slider("Lookahead (days)", 1, 21, 7)
    st.caption("DB: footy.duckdb")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "selected_match_id" not in st.session_state:
    st.session_state.selected_match_id = None

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown("# ⚽ Footy Predictor")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upcoming, tab_detail, tab_ai, tab_training, tab_db = st.tabs(
    ["📋 Upcoming", "🔍 Detail", "🤖 AI Insights", "🔄 Training", "💾 Database"]
)


# ---------------------------------------------------------------------------
# Shared: Odds comparison table
# ---------------------------------------------------------------------------
def render_odds_comparison(mid: int):
    """Show a multi-bookmaker odds comparison grid for a match."""
    try:
        row = con.execute("""
            SELECT b365h, b365d, b365a,
                   b365ch, b365cd, b365ca,
                   psh, psd, psa,
                   avgh, avgd, avga,
                   maxh, maxd, maxa,
                   b365_o25, b365_u25,
                   avg_o25, avg_u25,
                   max_o25, max_u25
            FROM match_extras WHERE match_id = ?
        """, [mid]).fetchone()
    except Exception:
        row = None

    if not row:
        st.info("No odds data available")
        return

    (b365h, b365d, b365a,
     b365ch, b365cd, b365ca,
     psh, psd, psa,
     avgh, avgd, avga,
     maxh, maxd, maxa,
     b365_o25, b365_u25,
     avg_o25, avg_u25,
     max_o25, max_u25) = row

    # --- 1X2 odds grid ---
    st.markdown("**1X2 Match Result**")
    sources = [
        ("Bet365 Open", b365h, b365d, b365a),
        ("Bet365 Close", b365ch, b365cd, b365ca),
        ("Pinnacle", psh, psd, psa),
        ("Market Avg", avgh, avgd, avga),
        ("Market Max", maxh, maxd, maxa),
    ]
    # Find best odds per outcome
    all_h = [x[1] for x in sources if x[1]]
    all_d = [x[2] for x in sources if x[2]]
    all_a = [x[3] for x in sources if x[3]]
    best_h = max(all_h) if all_h else None
    best_d = max(all_d) if all_d else None
    best_a = max(all_a) if all_a else None

    html = '<table class="odds-grid"><tr><th></th><th>Home</th><th>Draw</th><th>Away</th><th>Overround</th></tr>'
    for label, oh, od, oa in sources:
        if not oh and not od and not oa:
            continue
        imp = implied_1x2(oh, od, oa)
        ovr = f"{imp[3]:.1%}" if imp else "—"

        def _cell(val, best):
            if val is None:
                return "<td>—</td>"
            cls = " class='odds-best'" if best and abs(float(val) - float(best)) < 0.001 else ""
            return f"<td{cls}>{float(val):.2f}</td>"

        html += f"<tr><td class='row-label'>{label}</td>{_cell(oh, best_h)}{_cell(od, best_d)}{_cell(oa, best_a)}<td>{ovr}</td></tr>"

    # Implied probabilities row (from best available)
    ref_h, ref_d, ref_a = avgh or b365h, avgd or b365d, avga or b365a
    imp = implied_1x2(ref_h, ref_d, ref_a)
    if imp:
        html += (
            f"<tr style='background:#f0f0f0; font-style:italic'>"
            f"<td class='row-label'>Implied %</td>"
            f"<td>{imp[0]:.0%}</td><td>{imp[1]:.0%}</td><td>{imp[2]:.0%}</td>"
            f"<td></td></tr>"
        )
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

    # --- Over/Under 2.5 ---
    ou_sources = [
        ("Bet365", b365_o25, b365_u25),
        ("Market Avg", avg_o25, avg_u25),
        ("Market Max", max_o25, max_u25),
    ]
    has_ou = any(s[1] or s[2] for s in ou_sources)
    if has_ou:
        st.markdown("")
        st.markdown("**Over / Under 2.5 Goals**")
        ou_html = '<table class="odds-grid"><tr><th></th><th>Over 2.5</th><th>Under 2.5</th></tr>'
        for label, ov, un in ou_sources:
            if not ov and not un:
                continue
            ou_html += f"<tr><td class='row-label'>{label}</td><td>{fmt_f(ov)}</td><td>{fmt_f(un)}</td></tr>"
        ou_html += "</table>"
        st.markdown(ou_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared detail renderer
# ---------------------------------------------------------------------------
def render_match_detail(mid: int, home_raw: str, away_raw: str,
                        competition: str, utc_date, key_prefix: str = "d"):
    """Full detail view for a single match."""
    home = canonical_team_name(home_raw)
    away = canonical_team_name(away_raw)

    st.markdown(f"## {home_raw} vs {away_raw}")
    st.caption(f"{competition} · {utc_date}")

    # Predictions
    pred = con.execute(
        "SELECT p_home, p_draw, p_away, eg_home, eg_away, notes "
        "FROM predictions WHERE match_id=? AND model_version=?",
        [mid, model_version],
    ).fetchone()
    p_h = p_d = p_a = eg_h = eg_a = None
    notes = {}
    if pred:
        p_h, p_d, p_a, eg_h, eg_a, notes_str = pred
        if notes_str:
            try:
                notes = json.loads(notes_str)
            except Exception:
                pass

    # Top metrics row
    cols = st.columns(7)
    cols[0].metric("Home", fmt_pct1(p_h))
    cols[1].metric("Draw", fmt_pct1(p_d))
    cols[2].metric("Away", fmt_pct1(p_a))
    cols[3].metric("xG H", fmt_f(eg_h, 1))
    cols[4].metric("xG A", fmt_f(eg_a, 1))

    # BTTS / O2.5 from notes or estimated
    btts_p = notes.get("btts") or est_btts(eg_h, eg_a)
    o25_p = notes.get("o25") or est_o25(eg_h, eg_a)
    cols[5].metric("BTTS", fmt_pct(btts_p))
    cols[6].metric("O2.5", fmt_pct(o25_p))

    # Predicted score
    ps = notes.get("predicted_score")
    if ps:
        if isinstance(ps, list) and len(ps) == 2:
            st.caption(f"📊 Predicted score: {ps[0]}-{ps[1]}")
        elif isinstance(ps, str):
            st.caption(f"📊 Predicted score: {ps}")

    # Odds comparison
    with st.expander("💰 Market Odds Comparison", expanded=True):
        render_odds_comparison(mid)

    # H2H
    with st.expander("⚔️ Head-to-Head", expanded=False):
        try:
            from footy.h2h import get_h2h_any_venue
            h2h = get_h2h_any_venue(con, home, away, limit=10)
            stats = h2h.get("stats", {})
            if stats and stats.get("total_matches", 0) > 0:
                h1, h2_, h3, h4 = st.columns(4)
                h1.metric("Games", stats["total_matches"])
                h2_.metric(f"{home_raw} W", stats.get("team_a_wins", 0))
                h3.metric("Draws", stats.get("draws", 0))
                h4.metric(f"{away_raw} W", stats.get("team_b_wins", 0))

                # Show recent H2H results
                recent = h2h.get("matches", [])
                if recent:
                    st.markdown("**Recent meetings:**")
                    for m in recent[:5]:
                        ht = m.get("home_team", "?")
                        at_ = m.get("away_team", "?")
                        hg = m.get("home_goals", "?")
                        ag_ = m.get("away_goals", "?")
                        dt = str(m.get("utc_date", ""))[:10]
                        st.caption(f"{dt} — {ht} {hg}-{ag_} {at_}")
            else:
                st.info("No H2H history")
        except Exception:
            st.info("H2H module unavailable")

    # Elo ratings
    with st.expander("📊 Elo Ratings", expanded=False):
        try:
            elo_h = con.execute("SELECT rating FROM elo_state WHERE team=?", [home]).fetchone()
            elo_a = con.execute("SELECT rating FROM elo_state WHERE team=?", [away]).fetchone()
            e1, e2, e3 = st.columns(3)
            e1.metric(f"{home_raw}", fmt_f(elo_h[0] if elo_h else None, 0))
            e2.metric(f"{away_raw}", fmt_f(elo_a[0] if elo_a else None, 0))
            if elo_h and elo_a:
                diff = elo_h[0] - elo_a[0]
                e3.metric("Elo diff", f"{diff:+.0f}")
        except Exception:
            st.info("Elo data unavailable")

    # Form
    st.markdown("---")
    if st.button("📊 Load Form Analysis", use_container_width=True,
                 key=f"{key_prefix}_form_{mid}"):
        with st.spinner("Analyzing form..."):
            fc1, fc2 = st.columns(2)
            for col, team_name, label in [(fc1, home, home_raw), (fc2, away, away_raw)]:
                with col:
                    st.markdown(f"**{label}**")
                    form = load_form(team_name)
                    if form:
                        st.metric("Form", form.get("recent_form", "?"))
                        st.metric("Record", form.get("record", "?"))
                        st.metric("Momentum", f"{form.get('momentum', 0):.1f}")
                        if form.get("key_trends"):
                            st.caption("Trends: " + " · ".join(form["key_trends"]))
                        if form.get("concern_areas"):
                            st.caption("Concerns: " + " · ".join(form["concern_areas"]))
                    else:
                        st.info("No form data")

    # News
    if st.button("📰 Load Team News", use_container_width=True,
                 key=f"{key_prefix}_news_{mid}"):
        with st.spinner("Fetching news signals..."):
            nc1, nc2 = st.columns(2)
            for col, team_name, label in [(nc1, home, home_raw), (nc2, away, away_raw)]:
                with col:
                    st.markdown(f"**{label}**")
                    news = load_news(team_name)
                    if news:
                        score = news.get("availability_score", 0)
                        emoji = "🟢" if score > 0.3 else "🟡" if score > -0.3 else "🔴"
                        st.metric("Availability", f"{emoji} {score:.1f}")
                        if news.get("likely_absences"):
                            st.warning("Absences: " + ", ".join(news["likely_absences"]))
                        if news.get("key_notes"):
                            for n in news["key_notes"][:3]:
                                st.caption(f"• {n}")
                    else:
                        st.info("No news available")

    # AI explanation
    if st.button("🧠 Generate AI Explanation", use_container_width=True,
                 key=f"{key_prefix}_explain_{mid}"):
        with st.spinner("Generating explanation..."):
            expl = load_explanation(
                mid, home, away,
                float(p_h) if p_h else 0.33,
                float(p_d) if p_d else 0.33,
                float(p_a) if p_a else 0.33,
                model_version,
            )
            if expl:
                st.markdown(expl.get("explanation", "No explanation available"))
                if expl.get("key_factors"):
                    st.markdown("**Key factors:**")
                    for f in expl["key_factors"]:
                        st.markdown(f"- {f}")
                conf = expl.get("confidence_level", "Medium")
                conf_emoji = {"Very High": "🟢", "High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🟡")
                st.metric("Confidence", f"{conf_emoji} {conf}")
            else:
                st.info("AI explanation unavailable (is Ollama running?)")


# ============================================================================
# TAB 1: UPCOMING
# ============================================================================
with tab_upcoming:
    if st.session_state.selected_match_id is not None:
        if st.button("← Back to matches", type="secondary"):
            st.session_state.selected_match_id = None
            st.rerun()
        row = con.execute(
            "SELECT match_id, utc_date, competition, home_team, away_team "
            "FROM matches WHERE match_id=?",
            [st.session_state.selected_match_id],
        ).fetchone()
        if row:
            render_match_detail(row[0], row[3], row[4], row[2], row[1], key_prefix="up")
        else:
            st.error("Match not found")
            st.session_state.selected_match_id = None
    else:
        df = con.execute(f"""
            SELECT m.utc_date, m.competition, m.home_team, m.away_team,
                   p.p_home, p.p_draw, p.p_away, p.eg_home, p.eg_away,
                   p.notes, m.match_id
            FROM matches m
            LEFT JOIN predictions p ON p.match_id = m.match_id AND p.model_version = ?
            WHERE m.status IN ('SCHEDULED','TIMED')
              AND m.utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead)} DAY)
            ORDER BY m.utc_date
        """, [model_version]).df()

        if df.empty:
            st.info("No upcoming matches. Run `footy go` to ingest & predict.")
        else:
            st.caption(f"{len(df)} matches · {model_version}")
            df["date_str"] = df["utc_date"].astype(str).str[:10]
            for date_str, group in df.groupby("date_str"):
                st.markdown(f"#### {date_str}")
                for _, r in group.iterrows():
                    c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 1, 2])
                    with c1:
                        if st.button(
                            f"{r['home_team']}  vs  {r['away_team']}",
                            key=f"match_{r['match_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_match_id = int(r["match_id"])
                            st.rerun()
                        st.caption(r["competition"])
                    with c2:
                        st.markdown(
                            f"<div class='prob-badge prob-home'>{fmt_pct(r['p_home'])}</div>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"<div class='prob-badge prob-draw'>{fmt_pct(r['p_draw'])}</div>",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        st.markdown(
                            f"<div class='prob-badge prob-away'>{fmt_pct(r['p_away'])}</div>",
                            unsafe_allow_html=True,
                        )
                    with c5:
                        # BTTS & O2.5 mini badges
                        btts = None
                        o25 = None
                        notes_str = r.get("notes")
                        if notes_str and isinstance(notes_str, str):
                            try:
                                nt = json.loads(notes_str)
                                btts = nt.get("btts")
                                o25 = nt.get("o25")
                            except Exception:
                                pass
                        if btts is None:
                            btts = est_btts(r.get("eg_home"), r.get("eg_away"))
                        if o25 is None:
                            o25 = est_o25(r.get("eg_home"), r.get("eg_away"))

                        badges = []
                        if btts is not None:
                            badges.append(f"<span class='badge-btts'>BTTS {float(btts):.0%}</span>")
                        if o25 is not None:
                            badges.append(f"<span class='badge-o25'>O2.5 {float(o25):.0%}</span>")
                        if badges:
                            st.markdown(" ".join(badges), unsafe_allow_html=True)
                st.divider()

# ============================================================================
# TAB 2: MATCH DETAIL
# ============================================================================
with tab_detail:
    mids = con.execute(f"""
        SELECT match_id, utc_date, competition, home_team, away_team
        FROM matches
        WHERE status IN ('SCHEDULED','TIMED')
          AND utc_date <= (CURRENT_TIMESTAMP + INTERVAL {int(lookahead)} DAY)
        ORDER BY utc_date
    """).df()

    if mids.empty:
        st.info("No scheduled matches.")
    else:
        default_idx = 0
        if st.session_state.selected_match_id is not None:
            sel = mids.index[mids["match_id"] == st.session_state.selected_match_id]
            if len(sel) > 0:
                default_idx = int(sel[0])

        labels = mids.apply(
            lambda r: f"{str(r.utc_date)[:10]} · {r.competition} · {r.home_team} vs {r.away_team}",
            axis=1,
        )
        idx = st.selectbox("Match", range(len(mids)), index=default_idx,
                           format_func=lambda i: labels.iloc[i])
        row = mids.iloc[int(idx)]
        render_match_detail(int(row.match_id), row.home_team, row.away_team,
                            row.competition, row.utc_date, key_prefix="dt")

# ============================================================================
# TAB 3: AI INSIGHTS
# ============================================================================
with tab_ai:
    ai_sub = st.radio(
        "Section",
        ["Value Bets", "BTTS & O/U", "Accumulators", "League Form", "Accuracy", "Round Preview", "Post-Match Review"],
        horizontal=True,
    )

    # ---- Value Bets ----
    if ai_sub == "Value Bets":
        st.markdown("### 💰 Value Bet Scanner")
        edge_thresh = st.slider("Min edge %", 1, 20, 5, key="edge_sl") / 100
        try:
            from footy.llm.insights import value_bet_scan, ai_value_commentary
            values = value_bet_scan(min_edge=edge_thresh)
            if not values:
                st.info(f"No value bets with edge ≥ {edge_thresh:.0%}")
            else:
                for v in values[:15]:
                    vc1, vc2, vc3 = st.columns([3, 1, 1])
                    with vc1:
                        st.markdown(f"**{v['home_team']}** vs {v['away_team']} — {v['bet']}")
                        st.caption(v["date"])
                    with vc2:
                        st.metric("Odds", f"{v['odds']:.2f}")
                    with vc3:
                        st.markdown(f"<span class='edge-badge'>+{v['edge']:.1%}</span>", unsafe_allow_html=True)
                        st.caption(f"Model {v['model_prob']:.0%} vs Impl {v['implied_prob']:.0%}")
                    st.divider()

                commentary = ai_value_commentary(values)
                st.markdown(f"*{commentary}*")
        except Exception as e:
            st.error(f"Value scan error: {e}")

    # ---- BTTS & Over/Under ----
    elif ai_sub == "BTTS & O/U":
        st.markdown("### ⚽ Both Teams To Score & Over/Under 2.5")
        try:
            from footy.llm.insights import btts_ou_insights
            data = btts_ou_insights()

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("#### 🟢 BTTS Likely (>55%)")
                for item in data.get("btts_likely", []):
                    st.markdown(
                        f"**{item['home_team']}** vs **{item['away_team']}** "
                        f"<span class='badge-btts'>{item['btts_prob']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{item['competition']} · {item['date']} · "
                        f"xG {item['eg_home'] or '?'}-{item['eg_away'] or '?'}"
                    )
                if not data.get("btts_likely"):
                    st.info("No strong BTTS picks")

            with bc2:
                st.markdown("#### 🔴 BTTS Unlikely (<35%)")
                for item in data.get("btts_unlikely", []):
                    st.markdown(
                        f"**{item['home_team']}** vs **{item['away_team']}** "
                        f"<span class='badge-btts'>{item['btts_prob']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{item['competition']} · {item['date']} · "
                        f"xG {item['eg_home'] or '?'}-{item['eg_away'] or '?'}"
                    )
                if not data.get("btts_unlikely"):
                    st.info("No low-BTTS picks")

            st.divider()
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown("#### ⬆️ Over 2.5 Likely (>55%)")
                for item in data.get("over25", []):
                    st.markdown(
                        f"**{item['home_team']}** vs **{item['away_team']}** "
                        f"<span class='badge-o25'>{item['o25_prob']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{item['competition']} · {item['date']} · "
                        f"xG {item['eg_home'] or '?'}-{item['eg_away'] or '?'}"
                    )
                if not data.get("over25"):
                    st.info("No strong Over 2.5 picks")

            with oc2:
                st.markdown("#### ⬇️ Under 2.5 Likely (<35%)")
                for item in data.get("under25", []):
                    st.markdown(
                        f"**{item['home_team']}** vs **{item['away_team']}** "
                        f"<span class='badge-o25'>{item['o25_prob']:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{item['competition']} · {item['date']} · "
                        f"xG {item['eg_home'] or '?'}-{item['eg_away'] or '?'}"
                    )
                if not data.get("under25"):
                    st.info("No strong Under 2.5 picks")

        except Exception as e:
            st.error(f"BTTS/O2.5 error: {e}")

    # ---- Accumulators ----
    elif ai_sub == "Accumulators":
        st.markdown("### 🎰 Accumulator Builder")
        min_prob_acca = st.slider("Min single-pick prob", 40, 80, 55, key="acca_prob") / 100
        try:
            from footy.llm.insights import build_accumulators
            accas = build_accumulators(min_prob=min_prob_acca)
            if not accas:
                st.info("Not enough confident picks for accumulators.")
            else:
                for acca in accas:
                    st.markdown(f"<div class='acca-card'><div class='acca-header'>{acca['type']}</div>", unsafe_allow_html=True)
                    for leg in acca["legs"]:
                        lc1, lc2, lc3 = st.columns([3, 1, 1])
                        with lc1:
                            st.markdown(f"{leg['home_team']} vs {leg['away_team']}")
                            st.caption(f"{leg['competition']} · {leg['date']}")
                        with lc2:
                            st.caption(f"Pick: **{leg['pick']}**")
                        with lc3:
                            st.caption(f"{leg['prob']:.0%}" + (f" @ {leg['odds']}" if leg['odds'] else ""))

                    fc1, fc2 = st.columns(2)
                    fc1.metric("Combined Prob", f"{acca['combined_prob']:.1%}")
                    if acca.get("combined_odds"):
                        fc2.markdown(f"<span class='acca-odds'>Combined Odds: {acca['combined_odds']:.1f}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("")
        except Exception as e:
            st.error(f"Accumulator error: {e}")

    # ---- League Form ----
    elif ai_sub == "League Form":
        st.markdown("### 📊 League Form Table")
        league = st.selectbox("League", ["PL", "PD", "SA", "BL1", "FL1", "DED", "ELC"], key="form_league")
        last_n = st.slider("Last N matches", 3, 10, 6, key="form_n")
        try:
            from footy.llm.insights import league_form_table
            table = league_form_table(league, last_n=last_n)
            if not table:
                st.info("No data for this league")
            else:
                form_df = pd.DataFrame(table)
                form_df = form_df.rename(columns={
                    "team": "Team", "played": "P", "w": "W", "d": "D", "l": "L",
                    "ppg": "PPG", "gf": "GF", "ga": "GA", "gd": "GD",
                    "btts_pct": "BTTS%", "o25_pct": "O2.5%",
                })
                st.dataframe(
                    form_df[["Team", "P", "W", "D", "L", "PPG", "GF", "GA", "GD", "BTTS%", "O2.5%"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "PPG": st.column_config.NumberColumn(format="%.2f"),
                        "BTTS%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                        "O2.5%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                    },
                )
        except Exception as e:
            st.error(f"Form table error: {e}")

    # ---- Accuracy Dashboard ----
    elif ai_sub == "Accuracy":
        st.markdown("### 📈 Prediction Accuracy Dashboard")
        days = st.slider("Days back", 7, 90, 30, key="acc_days")
        try:
            from footy.llm.insights import prediction_accuracy_stats
            stats = prediction_accuracy_stats(days_back=days)
            if stats.get("total", 0) == 0:
                st.info("No finished predictions in this period.")
            else:
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric("Overall Accuracy", f"{stats['accuracy']:.1%}")
                ac2.metric("Correct", f"{stats['correct']}/{stats['total']}")
                ac3.metric("Brier Score", f"{stats['brier_score']:.4f}")

                # By confidence
                st.markdown("#### By Confidence Tier")
                conf = stats.get("by_confidence", {})
                cc1, cc2, cc3 = st.columns(3)
                for col, tier, emoji in [(cc1, "high", "🟢"), (cc2, "mid", "🟡"), (cc3, "low", "🔴")]:
                    d = conf.get(tier, {})
                    with col:
                        st.markdown(f"{emoji} **{tier.title()}** confidence")
                        if d.get("total", 0) > 0:
                            st.metric("Accuracy", f"{d['accuracy']:.1%}")
                            st.caption(f"{d['correct']}/{d['total']} matches")
                        else:
                            st.caption("No data")

                # By competition
                st.markdown("#### By Competition")
                comp = stats.get("by_competition", {})
                if comp:
                    comp_df = pd.DataFrame([
                        {"Competition": k, "Total": v["total"], "Correct": v["correct"],
                         "Accuracy": v["accuracy"]}
                        for k, v in sorted(comp.items())
                    ])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                                 column_config={"Accuracy": st.column_config.ProgressColumn(
                                     min_value=0, max_value=1, format="%.1%")})
        except Exception as e:
            st.error(f"Accuracy stats error: {e}")

    # ---- Round Preview ----
    elif ai_sub == "Round Preview":
        st.markdown("### 📋 League Round Preview")
        league = st.selectbox("League", ["PL", "PD", "SA", "BL1", "FL1", "DED", "ELC"], key="round_league")
        try:
            from footy.llm.insights import league_round_summary
            summary = league_round_summary(league)
            if summary.get("matches", 0) == 0:
                st.info("No upcoming matches with predictions in this league.")
            else:
                st.markdown(f"**{summary.get('matches', 0)} matches**")
                st.markdown(summary.get("summary", ""))
                if summary.get("headline_pick"):
                    st.success(f"🏟️ Headline: {summary['headline_pick']}")
                for p in summary.get("predictions", []):
                    pc1, pc2 = st.columns([3, 2])
                    with pc1:
                        st.markdown(f"{p['home']} vs {p['away']}")
                    with pc2:
                        st.caption(f"{p['pred']} ({p['probs']})")
        except Exception as e:
            st.error(f"Round preview error: {e}")

    # ---- Post-Match Review ----
    elif ai_sub == "Post-Match Review":
        st.markdown("### 📊 Post-Match Review")
        review_days = st.slider("Days back", 1, 14, 3, key="review_days")
        review_league = st.selectbox("League (optional)", ["All", "PL", "PD", "SA", "BL1", "FL1", "DED", "ELC"], key="review_league")
        try:
            from footy.llm.insights import post_match_review
            comp = None if review_league == "All" else review_league
            review = post_match_review(days_back=review_days, competition_code=comp)
            if review["matches_reviewed"] == 0:
                st.info("No recent finished matches with predictions.")
            else:
                acc = review["accuracy"]
                rc1, rc2 = st.columns(2)
                rc1.metric("Accuracy", f"{acc:.0%}")
                rc2.metric("Correct", f"{review['correct']}/{review['matches_reviewed']}")
                st.markdown(review["review"])
                if review.get("misses"):
                    st.markdown("**Notable misses:**")
                    for m in review["misses"]:
                        st.markdown(
                            f"- ✗ {m['match']} {m['score']} — predicted {m['predicted']} "
                            f"(was {m['actual']}, conf {m['confidence']:.0%})"
                        )
        except Exception as e:
            st.error(f"Review error: {e}")

# ============================================================================
# TAB 4: TRAINING
# ============================================================================
with tab_training:
    st.markdown("### Model Training & Drift")

    with st.expander("🔍 Drift Detection", expanded=True):
        try:
            from footy.continuous_training import get_training_manager
            mgr = get_training_manager()
            drift = mgr.detect_drift("v8_council")
            if drift.get("reason") == "insufficient_data":
                st.info(f"Not enough data (baseline: {drift.get('baseline_n',0)}, recent: {drift.get('recent_n',0)})")
            elif drift.get("drifted"):
                st.error(
                    f"**DRIFT DETECTED**  \n"
                    f"Baseline: {drift['baseline_acc']:.1%} ({drift['baseline_n']} matches)  \n"
                    f"Recent: {drift['recent_acc']:.1%} ({drift['recent_n']} matches)  \n"
                    f"Drop: {drift['accuracy_drop']:+.1%}"
                )
            else:
                st.success(
                    f"No drift.  \n"
                    f"Baseline: {drift.get('baseline_acc',0):.1%} · Recent: {drift.get('recent_acc',0):.1%}"
                )
        except Exception as e:
            st.warning(f"Drift check error: {e}")

    with st.expander("📈 Retraining Status", expanded=True):
        try:
            from footy.continuous_training import get_training_manager
            mgr = get_training_manager()
            status = mgr.get_retraining_status()
            if status:
                for mt, info in status.items():
                    ready_emoji = "🟢" if info["ready"] else "🟡"
                    st.markdown(
                        f"{ready_emoji} **{mt}** — {info['new_matches']}/{info['threshold']} matches  \n"
                        f"Last trained: {info.get('last_trained', 'Never')}"
                    )
            else:
                st.info("No models configured. Run `footy retraining-setup v8_council`")
        except Exception as e:
            st.warning(f"Status error: {e}")

    with st.expander("📜 Training History"):
        try:
            hist = con.execute("""
                SELECT model_version, training_date, n_matches_used, improvement_pct, deployed
                FROM model_training_records ORDER BY training_date DESC LIMIT 20
            """).df()
            if not hist.empty:
                st.dataframe(hist, use_container_width=True, hide_index=True)
            else:
                st.info("No training records")
        except Exception:
            st.info("Training records table not yet created")

    with st.expander("🚀 Deployments"):
        try:
            deps = con.execute("""
                SELECT model_type, active_version, deployed_at, previous_version
                FROM model_deployments ORDER BY deployed_at DESC
            """).df()
            if not deps.empty:
                st.dataframe(deps, use_container_width=True, hide_index=True)
            else:
                st.info("No deployments recorded")
        except Exception:
            st.info("Deployments table not yet created")

    # --- Self-improvement report ---
    with st.expander("🧠 Self-Improvement Report"):
        if st.button("Generate Report", use_container_width=True, key="gen_imp_report"):
            with st.spinner("Analyzing prediction errors..."):
                try:
                    from footy.performance_tracker import generate_improvement_report
                    report = generate_improvement_report()
                    st.code(report, language="text")
                except Exception as e:
                    st.error(f"Report error: {e}")

    st.markdown("---")
    if st.button("🔄 Run Auto-Retrain", use_container_width=True, type="primary"):
        with st.spinner("Training... this may take several minutes"):
            try:
                from footy.continuous_training import get_training_manager
                mgr = get_training_manager()
                try:
                    mgr.setup_continuous_training("v8_council", 20, 0.005)
                except Exception:
                    pass
                result = mgr.auto_retrain(force=True, verbose=True)
                action = result.get("action", "?")
                if action == "deployed":
                    st.success(f"Deployed {result['version']} (improvement: {result.get('improvement_pct',0):+.2f}%)")
                elif action == "rolled_back":
                    st.warning(f"Rolled back — regression {result.get('improvement_pct',0):+.2f}%")
                elif action == "train_failed":
                    st.error(f"Training failed: {result.get('error','?')}")
                else:
                    st.info(f"Result: {action}")
            except Exception as e:
                st.error(f"Auto-retrain error: {e}")

# ============================================================================
# TAB 5: DATABASE
# ============================================================================
with tab_db:
    st.markdown("### Database Summary")
    try:
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            total = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            finished = con.execute("SELECT COUNT(*) FROM matches WHERE status='FINISHED'").fetchone()[0]
            st.metric("Total Matches", total)
            st.metric("Finished", finished)
        with dc2:
            upcoming = con.execute("SELECT COUNT(*) FROM matches WHERE status IN ('TIMED','SCHEDULED')").fetchone()[0]
            teams = con.execute("SELECT COUNT(DISTINCT home_team) FROM matches").fetchone()[0]
            st.metric("Upcoming", upcoming)
            st.metric("Teams", teams)
        with dc3:
            preds = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            st.metric("Predictions", preds)
            try:
                models = con.execute("SELECT COUNT(DISTINCT model_version) FROM predictions").fetchone()[0]
                st.metric("Model Versions", models)
            except Exception:
                pass

        st.divider()
        st.markdown("#### By Competition")
        comp_df = con.execute("""
            SELECT competition, status, COUNT(*) n
            FROM matches GROUP BY 1, 2 ORDER BY 1, 2
        """).df()
        if not comp_df.empty:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Extras Coverage")
        try:
            xt = con.execute("""
                SELECT
                    COUNT(*) as total_extras,
                    SUM(CASE WHEN b365h IS NOT NULL THEN 1 ELSE 0 END) as has_b365,
                    SUM(CASE WHEN psh IS NOT NULL THEN 1 ELSE 0 END) as has_pinnacle,
                    SUM(CASE WHEN avgh IS NOT NULL THEN 1 ELSE 0 END) as has_avg,
                    SUM(CASE WHEN b365_o25 IS NOT NULL THEN 1 ELSE 0 END) as has_ou25,
                    SUM(CASE WHEN hthg IS NOT NULL THEN 1 ELSE 0 END) as has_ht
                FROM match_extras
            """).fetchone()
            if xt:
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Total Extras", xt[0])
                ec1.metric("Bet365", xt[1])
                ec2.metric("Pinnacle", xt[2])
                ec2.metric("Market Avg", xt[3])
                ec3.metric("O/U 2.5", xt[4])
                ec3.metric("Half-time", xt[5])
        except Exception:
            st.info("Extras table empty or not yet populated")

    except Exception as e:
        st.error(f"DB error: {e}")

# ---------------------------------------------------------------------------
st.divider()
st.caption("⚽ Footy Predictor · ML-powered football predictions · v8_council")
