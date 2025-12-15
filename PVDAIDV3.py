import time
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
import requests

# -----------------------------------------------------------
# PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
# -----------------------------------------------------------
st.set_page_config(
    page_title="PV + Battery + DA + ID Optimisation – FLEX",
    layout="wide"
)
# =========================
# PASSWORD PROTECTION
# =========================
def require_password():
    PASSWORD = st.secrets["APP_PASSWORD"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Login required")
        pwd = st.text_input("Enter password", type="password")

        if st.button("Login"):
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

        st.stop()

require_password()

# -----------------------------------------------------------
# FLEX THEME HELPERS
# -----------------------------------------------------------

FLEX_PRIMARY = "#ef4444"   # red X
FLEX_BG = "#020617"        # dark background


def inject_flex_theme():
    css = f"""
    <style>
    body {{
        background: radial-gradient(circle at top left, #111827, {FLEX_BG});
        color: #e5e7eb;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .sidebar .sidebar-content {{
        background: #020617;
    }}
    .stButton>button {{
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.5);
        padding: 0.4rem 1.1rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        border-color: {FLEX_PRIMARY};
        color: {FLEX_PRIMARY};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def apply_theme():
    theme = st.session_state.get("theme", "dark")
    if theme == "light":
        css = """
        <style>
        body { background: #f3f4f6; color: #111827; }
        .sidebar .sidebar-content { background: #e5e7eb; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        inject_flex_theme()


# -----------------------------------------------------------
# AUTH / LOGIN HELPERS
# -----------------------------------------------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str):
    """
    Returns (True, role) if username/password are valid, else (False, None).
    Users and hashes are stored in st.secrets["users"].
    """
    users = st.secrets.get("users", {})
    user_cfg = users.get(username)

    if not user_cfg:
        return False, None

    stored_hash = user_cfg.get("password_hash")
    role = user_cfg.get("role", "viewer")

    if hash_password(password) == stored_hash:
        return True, role
    return False, None


@st.cache_resource
def get_user_store():
    """
    Simple in-memory store shared across sessions on the same worker.
    NOTE: On Streamlit Cloud with multiple workers, this is best-effort only.
    """
    return {
        "logins": [],   # list of {user, role, time}
        "last_seen": {}  # {user: timestamp}
    }


def register_login(username: str, role: str):
    store = get_user_store()
    now = time.time()
    store["logins"].append({
        "user": username,
        "role": role,
        "time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    })
    store["last_seen"][username] = now


def update_last_seen(username: str):
    store = get_user_store()
    store["last_seen"][username] = time.time()


def get_active_users(max_idle_minutes: float = 10.0):
    store = get_user_store()
    now = time.time()
    active = []
    for user, ts in store["last_seen"].items():
        idle_min = (now - ts) / 60.0
        if idle_min <= max_idle_minutes:
            active.append({
                "user": user,
                "idle_minutes": round(idle_min, 1)
            })
    return active, store["logins"]


def login_screen():
    """
    Renders a custom login screen with logo, background and card.
    """
    login_css = """
    <style>
    .login-card {
        background-color: rgba(15,23,42,0.92);
        padding: 2.5rem 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.45);
        border: 1px solid rgba(148,163,184,0.45);
    }
    .login-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e5e7eb;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        font-size: 0.9rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .login-footer {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # Logo
        try:
            st.image("flex_logo.png", use_container_width=False, width=200)
        except Exception:
            st.markdown(
                "<p style='color:#f97316; text-align:center;'>flex_logo.png not found</p>",
                unsafe_allow_html=True
            )

        st.markdown('<div class="login-title">FLEX Optimisation Portal</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="login-subtitle">Sign in to explore PV + Battery + DA/ID optimisation scenarios.</div>',
            unsafe_allow_html=True
        )

        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in")

        if submit:
            ok, role = check_credentials(username, password)
            if ok:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = username
                st.session_state["auth_role"] = role

                register_login(username, role)

                st.rerun()
            else:
                st.error("❌ Incorrect username or password")

        st.markdown(
            '<div class="login-footer">Powered by FLEX · Secure access required</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


def ensure_authenticated():
    """
    Checks if user is authenticated; if not, shows login screen and returns False.
    """
    if st.session_state.get("authenticated"):
        return True

    login_screen()
    return False


# Gate everything behind login
if not ensure_authenticated():
    st.stop()

# At this point user is authenticated
update_last_seen(st.session_state.get("auth_user", "unknown"))

# default theme value
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# apply theme based on current setting
apply_theme()

# =====================================================================
#               CORE CALCULATION LOGIC (SINGLE SCENARIO)
# =====================================================================

def compute_scenario(
    load_kwh: float,
    pv_kwp: float,
    pv_yield: float,
    grid_price: float,
    fit_price: float,
    batt_capacity: float,
    batt_efficiency: float,
    cycles_per_day: float,
    sc_ratio_no_batt: float,
    da_spread: float,
    opt_capture: float,
    nonopt_capture: float,
    id_spread: float,
    id_capture: float,
):
    """
    Computes yearly energy flows and costs for a single household load
    under four configurations:
      1. No battery
      2. Battery – non-optimised
      3. Battery – DA-optimised
      4. Battery – DA + ID-optimised
    """

    pv_gen = pv_kwp * pv_yield

    # ----------------------------------------------------------
    # 1. NO BATTERY CASE
    # ----------------------------------------------------------
    pv_direct_sc = min(load_kwh * sc_ratio_no_batt, pv_gen)
    pv_export_no_batt = max(0.0, pv_gen - pv_direct_sc)
    grid_import_no_batt = max(0.0, load_kwh - pv_direct_sc)

    cost_no_batt = grid_import_no_batt * grid_price
    revenue_no_batt = pv_export_no_batt * fit_price
    net_no_batt = cost_no_batt - revenue_no_batt

    # ----------------------------------------------------------
    # 2. BATTERY — NON OPTIMISED
    # ----------------------------------------------------------
    batt_theoretical = batt_capacity * batt_efficiency * cycles_per_day * 365
    remaining_load = max(0.0, load_kwh - pv_direct_sc)
    batt_usable = min(batt_theoretical, remaining_load)

    pv_to_batt = batt_usable / batt_efficiency if batt_efficiency > 0 else 0
    pv_export_batt = max(0.0, pv_gen - pv_direct_sc - pv_to_batt)
    grid_import_batt = max(0.0, load_kwh - (pv_direct_sc + batt_usable))

    cost_batt_base = grid_import_batt * grid_price
    revenue_batt = pv_export_batt * fit_price
    net_batt_base = cost_batt_base - revenue_batt

    # Arbitrage energy only makes sense if grid import would have existed
    arbitrage_energy = batt_usable if grid_import_no_batt > 0 else 0

    # DA arbitrage with "dumb" use: non-optimised capture of spread
    arbitrage_non = arbitrage_energy * da_spread * nonopt_capture
    net_batt_nonopt = net_batt_base - arbitrage_non

    # ----------------------------------------------------------
    # 3. BATTERY — DA OPTIMISED
    # ----------------------------------------------------------
    arbitrage_opt = arbitrage_energy * da_spread * opt_capture
    net_batt_opt = net_batt_base - arbitrage_opt

    # ----------------------------------------------------------
    # 4. BATTERY — DA + INTRADAY OPTIMISED
    # ----------------------------------------------------------
    arbitrage_id = arbitrage_energy * id_spread * id_capture
    total_arbitrage_da_id = arbitrage_opt + arbitrage_id
    net_batt_da_id = net_batt_base - total_arbitrage_da_id

    # ----------------------------------------------------------
    # TABLE RESULTS
    # ----------------------------------------------------------
    df = pd.DataFrame([
        {
            "Configuration": "No battery",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": 0.0,
            "PV export (kWh)": pv_export_no_batt,
            "Grid import (kWh)": grid_import_no_batt,
            "Grid cost (€)": cost_no_batt,
            "EEG revenue (€)": revenue_no_batt,
            "DA arbitrage (€)": 0.0,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_no_batt,
        },
        {
            "Configuration": "Battery – non-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_non,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_batt_nonopt,
        },
        {
            "Configuration": "Battery – DA-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_opt,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_batt_opt,
        },
        {
            "Configuration": "Battery – DA+ID-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_opt,
            "ID arbitrage (€)": arbitrage_id,
            "Net annual cost (€)": net_batt_da_id,
        },
    ])
    return df


# =====================================================================
#                    PRESET MARKET SCENARIOS
# =====================================================================

def get_market_presets():
    """
    Returns a dict of named market presets:
    {preset_name: (da_spread, opt_cap, nonopt_cap, id_spread, id_cap)}
    """
    return {
        "Conservative (low spreads)": {
            "da_spread": 0.07,
            "opt_cap": 0.6,
            "nonopt_cap": 0.25,
            "id_spread": 0.10,
            "id_cap": 0.5,
        },
        "Base case (today-ish)": {
            "da_spread": 0.112,
            "opt_cap": 0.7,
            "nonopt_cap": 0.35,
            "id_spread": 0.18,
            "id_cap": 0.6,
        },
        "Volatile market (high spreads)": {
            "da_spread": 0.18,
            "opt_cap": 0.8,
            "nonopt_cap": 0.4,
            "id_spread": 0.30,
            "id_cap": 0.7,
        },
    }


# =====================================================================
#                  LIVE MARKET DATA (ENTSO-E – OPTIONAL)
# =====================================================================

@st.cache_data(ttl=3600)
def fetch_entsoe_da_prices(api_key: str,
                           start: datetime,
                           end: datetime,
                           bidding_zone: str = "10Y1001A1001A83F"):
    """
    Fetches day-ahead hourly prices from ENTSO-E for a given period.
    Returns a pandas Series indexed by datetime with prices in €/kWh.

    start/end are datetimes in UTC; bidding_zone default is DE/LU.
    """
    # ENTSO-E expects YYYYMMDDHHMM in UTC
    period_start = start.strftime("%Y%m%d%H%M")
    period_end = end.strftime("%Y%m%d%H%M")

    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": api_key,
        "documentType": "A44",           # Day-ahead prices
        "in_Domain": bidding_zone,
        "out_Domain": bidding_zone,
        "periodStart": period_start,
        "periodEnd": period_end,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    times = []
    prices = []

    # Namespace-agnostic parsing: check tag endings
    for ts in root.iter():
        if ts.tag.endswith("TimeSeries"):
            for period in ts:
                if not period.tag.endswith("Period"):
                    continue

                base_start = None
                # find timeInterval/start
                for child in period:
                    if child.tag.endswith("timeInterval"):
                        for grand in child:
                            if grand.tag.endswith("start"):
                                start_iso = grand.text
                                # e.g. 2024-01-01T23:00Z
                                if start_iso.endswith("Z"):
                                    start_iso = start_iso.replace("Z", "+00:00")
                                base_start = datetime.fromisoformat(start_iso)
                if base_start is None:
                    continue

                for child in period:
                    if child.tag.endswith("Point"):
                        pos = None
                        val = None
                        for el in child:
                            if el.tag.endswith("position"):
                                pos = int(el.text)
                            elif el.tag.endswith("price.amount"):
                                val = float(el.text)
                        if pos is not None and val is not None:
                            ts_point = base_start + pd.Timedelta(hours=pos - 1)
                            times.append(ts_point)
                            prices.append(val)

    if not times:
        raise ValueError("No prices parsed from ENTSO-E response.")

    # prices usually in €/MWh – convert to €/kWh
    s = pd.Series(prices, index=pd.DatetimeIndex(times, tz="UTC")) / 1000.0
    # convert to local German time
    s = s.tz_convert("Europe/Berlin")
    return s


def compute_daily_mean_spread(price_series: pd.Series) -> float:
    """
    Compute average daily spread = mean(max(price) - min(price)) across days.
    """
    if price_series.empty:
        raise ValueError("Empty price series for spread computation")

    daily = price_series.resample("1D").agg(["min", "max"])
    daily["spread"] = daily["max"] - daily["min"]
    return float(daily["spread"].mean())


def derive_spreads_from_prices(price_series: pd.Series):
    """
    Given an hourly price series (€/kWh), derive DA and ID spreads.

    DA spread = average daily (max - min).
    ID spread = 1.5 × DA spread (simple proxy for extra intraday volatility).
    """
    da_spread = compute_daily_mean_spread(price_series)
    id_spread = da_spread * 1.5
    return da_spread, id_spread


# =====================================================================
#        FUTURE MARKET DATA: API + MANUAL FILE UPLOAD (OPTIONAL)
# =====================================================================

@st.cache_data(ttl=3600)
def fetch_future_da_id_spreads(year: int, base_url: str, api_key: str | None = None):
    """
    Generic helper to fetch future DA/ID prices for a given year from a custom API.

    EXPECTED JSON FORMAT (example, you can adjust):

    {
      "da": { "2026-01-01T00:00": 0.12, "2026-01-01T01:00": 0.10, ... },
      "id": { "2026-01-01T00:00": 0.14, "2026-01-01T01:00": 0.11, ... }
    }

    Returns (da_spread, id_spread), both in €/kWh.
    """
    params = {"year": year}
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "da" not in data or "id" not in data:
        raise ValueError("API response must contain 'da' and 'id' price series.")

    da_series = pd.Series(data["da"])
    da_series.index = pd.to_datetime(da_series.index)

    id_series = pd.Series(data["id"])
    id_series.index = pd.to_datetime(id_series.index)

    da_spread = compute_daily_mean_spread(da_series)
    id_spread = compute_daily_mean_spread(id_series)

    return da_spread, id_spread


def compute_spreads_from_uploaded_file(uploaded_file):
    """
    Parse an uploaded CSV with columns:
      - timestamp
      - da_price
      - id_price
    and compute (da_spread, id_spread).
    """
    df = pd.read_csv(uploaded_file)

    required_cols = {"timestamp", "da_price", "id_price"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: timestamp, da_price, id_price"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    da_series = df["da_price"]
    id_series = df["id_price"]

    da_spread = compute_daily_mean_spread(da_series)
    id_spread = compute_daily_mean_spread(id_series)

    # Detect year(s) in file, just for info
    years = sorted(set(df.index.year))
    return da_spread, id_spread, years


# =====================================================================
#                           STREAMLIT UI
# =====================================================================

def main():
    username = st.session_state.get("auth_user", "")
    role = st.session_state.get("auth_role", "viewer")

    # ----------------------------------------------------------
    # SIDEBAR — USER, THEME, LOGOUT
    # ----------------------------------------------------------
    with st.sidebar:
        st.markdown(f"**👤 User:** `{username}`  \n**Role:** `{role}`")

        dark_on = st.toggle(
            "🌗 Dark mode",
            value=(st.session_state.get("theme", "dark") == "dark")
        )
        st.session_state["theme"] = "dark" if dark_on else "light"
        # theme will be applied on next rerun automatically

        st.markdown("---")
        if st.button("🚪 Logout"):
            for key in ("authenticated", "auth_user", "auth_role"):
                st.session_state.pop(key, None)
            st.rerun()

    st.title("⚡ FLEX – PV + Battery + DA & ID Optimisation (Germany / EEG)")
    st.markdown(
        """
Explore how PV, a battery, and **smart day-ahead + intraday optimisation** change your **annual energy cost**.

On the left you configure your system and market scenario, then explore the outputs in the tabs.
        """
    )

    # ----------------------------------------------------------
    # SIDEBAR — SYSTEM SETUP
    # ----------------------------------------------------------
    st.sidebar.header("🔧 System Setup")

    with st.sidebar.expander("💡 Quick explanation", expanded=True):
        st.markdown("""
### Energy Flow Mental Model
☀️ PV  
  │  
  ├──→ 🏠 Home (direct self-consumption)  
  │  
  ├──→ 🔋 Battery → 🏠 Home (shifted self-consumption)  
  │  
  └──→ 🔌 Grid (EEG export)

### What we compare
1️⃣ **No battery**  
2️⃣ **Battery – simple control** (increase self-consumption)  
3️⃣ **Battery – smart DA-optimised control**  
4️⃣ **Battery – DA + Intraday optimised** (uses both DA and ID markets)

### What “Net Annual Cost” means
- **Positive → you pay money overall**  
- **Negative → your PV exports earn you more than you pay the grid**
""")

    # --- SYSTEM PARAMETERS ---
    load_kwh = st.sidebar.number_input(
        "Annual household load (kWh)", min_value=0.0, value=3000.0, step=500.0
    )

    pv_kwp = st.sidebar.number_input("PV size (kWp)", 0.0, 20.0, 9.5, 0.1)
    pv_yield = st.sidebar.number_input("PV yield (kWh/kWp·yr)", 100.0, 1500.0, 950.0, 10.0)

    grid_price = st.sidebar.number_input("Grid price (€/kWh)", 0.0, 1.0, 0.39, 0.01)
    fit_price = st.sidebar.number_input("Feed-in tariff (€/kWh)", 0.0, 1.0, 0.08, 0.01)

    batt_capacity = st.sidebar.number_input("Battery capacity (kWh)", 0.0, 40.0, 8.8, 0.1)
    batt_eff = st.sidebar.slider("Battery efficiency (%)", 60, 100, 93) / 100
    cycles = st.sidebar.number_input("Cycles per day", 0.0, 2.0, 1.0, 0.1)

    sc_ratio = st.sidebar.slider(
        "Self-consumption ratio (no battery)",
        0.0, 1.0, 0.8, 0.05,
        help="How much of your consumption happens during PV production hours."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Market Parameters")

    # --- MARKET MODE: PRESETS VS EXPERT ---
    market_mode = st.sidebar.radio(
        "Market parameter mode",
        ["Presets", "Manual (expert)"],
        help="Presets use typical spreads & capture factors; manual lets you override everything."
    )

    presets = get_market_presets()

    if market_mode == "Presets":
        preset_name = st.sidebar.selectbox(
            "Choose market scenario",
            list(presets.keys()),
            index=1  # Base case as default
        )
        preset = presets[preset_name]

        da_spread = preset["da_spread"]
        opt_cap = preset["opt_cap"]
        nonopt_cap = preset["nonopt_cap"]
        id_spread = preset["id_spread"]
        id_cap = preset["id_cap"]

        st.sidebar.markdown(f"**Preset values ({preset_name}):**")
        st.sidebar.markdown(
            f"- DA spread: `{da_spread:.3f} €/kWh`\n"
            f"- DA optimiser capture: `{opt_cap:.2f}`\n"
            f"- Non-optimised DA capture: `{nonopt_cap:.2f}`\n"
            f"- ID spread: `{id_spread:.3f} €/kWh`\n"
            f"- ID optimiser capture: `{id_cap:.2f}`"
        )

    else:
        da_spread = st.sidebar.number_input(
            "Day-ahead price spread (€/kWh)", 0.0, 0.5, 0.112, 0.01
        )
        opt_cap = st.sidebar.slider(
            "DA optimiser spread capture", 0.0, 1.0, 0.7
        )
        nonopt_cap = st.sidebar.slider(
            "Non-optimised DA capture", 0.0, 1.0, 0.35
        )

        id_spread = st.sidebar.number_input(
            "Intraday price spread (€/kWh)",
            0.0, 1.0, 0.18, 0.01,
            help="Typical realised spread in intraday markets (often higher than DA)."
        )
        id_cap = st.sidebar.slider(
            "Intraday optimiser capture",
            0.0, 1.0, 0.6,
            help="How efficiently the control exploits intraday volatility."
        )

    # ----------------------------------------------------------
    # OPTIONAL: LIVE ENTSO-E DA / ID DATA (CURRENT MARKET)
    # ----------------------------------------------------------
    st.sidebar.markdown("---")
    use_live_market = st.sidebar.checkbox(
        "Use live DA/ID prices (ENTSO-E – beta)",
        value=False,
        help="Fetch recent DA prices from ENTSO-E and derive DA/ID spreads."
    )

    if use_live_market:
        api_key = st.secrets.get("ENTSOE_API_KEY")
        if not api_key:
            st.sidebar.error("ENTSOE_API_KEY missing in secrets. Add it in Manage app → Secrets.")
        else:
            # last 14 days as a reasonable window
            end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            start_dt = end_dt - pd.Timedelta(days=14)

            try:
                with st.spinner("📡 Fetching day-ahead prices from ENTSO-E..."):
                    prices = fetch_entsoe_da_prices(api_key, start_dt, end_dt)
                da_live, id_live = derive_spreads_from_prices(prices)
                da_spread = da_live
                id_spread = id_live
                st.sidebar.success(
                    f"Live DA spread: {da_spread:.3f} €/kWh\n"
                    f"Live ID spread (derived): {id_spread:.3f} €/kWh"
                )
            except Exception as e:
                st.sidebar.error(f"Error fetching ENTSO-E data: {e}")

    # ----------------------------------------------------------
    # OPTIONAL: FUTURE MARKET FORECAST (API or FILE)
    # ----------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Future market forecast (optional)")

    future_source = st.sidebar.radio(
        "Override spreads using:",
        ["None", "Upload CSV with future prices", "External forecast API"],
        index=0,
        help=(
            "Use either a file with future hourly DA/ID prices or a custom "
            "forecast API to simulate future-year arbitrage."
        ),
    )

    if future_source == "Upload CSV with future prices":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV (timestamp, da_price, id_price)",
            type=["csv"],
            help=(
                "CSV must contain columns: timestamp, da_price, id_price. "
                "Example row: 2026-01-01T00:00,0.12,0.14"
            )
        )
        if uploaded_file is not None:
            try:
                with st.spinner("📂 Processing uploaded price file..."):
                    da_fut, id_fut, years = compute_spreads_from_uploaded_file(uploaded_file)
                da_spread = da_fut
                id_spread = id_fut

                years_str = ", ".join(str(y) for y in years)
                st.sidebar.success(
                    f"Using uploaded future prices.\n"
                    f"Years in file: {years_str}\n\n"
                    f"DA spread: {da_spread:.3f} €/kWh\n"
                    f"ID spread: {id_spread:.3f} €/kWh"
                )
            except Exception as e:
                st.sidebar.error(f"Error reading uploaded file: {e}")

    elif future_source == "External forecast API":
        future_year = st.sidebar.number_input(
            "Forecast year",
            min_value=2024,
            max_value=2100,
            value=datetime.utcnow().year + 1,
            step=1,
            help="Year for which your custom API returns DA/ID price forecasts."
        )

        api_url = st.secrets.get("FUTURE_API_URL")
        api_key = st.secrets.get("FUTURE_API_KEY", None)

        if not api_url:
            st.sidebar.error(
                "FUTURE_API_URL missing in secrets. Add it in Manage app → Secrets.\n\n"
                "Expected API: JSON with 'da' and 'id' hourly price series."
            )
        else:
            try:
                with st.spinner(f"🌐 Fetching forecast prices for {future_year}..."):
                    da_fut, id_fut = fetch_future_da_id_spreads(future_year, api_url, api_key)
                da_spread = da_fut
                id_spread = id_fut
                st.sidebar.success(
                    f"Using forecast API for year {future_year}.\n\n"
                    f"DA spread: {da_spread:.3f} €/kWh\n"
                    f"ID spread: {id_spread:.3f} €/kWh"
                )
            except Exception as e:
                st.sidebar.error(f"Error calling forecast API: {e}")

    # ----------------------------------------------------------
    # RUN MODEL (with spinner)
    # ----------------------------------------------------------
    with st.spinner("⚙️ Running FLEX optimisation engine..."):
        df = compute_scenario(
            load_kwh, pv_kwp, pv_yield, grid_price, fit_price,
            batt_capacity, batt_eff, cycles, sc_ratio,
            da_spread, opt_cap, nonopt_cap,
            id_spread, id_cap,
        )

    df_display = df.copy()
    for col in df_display.columns:
        if col != "Configuration":
            df_display[col] = df_display[col].round(2)

    # ==========================================================
    # TABS (Admin tab only for admins)
    # ==========================================================
    tab_labels = [
        "🧮 Results",
        "📊 Parameter Guide",
        "⚙️ Optimisation Logic",
        "📈 Market Scenarios",
        "🧭 How to Read Results",
    ]
    if role == "admin":
        tab_labels.append("🛡 Admin")

    tabs = st.tabs(tab_labels)
    tab_results, tab_params, tab_logic, tab_market, tab_read = tabs[:5]
    tab_admin = tabs[5] if role == "admin" else None

    # ----------------------------------------------------------
    # TAB 1 — RESULTS
    # ----------------------------------------------------------
    with tab_results:
        st.header("🧮 Results")

        st.dataframe(df_display, use_container_width=True)

        costs = df.set_index("Configuration")["Net annual cost (€)"]
        nb = float(costs["No battery"])
        b_non = float(costs["Battery – non-optimised"])
        b_opt = float(costs["Battery – DA-optimised"])
        b_da_id = float(costs["Battery – DA+ID-optimised"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("No battery", f"{nb:,.0f} €")
        col2.metric(
            "Battery – non-opt",
            f"{b_non:,.0f} €",
            f"{nb - b_non:,.0f} € vs no battery"
        )
        col3.metric(
            "Battery – DA-opt",
            f"{b_opt:,.0f} €",
            f"{b_non - b_opt:,.0f} € vs non-opt"
        )
        col4.metric(
            "Battery – DA+ID-opt",
            f"{b_da_id:,.0f} €",
            f"{b_opt - b_da_id:,.0f} € vs DA-opt"
        )

        st.subheader("📉 Visual comparison – net annual cost")
        st.bar_chart(df.set_index("Configuration")["Net annual cost (€)"])

        st.subheader("💰 Arbitrage contributions")
        st.bar_chart(
            df.set_index("Configuration")[["DA arbitrage (€)", "ID arbitrage (€)"]]
        )

        with st.expander("📘 Short explanation"):
            st.markdown(
                """
### What these numbers mean:

- **Positive net cost** → you pay money overall.  
- **Negative net cost** → your PV exports earn you *more* than your grid import costs.  
- **Battery (non-opt)** shows how much you save simply by increasing PV self-consumption + some DA spread.  
- **Battery (DA-opt)** shows the value of reacting to day-ahead price patterns.  
- **Battery (DA+ID-opt)** adds the extra value of intraday volatility on top.
                """
            )

    # ----------------------------------------------------------
    # TAB 2 — PARAMETER GUIDE
    # ----------------------------------------------------------
    with tab_params:
        st.header("📊 Parameter Guide")
        st.markdown("""
This tab explains every slider in simple language.

### Load
Higher load → more room for the battery to provide value.

### PV Size & Yield
More PV → more self-consumption & EEG revenue.

### Battery
- Bigger battery = more energy shifting.
- Higher efficiency = less energy lost.
- More cycles/day = more annual throughput (if useful).

### Day-ahead Spread & Capture
- **DA spread**: average difference between cheap and expensive DA hours (€/kWh).
- **DA optimiser capture**: fraction of that spread you can realistically harvest with a good algorithm.
- **Non-optimised capture**: "dumb" usage that still happens to exploit some DA variation.

### Intraday Spread & Capture
- **Intraday spread**: additional price swings inside the day (ID market).
- **Intraday optimiser capture**: how much of that extra volatility the controller can exploit.

More spread × more capture = higher arbitrage revenue.
        """)

    # ----------------------------------------------------------
    # TAB 3 — OPTIMISATION LOGIC
    # ----------------------------------------------------------
    with tab_logic:
        st.header("⚙️ Optimisation Logic")
        st.markdown("""
The optimiser logic is simplified but reflects reality:

### Day-ahead (DA) optimisation

1. From the DA price curve, identify **cheap** and **expensive** hours.  
2. Plan charging in cheap hours and discharging in expensive ones.  
3. Respect battery constraints:
   - Energy capacity (kWh)
   - Power limits (implicitly via cycles/day)
   - Efficiency
   - Remaining household load

In the model:  
> **DA arbitrage (€)** = usable battery throughput × DA spread × capture factor

---

### Intraday (ID) optimisation

Intraday improves the DA plan by:

1. Reacting to **within-day price spikes/dips**.  
2. Correcting **forecast errors** from DA.  
3. Doing small adjustments around the DA schedule.

In the model:  
> **ID arbitrage (€)** = same usable energy × Intraday spread × ID capture factor

We assume DA and ID use *the same underlying battery cycles*,  
but ID monetises additional volatility → therefore **extra € on top of DA**.

---

All of this is aggregated to **yearly values** to keep the model fast & interpretable.
        """)

    # ----------------------------------------------------------
    # TAB 4 — MARKET SCENARIOS
    # ----------------------------------------------------------
    with tab_market:
        st.header("📈 Market Scenarios & Sensitivity")

        st.markdown(f"""
### Active mode: **{market_mode}**

- **Day-ahead spread**: `{da_spread:.3f} €/kWh`  
- **DA optimiser capture**: `{opt_cap:.2f}`  
- **Non-optimised DA capture**: `{nonopt_cap:.2f}`  
- **Intraday spread**: `{id_spread:.3f} €/kWh`  
- **Intraday optimiser capture**: `{id_cap:.2f}`  

Use this tab to reason about *how sensitive* your results are to market volatility.
        """)

        st.markdown("""
#### How to use this

- Switch between presets to see how **calm vs volatile markets** change the value of optimisation.
- Turn on **Manual (expert)** mode in the sidebar to plug in your own spread assumptions.
- Use ENTSO-E live data to reflect *current* volatility.
- Use CSV/API future forecasts to explore *future* arbitrage potential.
        """)

    # ----------------------------------------------------------
    # TAB 5 — HOW TO READ RESULTS
    # ----------------------------------------------------------
    with tab_read:
        st.header("🧭 How to Read the Results")

        st.markdown("""
## 1️⃣ What “Net Annual Cost” means

- **Positive number = you pay money overall.**
- **Negative number = your PV exports earn more than your grid costs.**

---

## 2️⃣ How to understand the four configurations

### 🏠 No battery
You use PV instantly, export the rest, and buy from grid when needed.

### 🔋 Battery – Non-optimised
Battery increases self-consumption and captures a bit of DA spread, but without smart scheduling.

### 🤖 Battery – DA-optimised
Same battery, but **planned** charging/discharging on the day-ahead price curve.

### ⚡ Battery – DA + Intraday-optimised
Builds on DA planning and adds **fast intraday reactions**  
→ monetises extra short-term volatility.

---

## 3️⃣ What to look for

- **Battery vs No Battery** → does a battery make sense at all?  
- **DA-opt vs Non-opt** → value of *basic smart control*.  
- **DA+ID vs DA-only** → incremental value of intraday optimisation.

If DA+ID only adds a few tens of €/year, a complex ID stack might not be worth it.
If it adds hundreds of €/year, it strongly supports advanced optimisation.

---

## 4️⃣ Why negative values happen

Large PV + small load → most energy exported at EEG →  
revenue > grid cost → **negative net cost**.

That is normal and correct in the model.
        """)

    # ----------------------------------------------------------
    # TAB 6 — ADMIN DASHBOARD (only for admins)
    # ----------------------------------------------------------
    if role == "admin" and tab_admin is not None:
        with tab_admin:
            st.header("🛡 Admin Dashboard – Active Users & Login History")

            active, history = get_active_users()

            st.subheader("🟢 Active users (last 10 minutes)")
            if active:
                st.table(active)
            else:
                st.info("No active users in the last 10 minutes.")

            st.subheader("📜 Login history")
            if history:
                st.table(history)
            else:
                st.info("No logins recorded yet.")


if __name__ == "__main__":
    main()
