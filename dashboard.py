import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

# ————————————————— Page config —————————————————————————————————————————————
st.set_page_config(
    page_title="Optim'eyes Dashboard",
    page_icon="🕶️",
    layout="wide"
)

# ————————————————— Sidebar with centered logo & client search ——————————————————
st.sidebar.markdown(
    """
    <div style="display:flex; justify-content:center; margin:16px 0;">
      <img src="logo.png" width="150" alt="Optim'eyes logo" />
    </div>
    """,
    unsafe_allow_html=True
)

# On ne lit plus de CSV locaux, on utilisera load_data() ci‑dessous
st.sidebar.header("🔍 Sélection client")

# ————————————————— Data Loading (Parquet S3) —————————————————————————————————
@st.cache_data(show_spinner=False)
def load_data():
    # on récupère le nom du bucket depuis vos Secrets
    bucket_name = st.secrets["S3_BUCKET"]
    base = f"s3://{bucket_name}"

    try:
        clients   = pd.read_parquet(f"{base}/clients.parquet")
        commandes = pd.read_parquet(f"{base}/commandes.parquet")
        produits  = pd.read_parquet(f"{base}/produits_montures.parquet")
    except Exception as e:
        st.error(f"❌ Impossible d’accéder au bucket S3 « {bucket_name} » : {e}")
        st.stop()

    commandes['Date_Commande'] = pd.to_datetime(
        commandes['Date_Commande'], errors='coerce'
    )
    return clients, commandes, produits

clients, commandes, produits = load_data()

# Sidebar : filtre sur le DataFrame chargé
q = st.sidebar.text_input("ID ou Nom")
if q:
    filt = clients[
        clients["Client_ID"].astype(str).str.contains(q, case=False, na=False) |
        clients["Nom"].str.contains(q, case=False, na=False)
    ]
else:
    filt = clients.copy()

if filt.empty:
    st.sidebar.warning("Aucun client trouvé")
    st.stop()

client_id = st.sidebar.selectbox("Client_ID", filt["Client_ID"].unique())

# ————————————————— Main page title (white & centered) ——————————————————————
st.markdown(
    """
    <div style='text-align:center; margin:24px 0;'>
      <span style='font-size:2.5rem; color:#FFFFFF; font-weight:700;'>
        📊 Dashboard de Recommandation — Optim'eyes
      </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ————————————————— Global CSS & WCAG —————————————————————————————————————
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background-color: #000 !important; }
:focus { outline: 3px dashed #ffdd57 !important; outline-offset: 4px; }
.kpi-card.primary { background: linear-gradient(135deg,#344767,#2a2a44); }
.kpi-card.secondary { background: linear-gradient(135deg,#3b3b60,#27293d); }
.kpi-card {
  margin:16px; border-radius:12px; padding:24px 16px; color:#fff;
  box-shadow:0 8px 20px rgba(0,0,0,0.4); text-align:center;
  transition:transform .2s; cursor:pointer;
}
.kpi-card:hover { transform:translateY(-6px); }
.kpi-icon { font-size:36px; margin-bottom:8px; color:#fff; }
.kpi-value { font-size:32px; font-weight:700; margin:6px 0; color:#fff; }
.kpi-label { font-size:14px!important; color:#ccc; }
.card-header { display:flex; justify-content:space-between; align-items:center; }
.badge       { background:#ffdd57; color:#1e1e2d; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:600; }
.stock-ok    { color:#8fbc8f; font-weight:600; }
.stock-low   { color:#ff6b6b; font-weight:600; }
h2,h3,h4     { color:#ffdd57; }
</style>
""", unsafe_allow_html=True)

# ————————————————— Forecast helper ————————————————————————————————————————
def forecast_sales(mid, commandes, periods=1):
    ts = (commandes[commandes['Monture_ID']==mid]
          .dropna(subset=['Date_Commande'])
          .set_index('Date_Commande')
          .resample('M').size().rename('sales'))
    if ts.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(ts.index.min(), ts.index.max(), freq='M')
    ts = ts.reindex(idx, fill_value=0)
    X = np.arange(len(ts)).reshape(-1,1)
    model = LinearRegression().fit(X, ts.values)
    future_X = np.arange(len(ts), len(ts)+periods).reshape(-1,1)
    preds = model.predict(future_X)
    future_idx = pd.date_range(
        ts.index[-1] + pd.offsets.MonthBegin(),
        periods=periods, freq='M'
    )
    return pd.concat([ts, pd.Series(preds, index=future_idx, name='sales')])

# ————————————————— KPI calculation —————————————————————————————————————————
num_clients   = clients["Client_ID"].nunique()
num_montures  = produits["Monture_ID"].nunique()
num_commandes = commandes.shape[0]
avg_basket    = commandes["Montant_Total (€)"].mean()
clv           = commandes.groupby("Client_ID")["Montant_Total (€)"].sum().mean()
recent_cut    = pd.Timestamp.now() - pd.DateOffset(months=3)
active        = commandes[commandes["Date_Commande"] >= recent_cut]["Client_ID"].nunique()
churn_rate    = 1 - active / num_clients
low_stock     = produits[produits["Stock"] < 5].shape[0]

# ————————————————— KPI Cards —————————————————————————————————————————————
primary = [
    ("👥", num_clients,   "Clients uniques"),
    ("🕶️", num_montures,  "Modèles montures"),
    ("📦", num_commandes, "Commandes totales")
]
for (icon, val, lbl), col in zip(primary, st.columns(3)):
    col.markdown(
        f"<div class='kpi-card primary'>"
        f"  <div class='kpi-icon'>{icon}</div>"
        f"  <div class='kpi-value'>{val}</div>"
        f"  <div class='kpi-label'>{lbl}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

secondary = [
    ("💶", f"{avg_basket:.2f} €", "Panier moyen"),
    ("📈", f"{clv:.2f} €",        "CLV moyen"),
    ("🔄", f"{churn_rate:.1%}",   "Taux de churn"),
    ("⚠️", f"{low_stock}",        "Stock critique")
]
for (icon, val, lbl), col in zip(secondary, st.columns(4)):
    col.markdown(
        f"<div class='kpi-card secondary'>"
        f"  <div class='kpi-icon'>{icon}</div>"
        f"  <div class='kpi-value'>{val}</div>"
        f"  <div class='kpi-label'>{lbl}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ————————————————— Distribution Charts (Altair) ——————————————————————————————
c1, c2 = st.columns(2)
with c1:
    st.subheader("Répartition par Type")
    df_type = produits["Type"].value_counts().rename_axis("Type").reset_index(name="Count")
    chart_type = (
        alt.Chart(df_type)
        .mark_bar(color="#69b3a2")
        .encode(
            alt.X("Type:N", sort="-y", title=None),
            alt.Y("Count:Q", title="Nombre de montures"),
            tooltip=["Type","Count"]
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_type, use_container_width=True)

with c2:
    st.subheader("Répartition par Forme")
    df_forme = produits["Forme"].value_counts().rename_axis("Forme").reset_index(name="Count")
    chart_forme = (
        alt.Chart(df_forme)
        .mark_bar(color="#4c78a8")
        .encode(
            alt.X("Forme:N", sort="-y", title=None),
            alt.Y("Count:Q", title="Nombre de montures"),
            tooltip=["Forme","Count"]
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_forme, use_container_width=True)

# ————————————————— Prepare recommenders ————————————————————————————————————
features = ["Marque","Modele","Type","Forme","Materiau","Couleur","Taille","Style"]
df_feat = produits.copy()
le = LabelEncoder()
for col in features:
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))
X_content = csr_matrix(pd.get_dummies(df_feat[features]))

pivot = commandes.pivot_table(
    index="Client_ID",
    columns="Monture_ID",
    aggfunc="size",
    fill_value=0
)
mat = csr_matrix(pivot.values.astype(float))
U,S,Vt = svds(mat, k=min(20, min(mat.shape)-1))
S = np.diag(S)
knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(mat)

def rec_contenu(last_mid, n=5):
    sub = produits[produits["Monture_ID"]==last_mid]
    if sub.empty: return []
    idx = sub.index[0]
    sims = cosine_similarity(X_content[idx], X_content)[0]
    ids = np.argsort(sims)[::-1][1:n+1]
    return produits.loc[ids, "Monture_ID"].tolist()

def rec_svd(cid, n=5):
    if cid not in pivot.index: return []
    i = list(pivot.index).index(cid)
    scores = U[i] @ S @ Vt
    ids = np.argsort(scores)[::-1][:n]
    return list(pivot.columns[ids])

def rec_knn(cid, k=5, n=5):
    if cid not in pivot.index: return []
    i = list(pivot.index).index(cid)
    nbrs = knn.kneighbors(mat[i], n_neighbors=k+1)[1].flatten()[1:]
    agg = pivot.iloc[nbrs].sum()
    curr = pivot.iloc[i]
    return agg[curr==0].sort_values(ascending=False).head(n).index.tolist()

methods = [
    ("🎭 Contenu", rec_contenu, "Dernière monture similaire"),
    ("👥 SVD",     rec_svd,     "Tendances globales"),
    ("👤 KNN",     rec_knn,     "Clients proches")
]

# ————————————————— Display recommendations ——————————————————————————————
st.markdown(f"## 🎯 Recommandations pour **{client_id}**")
cols = st.columns(len(methods))

for (title, func, desc), col in zip(methods, cols):
    with col:
        st.markdown(f"**{title}**  \n*{desc}*")
        if func is rec_contenu:
            hist = commandes[commandes["Client_ID"]==client_id]
            recs = func(hist["Monture_ID"].iloc[-1], 5) if not hist.empty else []
        elif func is rec_knn:
            recs = func(client_id, 5, 5)
        else:
            recs = func(client_id, 5)

        if not recs:
            st.info("— Pas assez de données —")
            continue

        # Forecast safe
        forecasts = {}
        for mid in recs:
            s = forecast_sales(mid, commandes, 1)
            forecasts[mid] = s.iloc[-1] if not s.empty else 0
        max_fc = max(forecasts.values()) if forecasts else 1

        for mid in sorted(forecasts, key=lambda m: forecasts[m], reverse=True):
            row = produits[produits["Monture_ID"]==mid].iloc[0]
            fc = forecasts[mid]
            with st.expander(f"🕶️ {row['Marque']} — {row['Modele']}"):
                st.markdown(
                    f"<div class='card-header'>"
                    f"<h4>🕶️ {row['Marque']} — {row['Modele']}</h4>"
                    f"<div class='badge'>+{int(fc)} ventes</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                serie = forecast_sales(mid, commandes, 1).reset_index()
                serie.columns = ["mois","sales"]
                chart = alt.Chart(serie).mark_line(point=True).encode(
                    x=alt.X("mois:T", axis=alt.Axis(format="%b %Y", ticks=False, grid=False)),
                    y=alt.Y("sales:Q", scale=alt.Scale(domain=[0, max_fc]), axis=alt.Axis(title="Ventes")),
                ).properties(height=100, width=250)
                st.altair_chart(chart, use_container_width=True)

                stock = row["Stock"]
                label = (
                    f"<span class='stock-low'>⚠️ Stock faible ({stock})</span>"
                    if stock < 5 else
                    f"<span class='stock-ok'>✅ Stock OK ({stock})</span>"
                )
                st.markdown(label, unsafe_allow_html=True)

                st.markdown(
                    f"- **Type :** {row['Type']}  \n"
                    f"- **Forme :** {row['Forme']}  \n"
                    f"- **Matériau :** {row['Materiau']}  \n"
                    f"- **Couleur :** {row['Couleur']}  \n"
                    f"- **Taille :** {row['Taille']}  \n"
                    f"- **Style :** {row['Style']}  \n"
                    f"- **Prix :** {row['Prix (€)']} €"
                )

st.markdown("---")
st.markdown("<p style='text-align:center;color:#777;'>Optim'eyes – © 2025</p>", unsafe_allow_html=True)
