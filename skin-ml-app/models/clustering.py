import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import base64
import io
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')
CLUSTER_COLORS = ['#6c63ff', '#00d4aa', '#ff6b6b', '#ffd93d', '#c084fc', '#38bdf8']
OUTLIER_COLOR  = '#6272a4'

# Unified mappings (Must match other modules)
LABEL_MAPS = {
    'Gender':         {'Female': 0, 'Male': 1},
    'Hydration_Level':{'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':      {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':    {'High': 0, 'Low': 1, 'Medium': 2},
}
FEATURE_COLS = ['Age', 'Gender_enc', 'Hydration_Level_enc', 'Oil_Level_enc', 'Sensitivity_enc', 'Humidity', 'Temperature']

def load_data():
    df = pd.read_csv(DATA_PATH)
    for col, mapping in LABEL_MAPS.items():
        df[col + '_enc'] = df[col].map(mapping)
    X = df[FEATURE_COLS].values
    return X, df, FEATURE_COLS

def encode_input(inp):
    return [
        float(inp.get('Age', 30)),
        LABEL_MAPS['Gender'].get(inp.get('Gender', 'Female'), 0),
        LABEL_MAPS['Hydration_Level'].get(inp.get('Hydration_Level', 'Medium'), 2),
        LABEL_MAPS['Oil_Level'].get(inp.get('Oil_Level', 'Medium'), 2),
        LABEL_MAPS['Sensitivity'].get(inp.get('Sensitivity', 'Low'), 1),
        float(inp.get('Humidity', 50.0)),
        float(inp.get('Temperature', 25.0)),
    ]

def _cluster_profiles(df, labels):
    df_c = df.copy()
    df_c['cluster'] = labels
    profiles = []
    for lbl in sorted(set(labels)):
        if lbl == -1: continue
        sub = df_c[df_c['cluster'] == lbl]
        profiles.append({
            'cluster': int(lbl) + 1,
            'size': int(len(sub)),
            'dominant_skin_type': sub['Skin_Type'].mode()[0] if len(sub) > 0 else 'N/A',
            'avg_age': round(float(sub['Age'].mean()), 1),
            'avg_skin_score': round(float(sub['skin_score'].mean()), 1),
            'color': CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
        })
    return profiles

_MODEL_CACHE = {}

def _generate_cluster_plot(X_scaled, labels, user_point_scaled, algorithm):
    """Generate a dark-themed PCA cluster scatter plot and return it as a base64 PNG."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    user_2d = pca.transform(user_point_scaled)

    unique_labels = sorted(set(labels))
    var_exp = pca.explained_variance_ratio_

    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#050510')
    ax.set_facecolor('#0d0d1f')   # dark navy background

    # ── Grid & spines ────────────────────────────────────────────────────────
    ax.grid(True, linestyle='--', linewidth=0.4, color='#1e1e40', alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2040')
        spine.set_linewidth(0.8)

    # ── Scatter clusters ────────────────────────────────────────────────────
    legend_handles = []
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            color = OUTLIER_COLOR
            label_str = 'Noise / Outlier'
            marker = 'x'
            alpha = 0.4
            size = 18
        else:
            color = CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            label_str = f'Group #{lbl + 1}'
            marker = 'o'
            alpha = 0.55
            size = 22

        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, s=size, alpha=alpha, marker=marker,
            linewidths=0.0, rasterized=True
        )
        legend_handles.append(
            Line2D([0], [0], marker=marker, color='w',
                   markerfacecolor=color, markersize=9,
                   label=label_str, linewidth=0)
        )

    # ── User point ──────────────────────────────────────────────────────────
    ax.scatter(
        user_2d[0, 0], user_2d[0, 1],
        c='#ffffff', s=220, marker='*', zorder=10,
        edgecolors='#6c63ff', linewidths=1.5,
        label='You'
    )
    ax.annotate(
        ' You', (user_2d[0, 0], user_2d[0, 1]),
        fontsize=10, color='white', fontweight='bold',
        va='center',
        path_effects=[pe.withStroke(linewidth=2, foreground='#050510')]
    )
    legend_handles.append(
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='#ffffff', markersize=13,
               markeredgecolor='#6c63ff', markeredgewidth=1.5,
               label='You', linewidth=0)
    )

    # ── Labels & title ──────────────────────────────────────────────────────
    ax.set_xlabel(
        f'Principal Component 1  ({var_exp[0]*100:.1f}% variance)',
        color='#a0a8c8', fontsize=10, labelpad=10
    )
    ax.set_ylabel(
        f'Principal Component 2  ({var_exp[1]*100:.1f}% variance)',
        color='#a0a8c8', fontsize=10, labelpad=10
    )
    ax.tick_params(colors='#6272a4', labelsize=8)

    algo_display = 'K-Means' if algorithm == 'kmeans' else 'DBSCAN'
    n_clusters = len([l for l in unique_labels if l != -1])
    ax.set_title(
        f'{algo_display} — {n_clusters} Clusters  ·  PCA Projection',
        color='white', fontsize=13, fontweight='700', pad=16
    )

    # ── Legend ──────────────────────────────────────────────────────────────
    legend = ax.legend(
        handles=legend_handles,
        loc='upper right',
        framealpha=0.25,
        facecolor='#0d0d1f',
        edgecolor='#2a2a55',
        fontsize=9,
        markerscale=1.1
    )
    # Style legend text colour (safe for all mpl versions)
    for text in legend.get_texts():
        text.set_color('#a0a8c8')

    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def predict_cluster(algorithm, encoded_features, params=None):
    cache_key = f"{algorithm}_{str(params)}"
    X, df, _ = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if cache_key in _MODEL_CACHE:
        model, labels = _MODEL_CACHE[cache_key]
    else:
        if algorithm == 'kmeans':
            k = int((params or {}).get('n_clusters', 4))
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:  # dbscan
            eps = float((params or {}).get('eps', 0.8))
            ms  = int((params or {}).get('min_samples', 10))
            model = DBSCAN(eps=eps, min_samples=ms)

        labels = model.fit_predict(X_scaled)
        _MODEL_CACHE[cache_key] = (model, labels)  # cache model+labels only

    # Always regenerate plot so the ★ reflects the current user point
    user_point_scaled = scaler.transform([encoded_features])
    plot_b64 = _generate_cluster_plot(X_scaled, labels, user_point_scaled, algorithm)

    x_in = scaler.transform([encoded_features])

    if algorithm == 'kmeans':
        cluster_idx = int(model.predict(x_in)[0])
    else:  # dbscan nearest-neighbour assignment
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(X_scaled)
        _, idx = nn.kneighbors(x_in)
        cluster_idx = labels[idx[0][0]]

    profiles = _cluster_profiles(df, labels)
    matched  = next((p for p in profiles if p['cluster'] == cluster_idx + 1), None)

    return {
        'cluster_name': f'Group #{cluster_idx + 1}' if cluster_idx != -1 else 'Unique (Outlier)',
        'profile':      matched,
        'algorithm':    algorithm.upper(),
        'plot':         plot_b64,
    }

def run_clustering(algorithm, params=None):
    X, df, _ = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=int((params or {}).get('n_clusters', 4)), random_state=42, n_init=10)
    else:
        model = DBSCAN(eps=float((params or {}).get('eps', 0.8)), min_samples=int((params or {}).get('min_samples', 10)))
    
    labels = model.fit_predict(X_scaled)
    profiles = _cluster_profiles(df, labels)
    
    return {
        'n_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
        'profiles': profiles
    }
