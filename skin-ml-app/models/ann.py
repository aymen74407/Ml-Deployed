import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import os

DATA_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')
FEATURE_COLS  = ['Age', 'Gender_enc', 'Hydration_Level_enc', 'Oil_Level_enc',
                 'Sensitivity_enc', 'Humidity', 'Temperature']
FEATURE_NAMES = ['Age', 'Gender', 'Hydration', 'Oil Level', 'Sensitivity', 'Humidity', 'Temp']
LABEL_MAPS = {
    'Gender':          {'Female': 0, 'Male': 1},
    'Hydration_Level': {'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':       {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':     {'High': 0, 'Low': 1, 'Medium': 2},
}


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    for col, mapping in LABEL_MAPS.items():
        df[col + '_enc'] = df[col].map(mapping)
    le = LabelEncoder()
    df['Skin_Type_enc'] = le.fit_transform(df['Skin_Type'])
    return df[FEATURE_COLS].values, df['Skin_Type_enc'].values, le


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


def _relu(x):
    return np.maximum(0, x)


def _b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


# ── Plot 1 : Network Activation Flow ────────────────────────────────────────

def _plot_network(model, scaler, encoded_features, le):
    W1, b1 = model.coefs_[0], model.intercepts_[0]
    x_sc   = scaler.transform([encoded_features])[0]
    h1_act = _relu(x_sc @ W1 + b1)
    proba  = model.predict_proba([x_sc])[0]

    n_in     = len(FEATURE_NAMES)
    n_h1_all = W1.shape[1]
    n_h1     = min(14, n_h1_all)
    n_out    = len(le.classes_)

    def layer_y(n):
        return np.linspace(0.06, 0.94, n)

    xs = dict(inp=0.07, h1=0.50, out=0.88)
    in_ys, h1_ys, out_ys = layer_y(n_in), layer_y(n_h1), layer_y(n_out)

    BG = '#050510'
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.02, 1.18)
    ax.set_ylim(-0.05, 1.10)
    ax.axis('off')

    # connections input → hidden
    W1s    = W1[:, :n_h1]
    contrib = W1s * x_sc[:, None]
    c_max   = np.abs(contrib).max() + 1e-9
    for i, iy in enumerate(in_ys):
        for j, hy in enumerate(h1_ys):
            a = float(abs(contrib[i, j]) / c_max)
            if a < 0.07:
                continue
            col = '#6c63ff' if contrib[i, j] > 0 else '#ff6b6b'
            ax.plot([xs['inp'], xs['h1']], [iy, hy],
                    color=col, alpha=min(a * 0.8, 0.72),
                    linewidth=0.25 + 2.5 * a, zorder=1)

    # connections hidden → output (dim)
    for hy in h1_ys:
        for oy in out_ys:
            ax.plot([xs['h1'], xs['out']], [hy, oy],
                    color='#1a1a3a', linewidth=0.5, alpha=0.45, zorder=1)

    # input nodes
    x_norm = (x_sc - x_sc.min()) / (x_sc.max() - x_sc.min() + 1e-9)
    for i, iy in enumerate(in_ys):
        c = plt.cm.plasma(0.2 + 0.8 * x_norm[i])
        ax.add_patch(plt.Circle((xs['inp'], iy), 0.030, color=c,
                                zorder=3, linewidth=1.2, edgecolor='#a29bfe'))
        ax.text(xs['inp'] - 0.04, iy, FEATURE_NAMES[i],
                ha='right', va='center', fontsize=8.5, color='#c0c8e8', fontweight='500')

    # hidden nodes
    h1_norm = h1_act[:n_h1] / (h1_act[:n_h1].max() + 1e-9)
    for j, hy in enumerate(h1_ys):
        c = plt.cm.cool(h1_norm[j])
        ax.add_patch(plt.Circle((xs['h1'], hy), 0.026, color=c,
                                zorder=3, linewidth=1.0, edgecolor='#00d4aa'))
    ax.text(xs['h1'], -0.025, f'showing {n_h1} of {n_h1_all} neurons',
            ha='center', va='top', fontsize=7.5, color='#6272a4')

    # output nodes
    max_p = proba.max()
    for k, oy in enumerate(out_ys):
        c = plt.cm.YlOrRd(0.15 + 0.85 * proba[k] / (max_p + 1e-9))
        ax.add_patch(plt.Circle((xs['out'], oy), 0.036, color=c,
                                zorder=3, linewidth=1.5, edgecolor='#ffd93d'))
        label_col = '#ffd93d' if proba[k] == max_p else '#a0a8c8'
        ax.text(xs['out'] + 0.05, oy + 0.01, le.classes_[k],
                ha='left', va='center', fontsize=9, color='white', fontweight='600')
        ax.text(xs['out'] + 0.05, oy - 0.035, f'{proba[k]*100:.1f}%',
                ha='left', va='center', fontsize=8, color=label_col)

    # layer labels
    for label, x in [('INPUT\nLAYER', xs['inp']),
                     ('HIDDEN LAYER', xs['h1']),
                     ('OUTPUT\nLAYER', xs['out'])]:
        ax.text(x, 1.04, label, ha='center', va='center',
                fontsize=8.5, color='#6272a4', fontweight='700', linespacing=1.4)

    # legend
    lx, ly = 0.70, 0.12
    ax.plot([lx, lx + 0.05], [ly, ly], color='#6c63ff', lw=2, alpha=0.9)
    ax.text(lx + 0.06, ly, 'Excitatory (+)', ha='left', va='center',
            fontsize=7.5, color='#6c63ff')
    ax.plot([lx, lx + 0.05], [ly - 0.05, ly - 0.05], color='#ff6b6b', lw=2, alpha=0.9)
    ax.text(lx + 0.06, ly - 0.05, 'Inhibitory (−)', ha='left', va='center',
            fontsize=7.5, color='#ff6b6b')

    ax.set_title('ANN — Activation Flow for Your Input',
                 color='white', fontsize=13, fontweight='700', pad=10)
    fig.tight_layout(pad=1.2)
    return _b64_fig(fig)


# ── Plot 2 : Feature Importance (first-layer weight magnitude) ───────────────

def _plot_importance(model):
    importance = np.abs(model.coefs_[0]).sum(axis=1)
    importance = importance / importance.sum() * 100
    colors = ['#6c63ff', '#00d4aa', '#ff6b6b', '#ffd93d', '#c084fc', '#38bdf8', '#ff79c6']

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#050510')
    ax.set_facecolor('#0d0d1f')
    bars = ax.barh(FEATURE_NAMES, importance, color=colors, edgecolor='none', height=0.55)
    for bar, val in zip(bars, importance):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left', fontsize=9,
                color='white', fontweight='600')
    ax.set_xlabel('Weight Magnitude Contribution (%)', color='#a0a8c8', fontsize=9)
    ax.set_title('Feature Importance via First-Layer Weight Magnitudes',
                 color='white', fontsize=11, fontweight='700', pad=12)
    ax.tick_params(colors='#a0a8c8', labelsize=9)
    ax.set_xlim(0, importance.max() * 1.25)
    ax.grid(axis='x', linestyle='--', linewidth=0.4, color='#1e1e40', alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e2040')
    fig.tight_layout(pad=1.2)
    return _b64_fig(fig)


# ── Cache + main predict ──────────────────────────────────────────────────────

_ANN_CACHE = {}


def predict_ann(encoded_features, params=None):
    p      = params or {}
    h1     = int(p.get('h1', 100))
    h2     = int(p.get('h2', 50))
    lr     = float(p.get('learning_rate', 0.001))
    hidden = (h1, h2)
    cache_key = f"ann_{hidden}_{lr}"

    X, y, le = load_and_preprocess()
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)

    if cache_key in _ANN_CACHE:
        model, le = _ANN_CACHE[cache_key]
    else:
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            learning_rate_init=lr,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        model.fit(X_scaled, y)
        _ANN_CACHE[cache_key] = (model, le)

    x_in     = scaler.transform([encoded_features])
    pred     = model.predict(x_in)[0]
    skin_type = le.inverse_transform([pred])[0]
    proba    = model.predict_proba(x_in)[0]
    proba_dict = {cls: round(float(v) * 100, 1) for cls, v in zip(le.classes_, proba)}

    val_score = round(model.best_validation_score_ * 100, 2) \
        if hasattr(model, 'best_validation_score_') else None

    return {
        'skin_type':       skin_type,
        'probabilities':   proba_dict,
        'val_accuracy':    val_score,
        'n_iter':          int(model.n_iter_),
        'hidden_layers':   list(hidden),
        'plot_network':    _plot_network(model, scaler, encoded_features, le),
        'plot_importance': _plot_importance(model),
    }
