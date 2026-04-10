import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              accuracy_score, classification_report)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load & preprocess ─────────────────────────────────────────────────────
df = pd.read_csv('Titanic-Dataset.csv')
df = df.copy()
df['Age']      = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']     = df['Fare'].fillna(df['Fare'].median())
df['Sex'] = (df['Sex'] == 'female').astype(int)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']
X = df[features].values.astype(np.float32)
y = df['Survived'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def make_loaders(X_tr, y_tr, batch_size):
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# ── 2. Model definitions ─────────────────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self, input_dim=8, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1).clamp(1e-6, 1-1e-6)

class DeepNet(nn.Module):
    def __init__(self, input_dim=8, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1).clamp(1e-6, 1-1e-6)

# ── 3. Training loop ─────────────────────────────────────────────────────────
def train_model(model, loader, X_test_t, y_test_t, lr, epochs, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        ep_loss = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            vloss = criterion(val_pred, y_test_t).item()
            vacc  = ((val_pred > 0.5).float() == y_test_t).float().mean().item()
        train_losses.append(np.mean(ep_loss))
        val_losses.append(vloss)
        val_accs.append(vacc)
    return train_losses, val_losses, val_accs

X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test)

# ── 4. Base experiments ───────────────────────────────────────────────────────
BASE_LR, BASE_EPOCHS, BASE_BS, BASE_WD = 0.001, 60, 32, 0.0
loader_base = make_loaders(X_train, y_train, BASE_BS)

simple_model = SimpleNet()
simple_losses, simple_vlosses, simple_vaccs = train_model(
    simple_model, loader_base, X_test_t, y_test_t, BASE_LR, BASE_EPOCHS, BASE_WD)

deep_model = DeepNet()
deep_losses, deep_vlosses, deep_vaccs = train_model(
    deep_model, loader_base, X_test_t, y_test_t, BASE_LR, BASE_EPOCHS, BASE_WD)

def get_preds(model, X):
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X)).numpy()
    return probs, (probs > 0.5).astype(int)

simple_probs, simple_preds = get_preds(simple_model, X_test)
deep_probs,   deep_preds   = get_preds(deep_model,   X_test)

# ── 5. Hyperparameter sweeps ──────────────────────────────────────────────────
hp_results = []

lr_vals = [0.01, 0.001, 0.0001]
for lr in lr_vals:
    m = DeepNet(); ldr = make_loaders(X_train, y_train, BASE_BS)
    train_model(m, ldr, X_test_t, y_test_t, lr, 60, BASE_WD)
    pr, pd_ = get_preds(m, X_test)
    hp_results.append({'param': f'lr={lr}', 'group': 'Learning Rate',
                        'auc': roc_auc_score(y_test, pr), 'acc': accuracy_score(y_test, pd_)})

wd_vals = [0.0, 0.001, 0.01]
for wd in wd_vals:
    m = DeepNet(dropout=0.3); ldr = make_loaders(X_train, y_train, BASE_BS)
    train_model(m, ldr, X_test_t, y_test_t, BASE_LR, 60, wd)
    pr, pd_ = get_preds(m, X_test)
    hp_results.append({'param': f'wd={wd}', 'group': 'L2 Regularisation',
                        'auc': roc_auc_score(y_test, pr), 'acc': accuracy_score(y_test, pd_)})

bs_vals = [16, 32, 64]
for bs in bs_vals:
    m = DeepNet(); ldr = make_loaders(X_train, y_train, bs)
    train_model(m, ldr, X_test_t, y_test_t, BASE_LR, 60, BASE_WD)
    pr, pd_ = get_preds(m, X_test)
    hp_results.append({'param': f'bs={bs}', 'group': 'Batch Size',
                        'auc': roc_auc_score(y_test, pr), 'acc': accuracy_score(y_test, pd_)})

ep_vals = [20, 60, 150]
for ep in ep_vals:
    m = DeepNet(); ldr = make_loaders(X_train, y_train, BASE_BS)
    train_model(m, ldr, X_test_t, y_test_t, BASE_LR, ep, BASE_WD)
    pr, pd_ = get_preds(m, X_test)
    hp_results.append({'param': f'ep={ep}', 'group': 'Epochs',
                        'auc': roc_auc_score(y_test, pr), 'acc': accuracy_score(y_test, pd_)})

dr_vals = [0.0, 0.3, 0.5]
for dr in dr_vals:
    m = DeepNet(dropout=dr); ldr = make_loaders(X_train, y_train, BASE_BS)
    train_model(m, ldr, X_test_t, y_test_t, BASE_LR, 60, BASE_WD)
    pr, pd_ = get_preds(m, X_test)
    hp_results.append({'param': f'drop={dr}', 'group': 'Dropout',
                        'auc': roc_auc_score(y_test, pr), 'acc': accuracy_score(y_test, pd_)})

hp_df = pd.DataFrame(hp_results)

# ── 6. ROC curves ─────────────────────────────────────────────────────────────
s_fpr, s_tpr, _ = roc_curve(y_test, simple_probs)
d_fpr, d_tpr, _ = roc_curve(y_test, deep_probs)
s_auc = roc_auc_score(y_test, simple_probs)
d_auc = roc_auc_score(y_test, deep_probs)

# ── 7. PLOTTING ───────────────────────────────────────────────────────────────
DARK   = '#0d1117'
CARD   = '#161b22'
ACCENT = '#58a6ff'
GREEN  = '#3fb950'
ORANGE = '#f78166'
PURPLE = '#bc8cff'
YELLOW = '#e3b341'
TEXT   = '#e6edf3'
MUTED  = '#8b949e'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': CARD,
    'axes.edgecolor': '#30363d', 'axes.labelcolor': TEXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': '#21262d',
    'grid.alpha': 0.7, 'font.family': 'monospace',
    'legend.facecolor': '#1c2128', 'legend.edgecolor': '#30363d',
})

fig = plt.figure(figsize=(22, 26), facecolor=DARK)
fig.suptitle('TITANIC SURVIVAL — NEURAL NETWORK ANALYSIS',
             fontsize=18, fontweight='bold', color=TEXT,
             y=0.98, fontfamily='monospace')

gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.52, wspace=0.42,
                        top=0.95, bottom=0.04, left=0.06, right=0.97)

# ── Row 0: Training curves ────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_title('Training Loss  —  Simple vs Deep', color=ACCENT, fontsize=11, pad=8)
ax0.plot(simple_losses,  color=ACCENT,  lw=2,   label='Simple  train')
ax0.plot(simple_vlosses, color=ACCENT,  lw=2,   linestyle='--', alpha=0.6, label='Simple  val')
ax0.plot(deep_losses,    color=GREEN,   lw=2,   label='Deep    train')
ax0.plot(deep_vlosses,   color=GREEN,   lw=2,   linestyle='--', alpha=0.6, label='Deep    val')
ax0.legend(fontsize=8); ax0.set_xlabel('Epoch'); ax0.set_ylabel('BCE Loss')
ax0.grid(True)

ax1 = fig.add_subplot(gs[0, 2:])
ax1.set_title('Validation Accuracy  —  Simple vs Deep', color=ACCENT, fontsize=11, pad=8)
ax1.plot(simple_vaccs, color=ACCENT, lw=2, label='Simple')
ax1.plot(deep_vaccs,   color=GREEN,  lw=2, label='Deep')
ax1.legend(fontsize=8); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.5, 1.0); ax1.grid(True)

# ── Row 1: ROC + Confusion matrices ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.set_title('ROC Curves', color=ACCENT, fontsize=11, pad=8)
ax2.plot(s_fpr, s_tpr, color=ACCENT, lw=2, label=f'Simple  AUC={s_auc:.3f}')
ax2.plot(d_fpr, d_tpr, color=GREEN,  lw=2, label=f'Deep    AUC={d_auc:.3f}')
ax2.plot([0,1],[0,1], color=MUTED, lw=1, linestyle=':')
ax2.fill_between(s_fpr, s_tpr, alpha=0.08, color=ACCENT)
ax2.fill_between(d_fpr, d_tpr, alpha=0.08, color=GREEN)
ax2.legend(fontsize=9); ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
ax2.grid(True)

cm_s = confusion_matrix(y_test, simple_preds)
cm_d = confusion_matrix(y_test, deep_preds)

ax3 = fig.add_subplot(gs[1, 2])
sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=0.5, linecolor='#0d1117',
            annot_kws={'size': 13, 'weight': 'bold', 'color': TEXT},
            cbar=False)
ax3.set_title('Confusion — Simple', color=ACCENT, fontsize=10, pad=6)
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
ax3.set_xticklabels(['Died','Survived']); ax3.set_yticklabels(['Died','Survived'])

ax4 = fig.add_subplot(gs[1, 3])
sns.heatmap(cm_d, annot=True, fmt='d', cmap='Greens', ax=ax4,
            linewidths=0.5, linecolor='#0d1117',
            annot_kws={'size': 13, 'weight': 'bold', 'color': TEXT},
            cbar=False)
ax4.set_title('Confusion — Deep', color=GREEN, fontsize=10, pad=6)
ax4.set_xlabel('Predicted'); ax4.set_ylabel('Actual')
ax4.set_xticklabels(['Died','Survived']); ax4.set_yticklabels(['Died','Survived'])

# ── Row 2: Error distribution ─────────────────────────────────────────────────
simple_errors = np.abs(simple_probs - y_test)
deep_errors   = np.abs(deep_probs   - y_test)

ax5 = fig.add_subplot(gs[2, :2])
ax5.set_title('Prediction Error Distribution', color=ACCENT, fontsize=11, pad=8)
ax5.hist(simple_errors, bins=30, color=ACCENT, alpha=0.6, label='Simple', density=True)
ax5.hist(deep_errors,   bins=30, color=GREEN,  alpha=0.6, label='Deep',   density=True)
ax5.axvline(simple_errors.mean(), color=ACCENT, lw=1.5, linestyle='--',
            label=f'Simple μ={simple_errors.mean():.3f}')
ax5.axvline(deep_errors.mean(),   color=GREEN,  lw=1.5, linestyle='--',
            label=f'Deep   μ={deep_errors.mean():.3f}')
ax5.legend(fontsize=8); ax5.set_xlabel('|prob − true|'); ax5.set_ylabel('Density')
ax5.grid(True)

# Probability calibration
ax6 = fig.add_subplot(gs[2, 2:])
ax6.set_title('Predicted Probability Distribution', color=ACCENT, fontsize=11, pad=8)
survived = y_test == 1
ax6.hist(simple_probs[survived],    bins=20, color=ACCENT, alpha=0.5, label='Simple — Survived', density=True)
ax6.hist(simple_probs[~survived],   bins=20, color=ORANGE, alpha=0.5, label='Simple — Died',     density=True)
ax6.hist(deep_probs[survived],      bins=20, color=GREEN,  alpha=0.5, label='Deep   — Survived', density=True, histtype='step', lw=2)
ax6.hist(deep_probs[~survived],     bins=20, color=YELLOW, alpha=0.5, label='Deep   — Died',     density=True, histtype='step', lw=2)
ax6.legend(fontsize=7); ax6.set_xlabel('Predicted Probability'); ax6.set_ylabel('Density')
ax6.grid(True)

# ── Rows 3-4: Hyperparameter panels ──────────────────────────────────────────
groups    = ['Learning Rate', 'L2 Regularisation', 'Batch Size', 'Epochs', 'Dropout']
clrs      = [ACCENT, GREEN, ORANGE, PURPLE, YELLOW]
positions = [(3,0), (3,1), (3,2), (3,3), (4,0)]

for idx, (grp, clr, pos) in enumerate(zip(groups, clrs, positions)):
    sub  = hp_df[hp_df['group'] == grp]
    ax   = fig.add_subplot(gs[pos[0], pos[1]])
    x    = np.arange(len(sub))
    w    = 0.35
    bars1 = ax.bar(x - w/2, sub['auc'], w, color=clr, alpha=0.85, label='AUC')
    bars2 = ax.bar(x + w/2, sub['acc'], w, color=clr, alpha=0.4,  label='Acc')
    ax.set_xticks(x); ax.set_xticklabels(sub['param'], fontsize=7, rotation=12)
    ax.set_ylim(0.6, 1.0)
    ax.set_title(grp, color=clr, fontsize=9, pad=5)
    ax.legend(fontsize=7); ax.grid(True, axis='y')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=6.5, color=TEXT)

# ── Summary bar chart (col 1-3, row 4) ───────────────────────────────────────
ax_sum = fig.add_subplot(gs[4, 1:])
metrics_simple = {
    'Accuracy': accuracy_score(y_test, simple_preds),
    'AUC':      s_auc,
    'Prec.':    confusion_matrix(y_test, simple_preds)[1,1] / simple_preds.sum(),
    'Recall':   confusion_matrix(y_test, simple_preds)[1,1] / y_test.sum(),
}
metrics_deep = {
    'Accuracy': accuracy_score(y_test, deep_preds),
    'AUC':      d_auc,
    'Prec.':    confusion_matrix(y_test, deep_preds)[1,1] / deep_preds.sum(),
    'Recall':   confusion_matrix(y_test, deep_preds)[1,1] / y_test.sum(),
}
xlabels = list(metrics_simple.keys())
x = np.arange(len(xlabels)); w = 0.35
b1 = ax_sum.bar(x - w/2, list(metrics_simple.values()), w, color=ACCENT, alpha=0.85, label='Simple (1×16)')
b2 = ax_sum.bar(x + w/2, list(metrics_deep.values()),   w, color=GREEN,  alpha=0.85, label='Deep (5 layers)')
ax_sum.set_xticks(x); ax_sum.set_xticklabels(xlabels, fontsize=10)
ax_sum.set_ylim(0.6, 1.05)
ax_sum.set_title('Model Comparison — Key Metrics', color=TEXT, fontsize=11, pad=8)
ax_sum.legend(fontsize=9)
ax_sum.grid(True, axis='y')
for bar in list(b1) + list(b2):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5, color=TEXT)

import os
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'titanic_nn_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f'Chart saved to: {os.path.abspath(save_path)}')
print("Saved!")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Simple Net ──")
print(classification_report(y_test, simple_preds, target_names=['Died','Survived']))
print(f"AUC: {s_auc:.4f}")
print("\n── Deep Net ──")
print(classification_report(y_test, deep_preds, target_names=['Died','Survived']))
print(f"AUC: {d_auc:.4f}")
print("\n── Hyperparameter Results ──")
print(hp_df.to_string(index=False))
