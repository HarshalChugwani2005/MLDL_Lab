import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import warnings, os

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

print("Generating SVM Visualization with Hyperplane, Margins & Support Vectors...")

df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))
q33, q66 = df['sale_price'].quantile(0.33), df['sale_price'].quantile(0.66)
df['price_category'] = pd.cut(df['sale_price'], bins=[0, q33, q66, float('inf')], labels=['Low', 'Medium', 'High'])

feature_cols = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners',
                'fuel_type', 'body_type', 'transmission', 'make', 'car_rating']
df_model = df[feature_cols + ['price_category']].copy()
df_model.fillna(df_model.mode().iloc[0], inplace=True)

for col in ['fuel_type', 'body_type', 'transmission', 'make', 'car_rating']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

X = df_model.drop('price_category', axis=1).values
y_raw = df_model['price_category']
le_y = LabelEncoder()
y = le_y.fit_transform(y_raw)
class_labels = le_y.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"PCA Explained Variance: {pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}% = {sum(pca.explained_variance_ratio_)*100:.1f}%")

np.random.seed(42)
sample_idx = np.random.choice(len(X_2d), size=min(2000, len(X_2d)), replace=False)
X_sample = X_2d[sample_idx]
y_sample = y[sample_idx]

colors_class = {0: '#e74c3c', 1: '#2ecc71', 2: '#3498db'}
class_display = {0: 'High', 1: 'Low', 2: 'Medium'}
pair_configs = [
    (0, 1, 'High vs Low'),
    (0, 2, 'High vs Medium'),
    (1, 2, 'Low vs Medium'),
]

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('SVM Decision Boundaries — Used Car Price Classification\n(Projected to 2D via PCA)',
             fontsize=16, fontweight='bold', y=1.02)

for idx, (c1, c2, title) in enumerate(pair_configs):
    ax = axes[0, idx]
    mask = np.isin(y_sample, [c1, c2])
    X_pair = X_sample[mask]
    y_pair = y_sample[mask]
    y_binary = (y_pair == c2).astype(int)

    svm = SVC(kernel='rbf', C=10, gamma='auto', random_state=42)
    svm.fit(X_pair, y_binary)

    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2.5, linestyles='-')
    ax.contour(xx, yy, Z, levels=[-1, 1], colors='black', linewidths=1.2, linestyles='--')

    for cls_val, orig_cls in [(0, c1), (1, c2)]:
        mask_cls = y_binary == cls_val
        ax.scatter(X_pair[mask_cls, 0], X_pair[mask_cls, 1], c=colors_class[orig_cls],
                   label=class_display[orig_cls], s=20, alpha=0.6, edgecolors='white', linewidths=0.3)

    sv = svm.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black',
               linewidths=1.5, label=f'Support Vectors ({len(sv)})', zorder=5)

    ax.set_xlabel('PCA Component 1', fontsize=10)
    ax.set_ylabel('PCA Component 2', fontsize=10)
    ax.set_title(f'{title}\n(C=10, γ=auto, RBF)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)

ax_all = axes[1, 0]
svm_all = SVC(kernel='rbf', C=10, gamma='auto', random_state=42, decision_function_shape='ovr')
svm_all.fit(X_sample, y_sample)

x_min, x_max = X_sample[:, 0].min() - 1, X_sample[:, 0].max() + 1
y_min, y_max = X_sample[:, 1].min() - 1, X_sample[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z_all = svm_all.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

from matplotlib.colors import ListedColormap
cmap_bg = ListedColormap(['#ffcccc', '#ccffcc', '#ccccff'])
ax_all.contourf(xx, yy, Z_all, cmap=cmap_bg, alpha=0.5)
ax_all.contour(xx, yy, Z_all, colors='black', linewidths=1.5)

for cls in [0, 1, 2]:
    mask_cls = y_sample == cls
    ax_all.scatter(X_sample[mask_cls, 0], X_sample[mask_cls, 1], c=colors_class[cls],
                   label=class_display[cls], s=20, alpha=0.6, edgecolors='white', linewidths=0.3)

sv_all = svm_all.support_vectors_
ax_all.scatter(sv_all[:, 0], sv_all[:, 1], s=80, facecolors='none', edgecolors='black',
               linewidths=1.2, label=f'Support Vectors ({len(sv_all)})', zorder=5)
ax_all.set_xlabel('PCA Component 1', fontsize=10)
ax_all.set_ylabel('PCA Component 2', fontsize=10)
ax_all.set_title('Multi-class Decision Regions (OvR)\n(C=10, γ=auto, RBF)', fontsize=11, fontweight='bold')
ax_all.legend(fontsize=8, loc='best', framealpha=0.9)
ax_all.grid(True, alpha=0.2)

ax_lin = axes[1, 1]
svm_lin = SVC(kernel='linear', C=10, random_state=42)
svm_lin.fit(X_sample, y_sample)
Z_lin = svm_lin.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax_lin.contourf(xx, yy, Z_lin, cmap=cmap_bg, alpha=0.5)
ax_lin.contour(xx, yy, Z_lin, colors='black', linewidths=1.5)
for cls in [0, 1, 2]:
    mask_cls = y_sample == cls
    ax_lin.scatter(X_sample[mask_cls, 0], X_sample[mask_cls, 1], c=colors_class[cls],
                   label=class_display[cls], s=20, alpha=0.6, edgecolors='white', linewidths=0.3)
sv_lin = svm_lin.support_vectors_
ax_lin.scatter(sv_lin[:, 0], sv_lin[:, 1], s=80, facecolors='none', edgecolors='black',
               linewidths=1.2, label=f'Support Vectors ({len(sv_lin)})', zorder=5)
ax_lin.set_xlabel('PCA Component 1', fontsize=10)
ax_lin.set_ylabel('PCA Component 2', fontsize=10)
ax_lin.set_title('Linear Kernel Decision Regions\n(C=10)', fontsize=11, fontweight='bold')
ax_lin.legend(fontsize=8, loc='best', framealpha=0.9)
ax_lin.grid(True, alpha=0.2)

ax_concept = axes[1, 2]
ax_concept.set_xlim(-3, 3); ax_concept.set_ylim(-3, 3)
np.random.seed(0)
c1_pts = np.random.randn(30, 2) * 0.6 + np.array([-1.2, 0.8])
c2_pts = np.random.randn(30, 2) * 0.6 + np.array([1.2, -0.8])
ax_concept.scatter(c1_pts[:, 0], c1_pts[:, 1], c='#e74c3c', s=50, label='Class A', edgecolors='white', zorder=3)
ax_concept.scatter(c2_pts[:, 0], c2_pts[:, 1], c='#2ecc71', s=50, label='Class B', edgecolors='white', zorder=3)

xx_line = np.linspace(-3, 3, 100)
ax_concept.plot(xx_line, xx_line * 0.67 + 0.1, 'k-', linewidth=2.5, label='Hyperplane')
ax_concept.plot(xx_line, xx_line * 0.67 + 0.1 + 0.9, 'k--', linewidth=1.2, label='Margin (+)')
ax_concept.plot(xx_line, xx_line * 0.67 + 0.1 - 0.9, 'k--', linewidth=1.2, label='Margin (−)')
ax_concept.fill_between(xx_line, xx_line * 0.67 + 0.1 - 0.9, xx_line * 0.67 + 0.1 + 0.9,
                         alpha=0.1, color='gold', label='Margin Width')

sv_concept = [c1_pts[np.argmin(np.abs(c1_pts @ [0.67, -1] + 0.1 + 0.9))],
              c2_pts[np.argmin(np.abs(c2_pts @ [0.67, -1] + 0.1 - 0.9))]]
for sv in sv_concept:
    ax_concept.scatter(sv[0], sv[1], s=200, facecolors='none', edgecolors='black', linewidths=2, zorder=5)
ax_concept.scatter([], [], s=200, facecolors='none', edgecolors='black', linewidths=2, label='Support Vectors')

ax_concept.annotate('Hyperplane\n(Decision Boundary)', xy=(0.5, 0.6), fontsize=9,
                     fontweight='bold', ha='center', color='black',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax_concept.annotate('Maximum\nMargin', xy=(-2, 0.1), fontsize=9, fontweight='bold',
                     ha='center', color='#d4ac0d',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax_concept.set_xlabel('Feature 1', fontsize=10)
ax_concept.set_ylabel('Feature 2', fontsize=10)
ax_concept.set_title('SVM Concept Diagram\nHyperplane, Margins & Support Vectors', fontsize=11, fontweight='bold')
ax_concept.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax_concept.grid(True, alpha=0.2)
ax_concept.set_aspect('equal')

plt.tight_layout()
save_path = os.path.join(script_dir, 'visualizations/svm_detailed_visualization.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")
print("Done!")
