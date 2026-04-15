"""
Experiment 6: K-Means and Hierarchical Clustering
Dataset: Used Car Price Prediction
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import issparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def build_preprocessor(numeric_columns, categorical_columns):
    """Create preprocessing pipeline for mixed data types."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_columns,
            ),
        ]
    )


def select_optimal_k(features, k_values):
    """Compute inertia and silhouette score for each K."""
    inertias = []
    silhouettes = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(features)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(features, labels))

    best_index = int(np.argmax(silhouettes))
    return inertias, silhouettes, k_values[best_index]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)


    print("\n1. LOADING DATA")
    file_path = os.path.join(script_dir, "Used_Car_Price_Prediction.csv")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Keep attributes that are useful for behavior/market-segment style clustering.
    numeric_cols = ["yr_mfr", "kms_run", "sale_price", "times_viewed", "total_owners"]
    categorical_cols = ["fuel_type", "body_type", "transmission", "make", "car_rating"]

    required_cols = numeric_cols + categorical_cols
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    print("\n2. FEATURE PREPARATION")
    data = df[required_cols].copy()

    # Fill missing numeric values with median.
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    # Fill missing categorical values with mode.
    for col in categorical_cols:
        mode_value = data[col].mode().iloc[0]
        data[col] = data[col].fillna(mode_value).astype(str)

    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    transformed = pipeline.fit_transform(data)
    if issparse(transformed):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.asarray(transformed)

    print(f"Transformed feature matrix shape: {transformed_dense.shape}")

    print("\n3. K-MEANS: SELECTING BEST K")
    k_values = list(range(2, 11))
    inertias, silhouettes, best_k = select_optimal_k(transformed_dense, k_values)

    print(f"Best K by silhouette score: {best_k}")
    for k, inertia, sil in zip(k_values, inertias, silhouettes):
        print(f"  K={k:<2} | Inertia={inertia:>12.2f} | Silhouette={sil:.4f}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(transformed_dense)
    kmeans_silhouette = silhouette_score(transformed_dense, kmeans_labels)

    print("\n4. HIERARCHICAL CLUSTERING")
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(transformed_dense)
    hierarchical_silhouette = silhouette_score(transformed_dense, hierarchical_labels)

    print(f"K-Means silhouette score       : {kmeans_silhouette:.4f}")
    print(f"Hierarchical silhouette score  : {hierarchical_silhouette:.4f}")

    print("\n5. DIMENSION REDUCTION FOR CLEAR VISUALIZATION")
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(transformed_dense)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA 2D explained variance: {explained:.2f}%")

    print("\n6. CREATING VISUALIZATIONS")

    # Build a manageable sample for dendrogram clarity/performance.
    max_dendrogram_points = 300
    rng = np.random.default_rng(42)
    if len(pca_2d) > max_dendrogram_points:
        sample_idx = rng.choice(len(pca_2d), size=max_dendrogram_points, replace=False)
        dendro_points = pca_2d[sample_idx]
    else:
        dendro_points = pca_2d

    link_matrix = linkage(dendro_points, method="ward")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Experiment 6: K-Means vs Hierarchical Clustering", fontsize=15, fontweight="bold")

    axes[0, 0].plot(k_values, inertias, marker="o", color="steelblue")
    axes[0, 0].axvline(best_k, color="gray", linestyle="--", label=f"Best K={best_k}")
    axes[0, 0].set_title("Elbow Method (K-Means)")
    axes[0, 0].set_xlabel("Number of Clusters (K)")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(k_values, silhouettes, marker="o", color="seagreen")
    axes[0, 1].axvline(best_k, color="gray", linestyle="--", label=f"Best K={best_k}")
    axes[0, 1].set_title("Silhouette Score by K")
    axes[0, 1].set_xlabel("Number of Clusters (K)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    dendrogram(link_matrix, ax=axes[0, 2], no_labels=True, color_threshold=None)
    axes[0, 2].set_title("Hierarchical Dendrogram (PCA Sample)")
    axes[0, 2].set_xlabel("Sample Index")
    axes[0, 2].set_ylabel("Ward Distance")

    scatter1 = axes[1, 0].scatter(
        pca_2d[:, 0],
        pca_2d[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        s=18,
        alpha=0.8,
        edgecolors="none",
    )
    axes[1, 0].set_title(f"K-Means Clusters in PCA Space (K={best_k})")
    axes[1, 0].set_xlabel("PCA 1")
    axes[1, 0].set_ylabel("PCA 2")
    axes[1, 0].grid(alpha=0.3)
    fig.colorbar(scatter1, ax=axes[1, 0], label="Cluster")

    scatter2 = axes[1, 1].scatter(
        pca_2d[:, 0],
        pca_2d[:, 1],
        c=hierarchical_labels,
        cmap="tab10",
        s=18,
        alpha=0.8,
        edgecolors="none",
    )
    axes[1, 1].set_title(f"Hierarchical Clusters in PCA Space (K={best_k})")
    axes[1, 1].set_xlabel("PCA 1")
    axes[1, 1].set_ylabel("PCA 2")
    axes[1, 1].grid(alpha=0.3)
    fig.colorbar(scatter2, ax=axes[1, 1], label="Cluster")

    kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    hierarchical_counts = pd.Series(hierarchical_labels).value_counts().sort_index()
    cluster_ids = sorted(set(kmeans_counts.index).union(set(hierarchical_counts.index)))

    width = 0.4
    x = np.arange(len(cluster_ids))
    axes[1, 2].bar(
        x - width / 2,
        [kmeans_counts.get(i, 0) for i in cluster_ids],
        width,
        label="K-Means",
        color="steelblue",
    )
    axes[1, 2].bar(
        x + width / 2,
        [hierarchical_counts.get(i, 0) for i in cluster_ids],
        width,
        label="Hierarchical",
        color="salmon",
    )
    axes[1, 2].set_title("Cluster Size Comparison")
    axes[1, 2].set_xlabel("Cluster ID")
    axes[1, 2].set_ylabel("Number of Cars")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(cluster_ids)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    main_plot_path = os.path.join(viz_dir, "clustering_results.png")
    plt.savefig(main_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Additional interpretable plot using two intuitive original attributes.
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    age = pd.Timestamp.now().year - data["yr_mfr"]
    ax2.scatter(
        age,
        data["sale_price"],
        c=kmeans_labels,
        cmap="tab10",
        s=22,
        alpha=0.8,
        edgecolors="none",
    )
    ax2.set_title("K-Means Clusters by Vehicle Age vs Sale Price")
    ax2.set_xlabel("Vehicle Age (Years)")
    ax2.set_ylabel("Sale Price")
    ax2.grid(alpha=0.3)
    interpretable_plot_path = os.path.join(viz_dir, "kmeans_age_vs_price.png")
    plt.tight_layout()
    plt.savefig(interpretable_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {main_plot_path}")
    print(f"Saved: {interpretable_plot_path}")

    print("\n7. CLUSTER PROFILE SUMMARY")
    profile_df = data.copy()
    profile_df["kmeans_cluster"] = kmeans_labels
    profile_df["hierarchical_cluster"] = hierarchical_labels

    kmeans_profile = (
        profile_df.groupby("kmeans_cluster")[numeric_cols]
        .mean()
        .round(2)
        .sort_index()
    )
    hierarchical_profile = (
        profile_df.groupby("hierarchical_cluster")[numeric_cols]
        .mean()
        .round(2)
        .sort_index()
    )

    print("\nK-Means cluster profile (numeric feature means):")
    print(kmeans_profile)

    print("\nHierarchical cluster profile (numeric feature means):")
    print(hierarchical_profile)

    output_df = df.copy()
    output_df["kmeans_cluster"] = kmeans_labels
    output_df["hierarchical_cluster"] = hierarchical_labels
    output_path = os.path.join(script_dir, "clustered_used_car_data.csv")
    output_df.to_csv(output_path, index=False)

    print(f"\nSaved clustered dataset: {output_path}")

    print("SUMMARY")
    print(f"Rows used: {len(data)}")
    print(f"Best K selected: {best_k}")
    print(f"K-Means silhouette score      : {kmeans_silhouette:.4f}")
    print(f"Hierarchical silhouette score : {hierarchical_silhouette:.4f}")


if __name__ == "__main__":
    main()
