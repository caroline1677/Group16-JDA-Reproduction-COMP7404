"""
JDA Paper Figure 4 - Strict Reproduction
Dataset: PIE1 -> PIE2, k=200, λ=1, T=20
Using only first 5 classes (200 samples: 100 source + 100 target)
"""

import numpy as np
import scipy.io as io
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import csv
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths - relative to jda_project directory
DATA_DIR = "./data/pie"
OUTPUT_DIR = "./paper_experiments"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pie_data(use_all_classes=True):
    """Load PIE data.
    Args:
        use_all_classes: If True, use all classes. If False, use only first 5 classes.
    PIE1 -> PIE05, PIE2 -> PIE07
    """
    src = io.loadmat(f"{DATA_DIR}/PIE1.mat")
    tar = io.loadmat(f"{DATA_DIR}/PIE2.mat")
    Xs, Ys = src["fea"].astype(np.float64), src["gnd"].ravel()
    Xt, Yt = tar["fea"].astype(np.float64), tar["gnd"].ravel()

    # Only keep first 5 classes if specified
    if not use_all_classes:
        mask_s = Ys <= 5
        mask_t = Yt <= 5
        Xs, Ys = Xs[mask_s], Ys[mask_s]
        Xt, Yt = Xt[mask_t], Yt[mask_t]

    # Normalize to [0,1] (和tune_parameters.py一致，不做Z-score)
    if Xs.max() > 1:
        Xs = Xs / 255.0
    if Xt.max() > 1:
        Xt = Xt / 255.0

    # Sort by class
    idx_s = np.argsort(Ys)
    idx_t = np.argsort(Yt)
    Xs, Ys = Xs[idx_s], Ys[idx_s]
    Xt, Yt = Xt[idx_t], Yt[idx_t]

    if use_all_classes:
        print(f"PIE1 (source, ALL classes): {Xs.shape}, labels: {np.unique(Ys)}")
        print(f"PIE2 (target, ALL classes): {Xt.shape}, labels: {np.unique(Yt)}")
    else:
        print(f"PIE1 (source, 5 classes): {Xs.shape}, labels: {np.unique(Ys)}")
        print(f"PIE2 (target, 5 classes): {Xt.shape}, labels: {np.unique(Yt)}")
    return Xs, Ys, Xt, Yt


def compute_mmd(Xs, Xt):
    """MMD as L2 distance between domain means (from experiment_d_convergence.py).
    分布距离：投影空间中源域与目标域均值的L2距离（MMD线性核代理）
    """
    mean_s = Xs.mean(axis=0)
    mean_t = Xt.mean(axis=0)
    return np.linalg.norm(mean_s - mean_t)


def compute_joint_mmd(Xs, Ys, Xt, Yt):
    """Joint MMD = 0.5 * marginal MMD + 0.5 * conditional MMD.
    使用L2距离作为MMD代理
    """
    C = len(np.unique(Ys))

    # Marginal MMD
    mmd_m = compute_mmd(Xs, Xt)

    # Conditional MMD
    mmd_c = 0
    for c in range(1, C+1):
        idx_s = np.where(Ys == c)[0]
        idx_t = np.where(Yt == c)[0]
        if len(idx_s) > 0 and len(idx_t) > 0:
            mmd_c += compute_mmd(Xs[idx_s], Xt[idx_t])
    mmd_c /= C

    return 0.5 * mmd_m + 0.5 * mmd_c


def run_nn(Xs, Ys, Xt, Yt):
    """1-NN baseline."""
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(Xs, Ys.ravel())
    return accuracy_score(Yt, clf.predict(Xt)) * 100


def run_pca(Xs, Ys, Xt, Yt, dim=200):
    """PCA baseline = PCA(n_components."""
    pca = PCA(n_components=min(dim, min(Xs.shape)-1))
    Xs_pca = pca.fit_transform(Xs)
    Xt_pca = pca.transform(Xt)
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(Xs_pca, Ys.ravel())
    acc = accuracy_score(Yt, clf.predict(Xt_pca)) * 100
    return acc, Xs_pca, Xt_pca


def run_tca(Xs, Ys, Xt, Yt, dim=200, lamb=1.0):
    """TCA - Transfer Component Analysis (using tune_parameters.py implementation)."""
    X = np.hstack((Xs.T, Xt.T))  # (n_features, ns+nt)
    m, n = X.shape  # m=n_features, n=samples
    ns, nt = len(Xs), len(Xt)

    # MMD matrix for marginal distribution
    e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
    M = e * e.T
    H = np.eye(n) - 1/n * np.ones((n, n))

    # Linear kernel (as in tune_parameters.py): K = X
    K = X  # (m, n) = (n_features, n_samples)
    a = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(m)
    b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

    a = (a + a.T) / 2
    b = (b + b.T) / 2

    try:
        w, V = scipy.linalg.eig(np.linalg.pinv(b) @ a)
    except:
        w, V = scipy.linalg.eig(a, b)
    w = np.real(w)
    V = np.real(V)
    ind = np.argsort(w)
    A = V[:, ind[:dim]]  # (m, dim)
    Z = A.T @ X  # (dim, n_samples)
    Z = np.real(Z)
    Z /= np.linalg.norm(Z, axis=0) + 1e-12
    Xs_tca, Xt_tca = Z[:, :ns].T, Z[:, ns:].T

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_tca, Ys.ravel())
    acc = accuracy_score(Yt, clf.predict(Xt_tca)) * 100
    return acc, Xs_tca, Xt_tca


def run_jda(Xs, Ys, Xt, Yt, dim=200, lamb=1.0, T=20):
    """JDA - Joint Distribution Adaptation (using tune_parameters.py implementation)."""
    X = np.hstack((Xs.T, Xt.T))  # (n_features, ns+nt)
    m, n = X.shape  # m=n_features, n=samples
    ns, nt = len(Xs), len(Xt)

    # MMD for marginal distribution
    e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
    M0 = e * e.T

    C = len(np.unique(Ys))
    H = np.eye(n) - 1/n * np.ones((n, n))

    Y_tar_pseudo = None
    A = None
    mmd_hist, acc_hist = [], []

    for t in range(T):
        N = np.zeros((n, n))

        # Conditional distribution adaptation
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(1, C + 1):
                e_c = np.zeros((n, 1))

                idx_s = np.where(Ys == c)[0]
                if len(idx_s) > 0:
                    e_c[idx_s] = 1.0 / len(idx_s)

                idx_t = np.where(Y_tar_pseudo == c)[0]
                if len(idx_t) > 0:
                    e_c[idx_t + ns] = -1.0 / len(idx_t)

                if np.sum(np.abs(e_c)) > 0:
                    N = N + np.dot(e_c, e_c.T)

        M = M0 + N

        # Linear kernel (as in tune_parameters.py): K = X
        K = X  # (m, n) = (n_features, n_samples)
        if A is None:
            # Initialize with PCA on first iteration
            pca = PCA(n_components=min(dim, m))
            pca.fit(X.T)
            A = pca.components_.T

        a = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(m)
        b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

        a = (a + a.T) / 2
        b = (b + b.T) / 2

        try:
            w, V = scipy.linalg.eig(np.linalg.pinv(b) @ a)
        except:
            w, V = scipy.linalg.eig(a, b)
        w = np.real(w)
        V = np.real(V)
        ind = np.argsort(w)
        A = V[:, ind[:dim]]  # (m, dim)
        Z = A.T @ X  # (dim, n_samples)
        Z = np.real(Z)
        Z /= np.linalg.norm(Z, axis=0) + 1e-12
        Xs_jda, Xt_jda = Z[:, :ns].T, Z[:, ns:].T

        # Compute joint MMD
        mmd = compute_joint_mmd(Xs_jda, Ys, Xt_jda, Y_tar_pseudo if Y_tar_pseudo is not None else Yt)
        mmd_hist.append(mmd)

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_jda, Ys.ravel())
        Y_tar_pseudo = clf.predict(Xt_jda)
        acc = accuracy_score(Yt, Y_tar_pseudo) * 100
        acc_hist.append(acc)

    # Final prediction with last A
    Z = A.T @ X  # (dim, n_samples)
    Z = np.real(Z)
    Z /= np.linalg.norm(Z, axis=0) + 1e-12
    Xs_jda, Xt_jda = Z[:, :ns].T, Z[:, ns:].T

    return acc_hist, mmd_hist, Xs_jda, Xt_jda


def compute_similarity_matrix(X, k=20):
    """Compute 20-NN similarity matrix using cosine similarity.
    Similarity = 1 / (1 + distance)
    """
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Similarity = 1 / (1 + distance)
    S = np.zeros((n, n))
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                S[i, j] = 1.0 / (1.0 + distances[i, j_idx])

    return S


if __name__ == "__main__":
    print("=" * 60)
    print("JDA Paper Figure 4 Reproduction")
    print("Dataset: PIE1 -> PIE2 (PIE05 -> PIE07)")
    print("=" * 60)

    # Load data - ALL classes for MMD/Accuracy calculation
    print("\n[1] Loading PIE data (ALL classes)...")
    Xs_all, Ys_all, Xt_all, Yt_all = load_pie_data(use_all_classes=True)

    # Parameters for each method (as specified)
    pca_dim = 20
    tca_dim = 200
    tca_lamb = 1  # TCA的lambda
    jda_dim = 75
    jda_lamb = 0.1  # JDA的lambda
    T = 20  # Iterations for JDA

    # Use ALL classes for MMD and Accuracy calculation
    Xs_sub = Xs_all
    Ys_sub = Ys_all
    Xt_sub = Xt_all
    Yt_sub = Yt_all

    n_samples_src = len(Xs_sub)
    n_samples_tgt = len(Xt_sub)
    print(f"\nUsing ALL classes for MMD/Acc: {n_samples_src} source + {n_samples_tgt} target = {n_samples_src + n_samples_tgt}")
    print(f"Parameters: PCA dim={pca_dim}, TCA dim={tca_dim}, TCA lambda={tca_lamb}, JDA dim={jda_dim}, JDA lambda={jda_lamb}, T={T}")

    # Run baseline methods
    print("\n[2] Running methods...")

    print("  - NN...")
    nn_acc = run_nn(Xs_sub, Ys_sub, Xt_sub, Yt_sub)
    nn_mmd = compute_joint_mmd(Xs_sub, Ys_sub, Xt_sub, Yt_sub)

    print("  - PCA...")
    pca_acc, Xs_pca, Xt_pca = run_pca(Xs_sub, Ys_sub, Xt_sub, Yt_sub, pca_dim)
    pca_mmd = compute_joint_mmd(Xs_pca, Ys_sub, Xt_pca, Yt_sub)

    print("  - TCA...")
    tca_acc, Xs_tca, Xt_tca = run_tca(Xs_sub, Ys_sub, Xt_sub, Yt_sub, tca_dim, tca_lamb)
    tca_mmd = compute_joint_mmd(Xs_tca, Ys_sub, Xt_tca, Yt_sub)

    print("  - JDA (T=20)...")
    jda_acc_hist, jda_mmd_hist, Xs_jda, Xt_jda = run_jda(
        Xs_sub, Ys_sub, Xt_sub, Yt_sub, jda_dim, jda_lamb, T
    )

    print(f"\n[3] Results:")
    print(f"    NN  : Acc={nn_acc:.2f}%, MMD={nn_mmd:.4f}")
    print(f"    PCA : Acc={pca_acc:.2f}%, MMD={pca_mmd:.4f}")
    print(f"    TCA : Acc={tca_acc:.2f}%, MMD={tca_mmd:.4f}")
    print(f"    JDA : Acc={jda_acc_hist[-1]:.2f}%, MMD={jda_mmd_hist[-1]:.4f}")

    # ============== Figure (a) & (b) ==============
    print("\n[4] Generating fig4_ab.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    it = np.arange(1, T+1)

    # (a) MMD Distance
    # NN, PCA, TCA are constants (no iteration), JDA iterates
    # Plot in order: TCA first (lowest), then JDA, then PCA, NN on top
    axes[0].plot(it, [tca_mmd] * T, 'b^-', markersize=8, linewidth=1.5, label='TCA')
    axes[0].plot(it, jda_mmd_hist, 'gd-', markersize=6, linewidth=2, label='JDA')
    axes[0].plot(it, [pca_mmd] * T, 'rs-', markersize=8, linewidth=1.5, label='PCA')
    axes[0].plot(it, [nn_mmd] * T, 'ko-', markersize=8, linewidth=1.5, label='NN')

    axes[0].set_xlabel('#Iterations', fontsize=12)
    axes[0].set_ylabel('MMD Distance', fontsize=12)
    axes[0].set_title('(a) MMD Distance w.r.t. #Iterations', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.5, T+0.5)

    # (b) Accuracy
    # Plot in order: TCA/JDA first (lower), then PCA, NN on top
    axes[1].plot(it, [tca_acc] * T, 'b^-', markersize=8, linewidth=1.5, label='TCA')
    axes[1].plot(it, jda_acc_hist, 'gd-', markersize=6, linewidth=2, label='JDA')
    axes[1].plot(it, [pca_acc] * T, 'rs-', markersize=8, linewidth=1.5, label='PCA')
    axes[1].plot(it, [nn_acc] * T, 'ko-', markersize=8, linewidth=1.5, label='NN')

    axes[1].set_xlabel('#Iterations', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('(b) Accuracy (%) w.r.t. #Iterations', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.5, T+0.5)
    axes[1].set_ylim(20, 70)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_ab.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/fig4_ab.png")
    plt.close()

    # ============== Figure (c) & (d) ==============
    print("\n[5] Generating fig4_cd.png (using first 5 classes only)...")

    # Load first 5 classes for similarity computation
    Xs_5, Ys_5, Xt_5, Yt_5 = load_pie_data(use_all_classes=False)

    # Run TCA and JDA on first 5 classes for similarity visualization
    _, Xs_tca_5, Xt_tca_5 = run_tca(Xs_5, Ys_5, Xt_5, Yt_5, tca_dim, tca_lamb)
    _, _, Xs_jda_5, Xt_jda_5 = run_jda(Xs_5, Ys_5, Xt_5, Yt_5, jda_dim, jda_lamb, T)

    # Compute similarity matrices (20-NN) using first 5 classes
    X_combined_tca = np.vstack([Xs_tca_5, Xt_tca_5])
    X_combined_jda = np.vstack([Xs_jda_5, Xt_jda_5])

    S_tca = compute_similarity_matrix(X_combined_tca, k=20)
    S_jda = compute_similarity_matrix(X_combined_jda, k=20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Common settings for heatmaps
    n_total = len(X_combined_tca)  # 365 (245 source + 120 target) for 5 classes only
    domain_boundary = len(Xs_5)  # 245 (between source and target)

    # Class boundaries in source domain (5 classes)
    unique_classes, counts = np.unique(Ys_5, return_counts=True)
    class_boundaries_src = np.cumsum(counts)

    # Class boundaries in target domain (5 classes)
    unique_classes_t, counts_t = np.unique(Yt_5, return_counts=True)
    class_boundaries_tgt = np.cumsum(counts_t)

    # (c) TCA similarity - grayscale (black=high, white=low), mask low similarity
    S_tca_masked = np.ma.masked_where(S_tca < 0.1, S_tca)
    im1 = axes[0].imshow(S_tca_masked, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('(c) Similarity of TCA embedding', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Example', fontsize=12)
    axes[0].set_ylabel('Example', fontsize=12)

    # Red lines for domain boundary (at index 245)
    axes[0].axvline(x=domain_boundary - 0.5, color='red', linewidth=2)
    axes[0].axhline(y=domain_boundary - 0.5, color='red', linewidth=2)

    # White dashed lines for class boundaries in source domain
    for cb in class_boundaries_src[:-1]:  # Skip last (245)
        axes[0].axvline(x=cb - 0.5, color='white', linewidth=1, linestyle='--')
        axes[0].axhline(y=cb - 0.5, color='white', linewidth=1, linestyle='--')

    # White dashed lines for class boundaries in target domain
    for cb in class_boundaries_tgt[:-1]:  # Skip last (120)
        axes[0].axvline(x=domain_boundary + cb - 0.5, color='white', linewidth=1, linestyle='--')
        axes[0].axhline(y=domain_boundary + cb - 0.5, color='white', linewidth=1, linestyle='--')

    # (d) JDA similarity - grayscale (black=high, white=low), mask low similarity
    S_jda_masked = np.ma.masked_where(S_jda < 0.1, S_jda)
    im2 = axes[1].imshow(S_jda_masked, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('(d) Similarity of JDA embedding', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Example', fontsize=12)
    axes[1].set_ylabel('Example', fontsize=12)

    # Red lines for domain boundary
    axes[1].axvline(x=domain_boundary - 0.5, color='red', linewidth=2)
    axes[1].axhline(y=domain_boundary - 0.5, color='red', linewidth=2)

    # White dashed lines for class boundaries in source domain
    for cb in class_boundaries_src[:-1]:
        axes[1].axvline(x=cb - 0.5, color='white', linewidth=1, linestyle='--')
        axes[1].axhline(y=cb - 0.5, color='white', linewidth=1, linestyle='--')

    # White dashed lines for class boundaries in target domain
    for cb in class_boundaries_tgt[:-1]:
        axes[1].axvline(x=domain_boundary + cb - 0.5, color='white', linewidth=1, linestyle='--')
        axes[1].axhline(y=domain_boundary + cb - 0.5, color='white', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_cd.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/fig4_cd.png")
    plt.close()

    # ============== CSV Output ==============
    print("\n[6] Generating fig4_results.csv...")
    csv_path = f'{OUTPUT_DIR}/fig4_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Iteration', 'Method', 'Accuracy(%)', 'MMD_Distance'])
        # Data for each iteration (1 to T)
        for t in range(1, T+1):
            # NN, PCA, TCA are constant (no iteration)
            writer.writerow([t, 'NN', f'{nn_acc:.2f}', f'{nn_mmd:.4f}'])
            writer.writerow([t, 'PCA', f'{pca_acc:.2f}', f'{pca_mmd:.4f}'])
            writer.writerow([t, 'TCA', f'{tca_acc:.2f}', f'{tca_mmd:.4f}'])
            # JDA has iteration history
            writer.writerow([t, 'JDA', f'{jda_acc_hist[t-1]:.2f}', f'{jda_mmd_hist[t-1]:.4f}'])
    print(f"    Saved: {csv_path}")

    print("\n" + "=" * 60)
    print("Done! Figure 4 generated successfully.")
    print(f"Output: {OUTPUT_DIR}/fig4_ab.png")
    print(f"        {OUTPUT_DIR}/fig4_cd.png")
    print(f"        {OUTPUT_DIR}/fig4_results.csv")
    print("=" * 60)
