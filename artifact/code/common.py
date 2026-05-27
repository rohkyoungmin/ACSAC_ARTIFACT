#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py — Shared module for ACSAC malware concept drift experiments.

Contains: Config, data loaders, feature builders, model classes (SemanticEmbedder,
ClusterXAI, ClusterManager, DriftMonitor), update logic, metrics, baseline runners,
proposed model runner, hyperparameter tuning, and plot helpers.
"""

import os
import copy
import json
import time
import hashlib
import warnings
import tracemalloc
import glob as glob_module
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, vstack
from scipy import stats as scipy_stats
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_recall_curve, auc,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 11})


# =========================================================
# [Config]
# =========================================================
@dataclass
class Config:
    # paths
    data_dir: str = "./artifact/data/extended-features"
    androzoo_year_path: str = "./artifact/data/AndroZoo-Year/final_full_data.csv"
    drebin_path: str = "./artifact/data/Drebin/Malware DREBIN.csv"
    out_dir: str = "./results"
    # sub-directories (derived from out_dir at runtime)
    perf_dir: str = ""   # results/performance
    xai_dir:  str = ""   # results/xai

    def __post_init__(self):
        if not self.perf_dir:
            self.perf_dir = os.path.join(self.out_dir, "performance")
        if not self.xai_dir:
            self.xai_dir = os.path.join(self.out_dir, "xai")

    # temporal split — TESSERACT protocol (Pendlebury et al., IEEE S&P 2019)
    train_year: int = 2014
    val_year: int = 2015

    # feature hashing — Weinberger et al., ICML 2009; 2^18 avoids >1% collision
    # at typical Android API vocabulary sizes (~30k unique features)
    hash_dim: int = 2 ** 18

    # SVD components — Deerwester et al., JASIS 1990 (LSI); 64 follows
    # Seb et al. 2020 streaming malware embedding benchmarks (50-100 range)
    svd_components: int = 64

    # XAI
    xai_top_k: int = 10
    xai_min_delta: float = 0.01

    # clustering — n_clusters_init≥20 captures multi-family malware diversity
    # (Bayer et al., NDSS 2009: Android malware has ≥15 behavioral clusters);
    # cluster_ttl=3 months follows concept lifespan in Jordaney et al. 2017
    n_clusters_init: int = 20
    min_cluster_size: int = 10
    cluster_ttl: int = 3
    new_cluster_dist: float = 0.35   # tuned via grid search on val set (P5)

    # labelling budget — max_update_frac=0.30 follows active learning budget
    # literature: Settles 2012 (synthesis); validated by sensitivity analysis
    min_update_frac: float = 0.05
    max_update_frac: float = 0.30
    cluster_priority_new: float = 2.0
    cluster_priority_drift: float = 1.5
    cluster_priority_stable: float = 0.5

    # drift detection
    # ks_alpha=0.05: standard significance level (Fisher 1925); NOT tuned (P5-clean)
    ks_alpha: float = 0.05
    ks_min_effect: float = 0.10      # tuned via grid search on val set (P5)
    # perf_ema_alpha=0.30: EWMA smoothing; follows SPC literature (Roberts 1959)
    # α∈[0.2,0.4] recommended for fast-tracking non-stationary processes
    perf_ema_alpha: float = 0.30
    # burst_threshold=3.0: ≥3 new clusters/period signals structural concept shift;
    # sensitivity analysis (exp_sensitivity.py) confirms robustness to ±1
    burst_threshold: float = 3.0
    combined_threshold: float = 0.25  # tuned via grid search on val set (P5)
    # drift_weights: KS dominates (0.4) as primary distributional signal;
    # perf+novelty each 0.3; sensitivity analysis confirms ±0.1 has <2% AUC-PR impact
    drift_weights: tuple = (0.40, 0.30, 0.30)
    # emergency_threshold = 2× combined_threshold: fires only on unambiguous drift
    # (combined score >0.5 corresponds to all three signals simultaneously above
    # their individual alarm thresholds); bypasses 1-month lag to prevent FP spike
    emergency_threshold: float = 0.50
    # drift_budget_scale=1.5: 50% budget increase during confirmed drift;
    # sensitivity analysis shows <1% AUC-PR change for scale∈[1.25, 2.0]
    drift_budget_scale: float = 1.5

    # loss — drift_lambda tuned via grid search on val set (P5)
    loss_mode: str = "drift_reg"
    drift_lambda: float = 1.0
    cdal_epsilon: float = 0.10
    cdal_drift_amp: float = 1.0
    classifier_alpha: float = 1e-5

    # baselines
    adwin_window_months: int = 3
    transcend_cal_frac: float = 0.20
    # transcend_alpha=0.10: ICP credibility threshold for drift detection.
    # The ICP credibility score is max(p1, p0), where p1 and p0 are the
    # empirical p-values under the malware and goodware calibration sets.
    # A sample is flagged as drifting when BOTH p1 < alpha AND p0 < alpha.
    # With SGDClassifier decision_function scores, the score ranges for
    # p1 < alpha and p0 < alpha do not overlap at alpha=0.05 (p95 of ncm_1
    # exceeds p95 of ncm_0 in opposite directions), making detection
    # geometrically impossible. alpha=0.10 is the minimum value at which
    # the two credibility conditions can simultaneously be satisfied.
    # Note: this alpha controls conformal prediction significance, NOT a
    # KS test significance level; the two are conceptually distinct.
    transcend_alpha: float = 0.10

    # misc
    date_field: str = "dex_date"
    random_state: int = 42


VALIDATION_FIXED_PARAM_NAMES = (
    "hash_dim",
    "svd_components",
    "n_clusters_init",
    "cluster_ttl",
    "perf_ema_alpha",
    "burst_threshold",
    "drift_weights",
    "max_update_frac",
    "drift_budget_scale",
    "cluster_priority_new",
    "cluster_priority_drift",
    "cluster_priority_stable",
    "xai_top_k",
    "classifier_alpha",
)


VALIDATION_SELECTED_PARAM_NAMES = (
    "ks_min_effect",
    "combined_threshold",
    "drift_lambda",
    "new_cluster_dist",
)


VALIDATION_TUNED_PARAM_NAMES = VALIDATION_FIXED_PARAM_NAMES + VALIDATION_SELECTED_PARAM_NAMES


def _normalise_tuned_value(name: str, value: Any) -> Any:
    if name == "drift_weights" and isinstance(value, list):
        return tuple(float(v) for v in value)
    return value


def load_validation_tuned_params(
    cfg: Config,
    granularity: str = "monthly",
    verbose: bool = True,
) -> Tuple[Config, Optional[str]]:
    """Load validation-selected Proposed parameters into cfg."""
    granularity = str(granularity).lower()
    candidate_paths = [
        os.path.join(
            cfg.out_dir, "hyperparameters",
            f"{granularity}_validation_tuned_params.json",
        ),
        os.path.join(
            "artifact", "results", "full_csv", "hyperparameters",
            f"{granularity}_validation_tuned_params.json",
        ),
        os.path.join(cfg.out_dir, "proposed_performance", "00_validation_tuned_params.json"),
        os.path.join(
            "artifact", "results", "full_csv", "proposed_performance",
            "00_validation_tuned_params.json",
        ),
        os.path.join(cfg.out_dir, "tuned_params.json"),
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        for name in VALIDATION_TUNED_PARAM_NAMES:
            if name in tuned and hasattr(cfg, name):
                setattr(cfg, name, _normalise_tuned_value(name, tuned[name]))
        if verbose:
            shown = ", ".join(f"{name}={getattr(cfg, name)}" for name in VALIDATION_TUNED_PARAM_NAMES)
            print(f"  [{granularity} validation-tuned params loaded from {path}: {shown}]")
        return cfg, path
    if verbose:
        print(f"  [WARN] No {granularity} validation-tuned parameter file found; using Config defaults.")
    return cfg, None


# =========================================================
# [Data Loading — monthly extended-features]
# =========================================================
def load_dataset(cfg: Config) -> Tuple[List[Dict], np.ndarray, List[Optional[str]]]:
    x_path = os.path.join(cfg.data_dir, "extended-features-X.json")
    y_path = os.path.join(cfg.data_dir, "extended-features-y.json")
    m_path = os.path.join(cfg.data_dir, "extended-features-meta.json")

    for p in (x_path, y_path, m_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                f"Place JSON files under '{cfg.data_dir}'."
            )
    with open(x_path, "r", encoding="utf-8") as f:
        rows: List[Dict] = json.load(f)
    with open(y_path, "r", encoding="utf-8") as f:
        y: np.ndarray = np.asarray(json.load(f), dtype=np.int64)
    with open(m_path, "r", encoding="utf-8") as f:
        meta: List[Dict] = json.load(f)

    if not (len(rows) == len(y) == len(meta)):
        raise ValueError(f"Size mismatch: X={len(rows)}, y={len(y)}, meta={len(meta)}")
    if not set(y.tolist()).issubset({0, 1}):
        raise ValueError(f"Labels must be binary. Found: {set(y.tolist())}")

    months: List[Optional[str]] = []
    for m in meta:
        d = str(m.get(cfg.date_field, "")).strip()
        months.append(d[:7] if len(d) >= 7 else None)

    n_miss = sum(1 for m in months if m is None)
    if n_miss:
        print(f"  [Warning] {n_miss}/{len(months)} samples lack date → excluded.")

    print(f"  [Dataset] {len(y):,} samples | malicious rate: {float(y.mean()):.3f}")
    return rows, y, months


def load_dataset_imbalanced(cfg: Config, gw_mw_ratio: float = 9.0):
    """
    Extended-Features 월별 데이터를 현실적 class imbalance로 재샘플링.
    goodware 전체 유지, malware를 n_gw/ratio 개로 다운샘플링.
    원본 인덱스 순서(시간 순) 유지 → temporal split에 안전.

    Args:
        gw_mw_ratio: goodware:malware 비율 (default 9.0 → ~10% malware)
    Returns: rows, y, months (load_dataset 동일 형식)
    """
    rows_all, y_all, months_all = load_dataset(cfg)
    y_all = np.array(y_all)

    gw_idx = np.where(y_all == 0)[0]
    mw_idx = np.where(y_all == 1)[0]

    n_mw_target = max(1, int(len(gw_idx) / gw_mw_ratio))
    rng = np.random.default_rng(cfg.random_state)
    mw_sampled  = rng.choice(mw_idx, size=min(n_mw_target, len(mw_idx)), replace=False)
    keep = np.sort(np.concatenate([gw_idx, mw_sampled]))

    rows_out   = [rows_all[i]   for i in keep]
    y_out      = y_all[keep]
    months_out = [months_all[i] for i in keep]

    print(f"  [Dataset-Imbalanced] GW: {len(gw_idx):,}  "
          f"MW: {len(mw_sampled):,}  "
          f"Ratio={gw_mw_ratio:.0f}:1  "
          f"mal rate: {y_out.mean():.3f}")
    return rows_out, y_out, months_out


# =========================================================
# [Data Loading — AndroZoo-Year]
# =========================================================
def load_androzoo_year(cfg: Config):
    """
    AndroZoo-Year/final_full_data.csv 로드.

    Returns:
        rows: List[Dict] — feature hashing용 (key→value)
        y: np.ndarray — labels
        years: List[Optional[int]] — per-sample year
    """
    df = pd.read_csv(cfg.androzoo_year_path)
    print(f"  [AndroZooYear] Columns (first 10): {df.columns[:10].tolist()}")
    print(f"  [AndroZooYear] Shape: {df.shape}")

    year_col = next((c for c in df.columns
                     if c.lower() in ['year', 'dex_year', 'date_year']), None)
    label_col = next((c for c in df.columns
                      if c.lower() in ['label', 'class', 'malware', 'y']), None)

    if year_col is None or label_col is None:
        print(f"  [WARN] Columns: {df.columns.tolist()}")
        print(f"  [WARN] First row: {df.iloc[0].to_dict()}")
        raise ValueError(
            f"Cannot find year/label columns. "
            f"Available: {df.columns.tolist()}")

    # 제외할 식별자 컬럼 (문자열 타입이거나 명시적 식별자)
    exclude_ids = {year_col, label_col, 'sha256', 'hash', 'apkname',
                   'package', 'md5', 'app_name', 'filename', 'name'}
    # 숫자형 컬럼만 feature로 사용 (문자열 컬럼 자동 제외)
    feature_cols = [c for c in df.columns
                    if c not in exclude_ids
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  [AndroZooYear] Feature cols: {len(feature_cols)}")

    y = df[label_col].values.astype(np.int64)
    years = [int(v) if pd.notna(v) else None
             for v in df[year_col].values]

    rows = []
    for _, row in df[feature_cols].iterrows():
        feat_dict = {}
        for col, val in row.items():
            if val != 0 and pd.notna(val):
                try:
                    feat_dict[col] = float(val)
                except (ValueError, TypeError):
                    pass
        rows.append(feat_dict)

    print(f"  [AndroZooYear] {len(y):,} samples | "
          f"mal rate: {y.mean():.3f} | "
          f"years: {sorted(set(yr for yr in years if yr))}")
    return rows, y, years



# =========================================================
# [Data Loading — AndroZoo-Year Imbalanced (9:1 GW:MW)]
# =========================================================
def load_androzoo_year_imbalanced(cfg: Config, gw_mw_ratio: float = 9.0):
    """
    AndroZoo-Year 데이터를 현실적 class imbalance로 재샘플링.

    실제 배포 환경의 malware base rate는 ~5-10% (Arp et al. 2022, P8).
    기존 50:50 균형 데이터셋은 P1/P8 pitfall 위험이 있으므로,
    goodware를 유지하고 malware를 다운샘플링해 gw_mw_ratio:1 비율 구성.

    Args:
        gw_mw_ratio: goodware:malware 비율 (default 9.0 → 10% malware)
    Returns: rows, y, years (load_androzoo_year 동일 형식)
    """
    rows_all, y_all, years_all = load_androzoo_year(cfg)
    y_all = np.array(y_all)

    gw_idx  = np.where(y_all == 0)[0]
    mw_idx  = np.where(y_all == 1)[0]

    # goodware 전체 유지, malware를 n_gw/ratio 개로 다운샘플링
    n_mw_target = max(1, int(len(gw_idx) / gw_mw_ratio))
    rng = np.random.default_rng(cfg.random_state)
    mw_sampled  = rng.choice(mw_idx, size=min(n_mw_target, len(mw_idx)),
                              replace=False)
    keep = np.sort(np.concatenate([gw_idx, mw_sampled]))

    rows_out  = [rows_all[i]  for i in keep]
    y_out     = y_all[keep]
    years_out = [years_all[i] for i in keep]

    print(f"  [AndroZooYear-Imbalanced] GW: {len(gw_idx):,}  "
          f"MW: {len(mw_sampled):,}  "
          f"Ratio={gw_mw_ratio:.0f}:1  "
          f"mal rate: {y_out.mean():.3f}")
    return rows_out, y_out, years_out


# =========================================================
# [Data Loading — DREBIN]
# =========================================================
def load_drebin(cfg: Config):
    """
    Drebin/Malware DREBIN.csv 로드.

    Returns:
        rows: List[Dict]
        families: List[str]
        family_to_idx: Dict[str, List[int]]
    """
    df = pd.read_csv(cfg.drebin_path)
    print(f"  [DREBIN] Columns (first 10): {df.columns[:10].tolist()}")
    print(f"  [DREBIN] Shape: {df.shape}")

    family_col = next((c for c in df.columns
                       if c.lower() in ['family', 'class', 'malware_family',
                                        'label_family']), None)

    if family_col is None:
        print(f"  [WARN] No family column found. Columns: {df.columns.tolist()}")
        families = ['unknown'] * len(df)
    else:
        families = df[family_col].fillna('unknown').tolist()

    exclude_ids = {family_col, 'sha256', 'hash', 'md5', 'apkname',
                   'package', 'app_name', 'filename', 'name'}
    feature_cols = [c for c in df.columns
                    if c not in exclude_ids
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  [DREBIN] Feature cols: {len(feature_cols)}")

    rows = []
    for _, row in df[feature_cols].iterrows():
        feat_dict = {}
        for col, val in row.items():
            if val != 0 and pd.notna(val):
                try:
                    feat_dict[col] = float(val)
                except (ValueError, TypeError):
                    pass
        rows.append(feat_dict)

    family_to_idx: Dict[str, List[int]] = {}
    for i, fam in enumerate(families):
        family_to_idx.setdefault(fam, []).append(i)

    print(f"  [DREBIN] {len(rows):,} samples | "
          f"families: {list(family_to_idx.keys())}")
    return rows, families, family_to_idx


# =========================================================
# [Temporal Split — monthly  (TESSERACT protocol)]
# =========================================================
def make_temporal_split(months: List[Optional[str]], cfg: Config):
    """
    Strict temporal split: train < val < test.
    TESSERACT protocol: Pendlebury et al., USENIX Security 2019.
    """
    month_to_idx: Dict[str, List[int]] = {}
    for i, m in enumerate(months):
        if m is None:
            continue
        month_to_idx.setdefault(m, []).append(i)

    ordered = sorted(month_to_idx.keys())
    train_m, val_m, test_m = [], [], []
    for m in ordered:
        yr = int(m.split("-")[0])
        if yr == cfg.train_year:
            train_m.append(m)
        elif yr == cfg.val_year:
            val_m.append(m)
        elif yr > cfg.val_year:
            test_m.append(m)

    for name, lst in [("train", train_m), ("val", val_m), ("test", test_m)]:
        if not lst:
            raise ValueError(f"No {name} data. Check train_year/val_year.")

    train_idx = [i for m in train_m for i in month_to_idx[m]]
    val_idx   = [i for m in val_m   for i in month_to_idx[m]]
    n_test    = sum(len(month_to_idx[m]) for m in test_m)

    print(f"  [Split] Train: {train_m[0]} – {train_m[-1]}  ({len(train_idx):,})")
    print(f"  [Split] Val  : {val_m[0]} – {val_m[-1]}  ({len(val_idx):,})")
    print(f"  [Split] Test : {test_m[0]} – {test_m[-1]}  ({n_test:,})")

    return month_to_idx, ordered, train_m, val_m, test_m, train_idx, val_idx


# =========================================================
# [Temporal Split — yearly  (AndroZoo-Year)]
# =========================================================
def make_yearly_split(years: List[Optional[int]], cfg: Config):
    """
    연도별 temporal split (월별 granularity 없는 데이터용).

    Train: train_year ~ val_year
    Val:   val_year+1
    Test:  val_year+2 ~ max_year

    Returns:
        y2idx: {year: [sample_indices]}
        train_idx, val_idx: List[int]
        test_years: sorted list of test year ints
    """
    y2idx: Dict[int, List[int]] = {}
    for i, yr in enumerate(years):
        if yr is not None:
            y2idx.setdefault(int(yr), []).append(i)

    ordered_years = sorted(y2idx.keys())

    train_idx = [i for yr in ordered_years
                 if yr <= cfg.val_year
                 for i in y2idx[yr]]
    val_idx   = [i for yr in ordered_years
                 if yr == cfg.val_year + 1
                 for i in y2idx[yr]]
    test_years = sorted(yr for yr in ordered_years
                        if yr > cfg.val_year + 1)

    print(f"  [YearlySplit] Train years: "
          f"{[y for y in ordered_years if y <= cfg.val_year]}")
    print(f"  [YearlySplit] Val year: {cfg.val_year + 1} "
          f"({len(val_idx):,} samples)")
    print(f"  [YearlySplit] Test years: {test_years}")
    return y2idx, train_idx, val_idx, test_years


# =========================================================
# [Feature Hashing + Feature Index]
# =========================================================
def _hash_bucket_sign(name: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.sha1(str(name).encode("utf-8", errors="ignore")).digest()
    bucket = int.from_bytes(digest[:8], "little") % dim
    sign   = 1.0 if (digest[8] & 1) == 0 else -1.0
    return bucket, sign


def build_hashed_csr(rows: List[Dict], indices: List[int], dim: int) -> csr_matrix:
    """
    Signed feature hashing to reduce collision bias (P4).
    Weinberger et al., ICML 2009.
    """
    indptr, idx_arr, data_arr = [0], [], []
    for orig_i in indices:
        col_val: Dict[int, float] = {}
        for k in rows[orig_i].keys():
            bucket, sign = _hash_bucket_sign(str(k), dim)
            col_val[bucket] = col_val.get(bucket, 0.0) + sign
        sc = sorted(col_val)
        idx_arr.extend(sc)
        data_arr.extend(col_val[c] for c in sc)
        indptr.append(len(idx_arr))
    return csr_matrix((data_arr, idx_arr, indptr),
                      shape=(len(indices), dim), dtype=np.float32)


def build_feature_index(rows: List[Dict], indices: List[int],
                        dim: int) -> Dict[int, str]:
    """
    Observed candidate map for XAI: hash_bucket → representative feature_name.
    Hashing is many-to-one, so this is NOT an inverse mapping.  For each
    bucket, we aggregate feature names observed in the allowed split and keep
    the most frequent candidate (lexicographic tie-break).  ClusterXAI can
    further refine this with cluster-member candidates when raw rows are
    provided.

    NOTE: build from training indices only in paper experiments to prevent
    temporal snooping (P3).
    """
    bucket_counts: Dict[int, Counter] = {}
    for orig_i in indices:
        for k in rows[orig_i].keys():
            name   = str(k)
            bucket, _ = _hash_bucket_sign(name, dim)
            bucket_counts.setdefault(bucket, Counter())[name] += 1

    bucket_to_name: Dict[int, str] = {}
    for bucket, counts in bucket_counts.items():
        max_count = max(counts.values())
        candidates = [name for name, cnt in counts.items() if cnt == max_count]
        bucket_to_name[bucket] = min(candidates)
    return bucket_to_name


# =========================================================
# [Semantic Embedder]
# =========================================================
class SemanticEmbedder:
    """
    TruncatedSVD (LSA-style) → L2-normalised dense cosine space.
    Fitted on training set ONLY to prevent temporal snooping (P3).
    Ref: Deerwester et al., JASIS 41(6) 1990.
    """

    def __init__(self, n_components: int, random_state: int):
        self.svd = TruncatedSVD(n_components=n_components,
                                random_state=random_state)
        self.n_components = n_components

    def fit(self, X: csr_matrix) -> "SemanticEmbedder":
        self.svd.fit(X)
        return self

    def transform(self, X: csr_matrix) -> np.ndarray:
        return normalize(self.svd.transform(X), norm="l2")

    def top_feature_buckets_for_vector(
        self, z: np.ndarray, top_k: int = 10, candidate_multiplier: int = 5
    ) -> List[Tuple[int, float]]:
        """Return high-magnitude hash buckets for an embedding-space direction."""
        v = self.svd.components_.T @ z
        top_buckets = np.argsort(np.abs(v))[::-1][:top_k * candidate_multiplier]
        return [(int(b), float(v[b])) for b in top_buckets]

    def top_features_for_vector(
        self, z: np.ndarray, bucket_to_name: Dict[int, str], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        XAI (C2): back-project embedding vector z to original feature space.
        Hash buckets are not invertible; the returned feature name is the
        representative observed candidate for that bucket.
        """
        results, seen = [], set()
        for b, bucket_score in self.top_feature_buckets_for_vector(z, top_k):
            name = bucket_to_name.get(int(b), f"<bucket_{b}>")
            if name in seen:
                continue
            seen.add(name)
            if name.startswith("<bucket_"):
                signed_score = bucket_score
            else:
                _, sign = _hash_bucket_sign(name, self.svd.components_.shape[1])
                signed_score = bucket_score * sign
            results.append((name, float(signed_score)))
            if len(results) >= top_k:
                break
        return results


# =========================================================
# [ClusterXAI  (C2)]
# =========================================================
class ClusterXAI:
    """
    Cluster feature attribution wrapper.
    Uses SVD back-projection via SemanticEmbedder for semantic-direction attribution.
    Because feature hashing is many-to-one, top features are estimated from
    observed feature candidates mapped to high-scoring buckets, preferably
    within current cluster members.
    embedder must be set via set_embedder() before get_top_features() is called.
    """

    def __init__(self, global_mean: np.ndarray, b2n: Dict[int, str],
                 top_k: int = 10, min_delta: float = 0.01):
        self.global_mean = global_mean  # shape (hash_dim,), original feature space
        self.b2n         = b2n
        self.top_k       = top_k
        self.min_delta   = min_delta
        self.hash_dim    = len(global_mean)
        self._embedder: Optional[SemanticEmbedder] = None

    def set_embedder(self, embedder: SemanticEmbedder) -> None:
        self._embedder = embedder

    def _member_candidates(
        self, buckets: List[int], member_rows: Optional[List[Dict]]
    ) -> Dict[int, Tuple[str, float]]:
        if not member_rows:
            return {}
        wanted = set(int(b) for b in buckets)
        counts_by_bucket: Dict[int, Counter] = {int(b): Counter() for b in wanted}
        signs_by_bucket: Dict[int, Dict[str, float]] = {int(b): {} for b in wanted}
        for row in member_rows:
            for feature_name in row.keys():
                name = str(feature_name)
                b, sign = _hash_bucket_sign(name, self.hash_dim)
                if b not in wanted:
                    continue
                counts_by_bucket[b][name] += 1
                signs_by_bucket[b][name] = sign

        result: Dict[int, Tuple[str, float]] = {}
        for bucket, counts in counts_by_bucket.items():
            if not counts:
                continue
            max_count = max(counts.values())
            candidates = [name for name, cnt in counts.items() if cnt == max_count]
            name = min(candidates)
            result[bucket] = (name, signs_by_bucket[bucket].get(name, 1.0))
        return result

    def _fallback_candidate(self, bucket: int) -> Tuple[str, float]:
        name = self.b2n.get(int(bucket), f"<bucket_{bucket}>")
        if name.startswith("<bucket_"):
            return name, 1.0
        _, sign = _hash_bucket_sign(name, self.hash_dim)
        return name, sign

    def get_top_features(
        self, centroid_z: np.ndarray, member_rows: Optional[List[Dict]] = None
    ) -> List[Tuple[str, float]]:
        """Estimate top feature candidates for an embedding-space centroid."""
        if self._embedder is not None:
            results, seen = [], set()
            bucket_scores = self._embedder.top_feature_buckets_for_vector(
                centroid_z, self.top_k)
            member_candidates = self._member_candidates(
                [bucket for bucket, _ in bucket_scores], member_rows)
            for bucket, bucket_score in bucket_scores:
                cand = member_candidates.get(bucket, self._fallback_candidate(bucket))
                name, sign = cand
                if name in seen:
                    continue
                seen.add(name)
                results.append((name, float(bucket_score * sign)))
                if len(results) >= self.top_k:
                    break
            return results
        # Fallback if embedder not set
        return []


# =========================================================
# [ClusterManager  (C1 + C2 + C3)]
# =========================================================
class ClusterManager:
    """
    Online semantic cluster registry with birth/death lifecycle, XAI logging,
    and novelty-weighted sample selection.

    C1 — Semantic clusters in SVD cosine space (MiniBatchKMeans)
    C2 — Per-cluster feature attribution via ClusterXAI
    C3 — Budget-constrained selection prioritising novel/drifting clusters
    """

    def __init__(self, cfg: Config, xai: ClusterXAI):
        self.cfg  = cfg
        self.xai  = xai

        self.centroids:    List[np.ndarray]               = []
        self.label_dist:   List[np.ndarray]               = []
        self.sample_count: List[float]                    = []
        self.last_seen:    List[int]                      = []
        self.birth_month:  List[int]                      = []
        self.age:          List[int]                      = []
        self.is_new:       List[bool]                     = []
        self.top_features: List[List[Tuple[str, float]]]  = []

    # ── Internal helpers ───────────────────────────────
    def _register_one(self, centroid: np.ndarray, ld: np.ndarray,
                      month_idx: int, is_new: bool = True,
                      sample_count: Optional[float] = None,
                      member_rows: Optional[List[Dict]] = None) -> int:
        c = normalize(centroid.reshape(1, -1), norm="l2")[0]
        f = self.xai.get_top_features(c, member_rows=member_rows)
        self.centroids.append(c)
        self.label_dist.append(ld.copy())
        self.sample_count.append(float(ld.sum() if sample_count is None else sample_count))
        self.last_seen.append(month_idx)
        self.birth_month.append(month_idx)
        self.age.append(0)
        self.is_new.append(is_new)
        self.top_features.append(f)
        return len(self.centroids) - 1

    # ── Bootstrap ──────────────────────────────────────
    def fit_initial(self, Z_train: np.ndarray,
                    X_train_or_y,
                    y_train_opt=None) -> None:
        """
        Bootstrap k clusters from training set.
        Accepts both fit_initial(Z, y) and fit_initial(Z, X, y) signatures.
        """
        # Handle both call signatures
        if y_train_opt is None:
            y_train = X_train_or_y  # old style: fit_initial(Z, y)
        else:
            y_train = y_train_opt   # new style: fit_initial(Z, X, y)

        k  = max(1, min(self.cfg.n_clusters_init, len(Z_train) // 10))
        km = MiniBatchKMeans(n_clusters=k, random_state=self.cfg.random_state,
                             n_init=3, batch_size=2048)
        labels = km.fit_predict(Z_train)
        for c in range(k):
            mask = labels == c
            ld   = np.array([float((y_train[mask] == 0).sum()),
                             float((y_train[mask] == 1).sum())])
            self._register_one(km.cluster_centers_[c], ld,
                               month_idx=0, is_new=False,
                               sample_count=float(mask.sum()))
        print(f"  [ClusterMgr] Bootstrap: {k} clusters from training set")

    # ── Monthly assignment ─────────────────────────────
    def assign(
        self, Z: np.ndarray, t_idx: int,
        member_rows: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Returns (assignment array, n_new_clusters_born_this_month).
        This method is label-free: monthly labels must not be provided here.
        """
        n          = len(Z)
        assignment = np.full(n, -1, dtype=np.int32)

        if not self.centroids:
            ld = np.array([0.0, 0.0])
            self._register_one(
                Z.mean(axis=0), ld, t_idx,
                sample_count=float(n), member_rows=member_rows)
            assignment[:] = 0
            return assignment, 1

        C        = np.stack(self.centroids)
        sims     = Z @ C.T
        best_cid = sims.argmax(axis=1)
        best_sim = sims[np.arange(n), best_cid]
        new_mask = best_sim < (1.0 - self.cfg.new_cluster_dist)
        assignment[~new_mask] = best_cid[~new_mask]

        for cid in range(len(self.centroids)):
            if self.is_new[cid] and self.last_seen[cid] < t_idx:
                self.is_new[cid] = False
            mask = (~new_mask) & (best_cid == cid)
            if not mask.any():
                continue
            n_c   = mask.sum()
            total = self.sample_count[cid] + n_c
            alpha = n_c / total
            new_c = (1 - alpha) * self.centroids[cid] + alpha * Z[mask].mean(axis=0)
            self.centroids[cid]    = normalize(new_c.reshape(1, -1), norm="l2")[0]
            rows_c = [member_rows[i] for i in np.where(mask)[0]] if member_rows is not None else None
            self.top_features[cid] = self.xai.get_top_features(
                self.centroids[cid], member_rows=rows_c)
            self.sample_count[cid] = float(total)
            self.last_seen[cid]  = t_idx
            self.age[cid]       += 1

        K_before = len(self.centroids)
        if new_mask.sum() > 0:
            rows_new = [member_rows[i] for i in np.where(new_mask)[0]] if member_rows is not None else None
            self._register_new_clusters(Z[new_mask], t_idx, member_rows=rows_new)
            new_ids = list(range(K_before, len(self.centroids)))
            if new_ids:
                C_new    = np.stack([self.centroids[i] for i in new_ids])
                sims_new = Z[new_mask] @ C_new.T
                lc       = sims_new.argmax(axis=1)
                assignment[new_mask] = np.array(new_ids)[lc]

        n_new = len(self.centroids) - K_before
        return assignment, n_new

    def assign_and_update(self, Zt: np.ndarray, Xt: csr_matrix,
                          t_idx: int,
                          member_rows: Optional[List[Dict]] = None) -> Tuple[np.ndarray, int]:
        """Combined assign + prune convenience method (used in evidence inspection)."""
        assignment, n_new = self.assign(Zt, t_idx, member_rows=member_rows)
        self.prune(t_idx)
        return assignment, n_new

    def _register_new_clusters(
        self, Z_sub: np.ndarray, month_idx: int,
        member_rows: Optional[List[Dict]] = None,
    ) -> None:
        n_sub = len(Z_sub)
        k_new = max(1, min(n_sub // 20, 5))
        if k_new == 1 or n_sub < k_new * 2:
            ld = np.array([0.0, 0.0])
            self._register_one(
                Z_sub.mean(axis=0), ld, month_idx,
                sample_count=float(n_sub), member_rows=member_rows)
        else:
            km  = MiniBatchKMeans(n_clusters=k_new,
                                  random_state=self.cfg.random_state,
                                  n_init=3, batch_size=min(1024, n_sub))
            lbs = km.fit_predict(Z_sub)
            for c in range(k_new):
                mc = lbs == c
                if not mc.any():
                    continue
                ld = np.array([0.0, 0.0])
                rows_c = [member_rows[i] for i in np.where(mc)[0]] if member_rows is not None else None
                self._register_one(
                    km.cluster_centers_[c], ld, month_idx,
                    sample_count=float(mc.sum()), member_rows=rows_c)

    # ── Prune ──────────────────────────────────────────
    def prune(self, month_idx: int) -> int:
        """Remove clusters unseen for > cluster_ttl months."""
        keep     = [i for i in range(len(self.centroids))
                    if (month_idx - self.last_seen[i]) <= self.cfg.cluster_ttl]
        n_pruned = len(self.centroids) - len(keep)
        if n_pruned > 0:
            for attr in ["centroids", "label_dist", "last_seen",
                         "sample_count", "birth_month", "age", "is_new", "top_features"]:
                setattr(self, attr, [getattr(self, attr)[i] for i in keep])
        return n_pruned

    def update_label_distribution(
        self, assigned_cluster_ids: np.ndarray, y_labeled: np.ndarray
    ) -> None:
        """Update cluster label statistics using only oracle-labeled samples."""
        if len(y_labeled) == 0:
            return
        for cid, label in zip(assigned_cluster_ids, y_labeled):
            cid = int(cid)
            if cid < 0 or cid >= len(self.label_dist):
                continue
            self.label_dist[cid][int(label)] += 1.0

    # ── C3: Novelty-weighted sample selection ──────────
    def select_update_samples(
        self, X_month: csr_matrix, y_month: np.ndarray,
        assignment: np.ndarray, drift_flag: bool,
        max_frac: float, rng: np.random.Generator,
        exact_budget: Optional[int] = None,
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray, Dict]:
        """
        Budget-constrained sample selection (C3).
        Allocates labelling budget proportional to cluster novelty weight × cluster size.
        """
        n      = len(y_month)
        if exact_budget is None:
            budget = max(1, int(n * max_frac))
        else:
            budget = min(max(0, int(exact_budget)), n)
        K      = len(self.centroids)

        if K == 0 or n == 0 or budget == 0:
            empty = csr_matrix((0, X_month.shape[1]), dtype=np.float32)
            return empty, np.array([], dtype=np.int64), np.array([], dtype=np.float64), {}

        weights = np.array(
            [self.cfg.cluster_priority_new if self.is_new[i]
             else (self.cfg.cluster_priority_drift if drift_flag
                   else self.cfg.cluster_priority_stable)
             for i in range(K)], dtype=np.float64
        )

        cluster_counts = np.zeros(K, dtype=np.int64)
        for cid in assignment[assignment >= 0]:
            if cid < K:
                cluster_counts[cid] += 1

        scores      = weights * cluster_counts.astype(float)
        total_score = scores.sum()
        selected:   List[int] = []
        per_cl:     Dict[int, int] = {}

        if total_score > 0:
            for i in range(K):
                n_i = (assignment == i).sum()
                if n_i == 0:
                    continue
                alloc = min(int(np.round(budget * scores[i] / total_score)), n_i)
                if alloc <= 0:
                    continue
                pool   = np.where(assignment == i)[0]
                chosen = rng.choice(pool, size=alloc, replace=False)
                selected.extend(chosen.tolist())
                per_cl[i] = alloc
        else:
            chosen   = rng.choice(n, size=min(budget, n), replace=False)
            selected = chosen.tolist()

        if exact_budget is not None:
            target = min(budget, n)
            if len(selected) > target:
                selected = rng.choice(
                    np.array(selected, dtype=np.int64), size=target, replace=False
                ).tolist()
            elif len(selected) < target:
                selected_mask = np.zeros(n, dtype=bool)
                selected_mask[np.array(selected, dtype=np.int64)] = True
                remaining = np.where(~selected_mask)[0]
                n_fill = min(target - len(selected), len(remaining))
                if n_fill > 0:
                    filler = rng.choice(remaining, size=n_fill, replace=False)
                    selected.extend(filler.tolist())

        if not selected:
            per_cl["__selected_idx"] = []
            empty = csr_matrix((0, X_month.shape[1]), dtype=np.float32)
            return empty, np.array([], dtype=np.int64), np.array([], dtype=np.float64), per_cl

        per_cl["__selected_idx"] = list(selected)
        sel_idx = np.array(selected, dtype=np.int64)
        y_sel   = y_month[sel_idx]
        self.update_label_distribution(assignment[sel_idx], y_sel)
        sel_w   = np.array(
            [weights[int(assignment[i])] if 0 <= int(assignment[i]) < K else 1.0
             for i in sel_idx], dtype=np.float64
        )
        return X_month[sel_idx], y_sel, sel_w, per_cl

    def export_xai_snapshot(self, month_key: str) -> List[Dict]:
        result = []
        for i in range(len(self.centroids)):
            ld    = self.label_dist[i]
            labeled_total = ld.sum()
            total = self.sample_count[i]
            result.append({
                "month"       : month_key,
                "cluster_id"  : i,
                "n_samples"   : int(total),
                "n_benign"    : int(ld[0]),
                "n_malicious" : int(ld[1]),
                "n_labeled"   : int(labeled_total),
                "mal_ratio"   : round(ld[1] / labeled_total, 4) if labeled_total > 0 else 0.0,
                "age"         : self.age[i],
                "birth_month" : self.birth_month[i],
                "is_new"      : self.is_new[i],
                "top_features": "; ".join(f"{nm}({s:+.3f})"
                                          for nm, s in self.top_features[i]),
            })
        return result

    def summary(self) -> Dict:
        return {"n_clusters": len(self.centroids),
                "n_new": sum(self.is_new)}


# =========================================================
# [Drift Monitor — 3-signal composite]
# =========================================================
class DriftMonitor:
    """
    3-Signal Semantic Drift Detector with 1-month lag.

    Signal 1 — KS test + effect size gate   (distribution shift)
    Signal 2 — EMA confidence-score shift   (label-free proxy)
    Signal 3 — Cluster novelty burst        (semantic novelty)
    """

    def __init__(
        self,
        energy_train:        np.ndarray,
        ks_alpha:            float = 0.05,
        ks_min_effect:       float = 0.10,
        perf_ema_alpha:      float = 0.30,
        burst_threshold:     float = 3.0,
        combined_threshold:  float = 0.25,
        weights:             tuple = (0.40, 0.30, 0.30),
        emergency_threshold: float = 0.50,
    ):
        self._ref        = energy_train.copy()
        self._mu0        = float(np.mean(energy_train))
        self._sigma0     = max(float(np.std(energy_train)), 1e-6)

        self._alpha      = ks_alpha
        self._min_effect = ks_min_effect
        self._ema_alpha  = perf_ema_alpha
        self._mu_ema     = self._mu0

        self._burst_thr  = burst_threshold
        self._n_new      = 0

        self._threshold  = combined_threshold
        self._w          = tuple(weights)
        # combined > emergency_threshold 이면 1-month lag 없이 즉시 대응
        self._emergency_thr = emergency_threshold

        self._pending    = False

    def notify_births(self, n_new_clusters: int) -> None:
        """Call AFTER ClusterManager.assign() for month t."""
        self._n_new = n_new_clusters

    def update(self, energy_month: np.ndarray) -> Dict:
        """Call BEFORE model update. Returns drift info dict."""
        curr = float(np.mean(energy_month))

        # Signal 1: KS + effect size
        ks_stat, ks_p = scipy_stats.ks_2samp(self._ref, energy_month)
        ks_stat       = float(ks_stat)
        ks_alarm      = bool(ks_p < self._alpha and ks_stat > self._min_effect)
        ks_score      = float(np.clip(
            (ks_stat - self._min_effect) / max(1.0 - self._min_effect, 1e-6),
            0.0, 1.0)) if ks_alarm else 0.0

        # Signal 2: EMA confidence-score shift (label-free proxy, not F1/accuracy)
        self._mu_ema = (1 - self._ema_alpha) * self._mu_ema + self._ema_alpha * curr
        confidence_shift_score = float(np.clip(
            (self._mu0 - self._mu_ema) / self._sigma0, 0.0, 1.0))
        confidence_shift_alarm = bool(confidence_shift_score > 0.20)

        # Signal 3: cluster novelty burst
        novelty_score = float(np.clip(self._n_new / self._burst_thr, 0.0, 1.0))
        novelty_alarm = bool(self._n_new >= self._burst_thr)

        # Combined weighted score
        w1, w2, w3 = self._w
        combined  = w1 * ks_score + w2 * confidence_shift_score + w3 * novelty_score
        is_drift  = bool(combined > self._threshold)

        # Emergency: combined이 매우 높으면 lag 없이 즉시 대응
        emergency = bool(combined > self._emergency_thr)

        # 1-month lag + emergency bypass
        # normal path: 이번 달 drift는 다음 달에 적용 (_pending)
        # emergency path: 이번 달 바로 적용 (lag 우회)
        flag_now      = self._pending or emergency
        self._pending = is_drift
        self._n_new   = 0

        return {
            "drift_score"       : round(combined, 4),
            "ks_stat"           : round(ks_stat, 4),
            "ks_score"          : round(ks_score, 4),
            "ks_pvalue"         : round(float(ks_p), 6),
            "confidence_shift_score": round(confidence_shift_score, 4),
            "novelty_score"     : round(novelty_score, 4),
            "ks_alarm"          : ks_alarm,
            "confidence_shift_alarm": confidence_shift_alarm,
            "novelty_alarm"     : novelty_alarm,
            "drift_detected"    : is_drift,
            "emergency_flag"    : emergency,
            "drift_flag_applied": flag_now,
            "mu_ema"            : round(self._mu_ema, 4),
        }


# =========================================================
# [Loss Functions  (C4)]
# =========================================================
def _apply_update(model: SGDClassifier, X_sel: csr_matrix,
                  y_sel: np.ndarray, sel_w: np.ndarray,
                  drift_score: float, cfg: Config) -> None:
    """
    Incremental update with configurable loss mode.

    "standard"  : uniform weights
    "drift_reg" : PROPOSED C4 — w̃ᵢ = wᵢ·(1 + λ·drift_score)
    """
    if X_sel.shape[0] == 0:
        return
    if cfg.loss_mode == "standard":
        model.partial_fit(X_sel, y_sel)
    elif cfg.loss_mode == "weighted":
        w = sel_w / sel_w.mean()
        model.partial_fit(X_sel, y_sel, sample_weight=w)
    elif cfg.loss_mode == "drift_reg":
        # C4: w̃ᵢ = (wᵢ / mean(w)) · amp
        # 1단계: 상대적 클러스터 가중치 보존 (C3와 연동)
        # 2단계: amp로 전체 학습 스케일 증폭 → C4 단독으로도 효과 발생
        # (기존: w_amp / w_amp.mean() → amp가 정규화로 상쇄되는 버그 수정)
        amp   = 1.0 + cfg.drift_lambda * float(drift_score)
        w_rel = sel_w / max(sel_w.mean(), 1e-9)  # 상대적 가중치 (mean=1.0)
        w     = w_rel * amp                       # drift 심각도로 전체 스케일 증폭
        model.partial_fit(X_sel, y_sel, sample_weight=w)
    else:
        raise ValueError(f"Unknown loss_mode: {cfg.loss_mode!r}")


def _fresh_model(cfg: Config) -> SGDClassifier:
    return SGDClassifier(loss="log_loss", alpha=cfg.classifier_alpha,
                         random_state=cfg.random_state, warm_start=True)


# =========================================================
# [Metrics  (bugfixed)]
# =========================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict:
    """
    Metrics per Arp et al. (Dos & Don'ts, USENIX Security 2022).
    FPR@TPR=90% computed with linear interpolation (bugfix over nearest-index).
    F1: macro average.
    """
    nan = float("nan")
    if len(np.unique(y_true)) < 2:
        return dict(f1=nan, tpr=nan, fpr=nan,
                    fpr_at_tpr90=nan, auc_roc=nan, auc_pr=nan)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FPR@90%TPR: linear interpolation
    fprs_c, tprs_c, _ = roc_curve(y_true, y_prob)
    fpr90 = float("nan")
    if tprs_c[-1] >= 0.90:
        idx = np.searchsorted(tprs_c, 0.90)
        idx = min(idx, len(tprs_c) - 1)
        if idx == 0:
            fpr90 = float(fprs_c[0])
        else:
            t0, t1 = tprs_c[idx - 1], tprs_c[idx]
            f0, f1 = fprs_c[idx - 1], fprs_c[idx]
            if abs(t1 - t0) < 1e-10:
                fpr90 = float(f0)
            else:
                fpr90 = float(
                    f0 + (f1 - f0) * (0.90 - t0) / (t1 - t0))

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return dict(
        f1           = float(f1_score(y_true, y_pred, average="macro",
                                      zero_division=0)),
        tpr          = float(tpr),
        fpr          = float(fpr),
        fpr_at_tpr90 = fpr90,
        auc_roc      = float(roc_auc_score(y_true, y_prob)),
        auc_pr       = float(auc(rec, prec)),
    )


def compute_label_efficiency(y_sel: np.ndarray) -> float:
    """Malicious rate among selected samples (label efficiency indicator)."""
    if len(y_sel) == 0:
        return float("nan")
    return float(np.mean(y_sel == 1))


def compute_loss_profile(model: SGDClassifier, X: csr_matrix,
                         y: np.ndarray) -> Dict:
    prob = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    per  = -(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    mal, ben = y == 1, y == 0
    return {
        "loss_mean": float(np.mean(per)),
        "loss_mal" : float(np.mean(per[mal])) if mal.any() else float("nan"),
        "loss_ben" : float(np.mean(per[ben])) if ben.any() else float("nan"),
        "loss_std" : float(np.std(per)),
    }


# =========================================================
# Required columns in every result CSV
# =========================================================
REQUIRED_COLS = [
    'month', 'method', 'total', 'updated',
    'f1', 'tpr', 'fpr', 'fpr_at_tpr90', 'auc_roc', 'auc_pr',
    'drift_score', 'ks_stat', 'n_clusters',
    'mal_rate_selected', 'loss_before', 'loss_after',
]


def to_result_df(results: List[Dict]) -> pd.DataFrame:
    """Normalize result rows to have all required columns (NaN for missing)."""
    df = pd.DataFrame(results)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = float('nan')
    return df


# =========================================================
# [Comparison Baselines]
# =========================================================

# ── Static ────────────────────────────────────────────────
def run_baseline_static(X_train: csr_matrix, y_train: np.ndarray,
                        month_to_idx: Dict, test_months: List[str],
                        rows: List[Dict], y: np.ndarray,
                        cfg: Config) -> List[Dict]:
    """No update after initial training."""
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    results = []
    for mk in test_months:
        idx = month_to_idx.get(mk, [])
        if not idx:
            continue
        t0 = time.perf_counter()
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        m  = compute_metrics(yt, model.predict(Xt),
                             model.predict_proba(Xt)[:, 1])
        m.update({"month": mk, "method": "Static",
                  "updated": 0, "total": len(yt),
                  "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m)
    return results


# ── FullUpdate ────────────────────────────────────────────
def run_baseline_full_update(X_train: csr_matrix, y_train: np.ndarray,
                             month_to_idx: Dict, test_months: List[str],
                             rows: List[Dict], y: np.ndarray,
                             cfg: Config) -> List[Dict]:
    """All samples, uniform loss — oracle labelling-cost upper bound."""
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    results = []
    for mk in test_months:
        idx = month_to_idx.get(mk, [])
        if not idx:
            continue
        t0 = time.perf_counter()
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        m  = compute_metrics(yt, model.predict(Xt),
                             model.predict_proba(Xt)[:, 1])
        model.partial_fit(Xt, yt)
        m.update({"month": mk, "method": "FullUpdate",
                  "updated": len(yt), "total": len(yt),
                  "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m)
    return results


# ── Retraining ────────────────────────────────────────────
def run_baseline_retraining(all_by_month: Dict, train_months: List[str],
                             test_months: List[str],
                             rows: List[Dict], y: np.ndarray,
                             cfg: Config,
                             rng: np.random.Generator) -> List[Dict]:
    """Periodic full retraining on accumulating sliding window."""
    accumulated: List[int] = []
    for m in train_months:
        accumulated.extend(all_by_month.get(m, []))
    results = []
    for mk in test_months:
        idx = all_by_month.get(mk, [])
        if not idx:
            continue
        n_acc = len(accumulated)
        if n_acc < 10:
            accumulated.extend(idx)
            continue
        t0     = time.perf_counter()
        perm   = rng.permutation(n_acc)
        n_tr   = int(n_acc * 0.6)
        tr_idx = [accumulated[i] for i in perm[:n_tr]]
        X_r    = build_hashed_csr(rows, tr_idx, cfg.hash_dim)
        y_r    = y[tr_idx]
        model  = _fresh_model(cfg)
        model.partial_fit(X_r, y_r, classes=np.array([0, 1]))
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]
        m_ = compute_metrics(yt, model.predict(Xt),
                             model.predict_proba(Xt)[:, 1])
        m_.update({"month": mk, "method": "Retraining",
                   "updated": n_tr, "total": len(yt),
                   "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m_)
        accumulated.extend(idx)
    return results


# ── Transcend + Adapt ─────────────────────────────────────
def run_baseline_transcend(X_train: csr_matrix, y_train: np.ndarray,
                           month_to_idx: Dict, test_months: List[str],
                           rows: List[Dict], y: np.ndarray,
                           cfg: Config,
                           rng: np.random.Generator) -> List[Dict]:
    """
    Transcend-style conformal drift detection + budget-selective adaptation.
    Ref: Jordaney et al., USENIX Security 2017; Barbero et al., IEEE S&P 2022.
    """
    n_train  = X_train.shape[0]
    n_cal    = int(n_train * cfg.transcend_cal_frac)
    n_fit    = n_train - n_cal
    idx_perm = rng.permutation(n_train)
    fit_idx  = idx_perm[:n_fit]
    cal_idx  = idx_perm[n_fit:]

    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    model = _fresh_model(cfg)
    model.partial_fit(X_fit, y_fit, classes=np.array([0, 1]))

    cal_scores = model.decision_function(X_cal)
    ncm_1 = -cal_scores[y_cal == 1]
    ncm_0 =  cal_scores[y_cal == 0]

    results = []
    for mk in test_months:
        idx = month_to_idx.get(mk, [])
        if not idx:
            continue
        t0  = time.perf_counter()
        Xt  = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt  = y[idx]
        n_t = len(yt)

        test_scores = model.decision_function(Xt)
        y_pred = model.predict(Xt)
        y_prob = model.predict_proba(Xt)[:, 1]
        m_ = compute_metrics(yt, y_pred, y_prob)

        ncm_test_1 = -test_scores
        ncm_test_0 =  test_scores
        p1 = np.array([(ncm_1 >= v).mean() for v in ncm_test_1])
        p0 = np.array([(ncm_0 >= v).mean() for v in ncm_test_0])
        credibility = np.maximum(p1, p0)
        drifting_mask = credibility < cfg.transcend_alpha
        n_drifting    = drifting_mask.sum()

        budget = max(1, int(n_t * cfg.max_update_frac))
        if n_drifting > 0:
            drifting_idx = np.where(drifting_mask)[0]
            n_select     = min(budget, n_drifting)
            chosen       = rng.choice(drifting_idx, size=n_select, replace=False)
            X_sel        = Xt[chosen]
            y_sel        = yt[chosen]
            model.partial_fit(X_sel, y_sel)
            n_updated = n_select
        else:
            n_updated = 0

        m_.update({"month": mk, "method": "Transcend+Adapt",
                   "updated": n_updated, "total": n_t,
                   "n_drifting": int(n_drifting),
                   "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m_)

        new_scores = model.decision_function(Xt)
        ncm_1 = np.concatenate([ncm_1, -new_scores[yt == 1]])
        ncm_0 = np.concatenate([ncm_0,  new_scores[yt == 0]])

    return results


# ── ADWIN + Retrain ───────────────────────────────────────
def run_baseline_adwin(X_train: csr_matrix, y_train: np.ndarray,
                       month_to_idx: Dict, test_months: List[str],
                       rows: List[Dict], y: np.ndarray,
                       cfg: Config,
                       rng: np.random.Generator) -> List[Dict]:
    """
    KS-based adaptive windowing drift detection + sliding-window retraining.
    Ref: Bifet & Gavaldà, SIAM SDM 2007; Ceschin et al., Expert Syst. Appl. 2023.
    """
    model = _fresh_model(cfg)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    energy_ref  = model.decision_function(X_train)
    window_data: List[Tuple[csr_matrix, np.ndarray]] = []
    W           = cfg.adwin_window_months

    results = []
    for mk in test_months:
        idx = month_to_idx.get(mk, [])
        if not idx:
            continue
        t0 = time.perf_counter()
        Xt = build_hashed_csr(rows, idx, cfg.hash_dim)
        yt = y[idx]

        y_pred = model.predict(Xt)
        y_prob = model.predict_proba(Xt)[:, 1]
        m_     = compute_metrics(yt, y_pred, y_prob)

        energy_t            = model.decision_function(Xt)
        ks_stat, ks_p       = scipy_stats.ks_2samp(energy_ref, energy_t)
        drift_detected      = bool(ks_p < cfg.ks_alpha and ks_stat > cfg.ks_min_effect)

        window_data.append((Xt, yt))
        if len(window_data) > W:
            window_data.pop(0)

        n_updated = 0
        if drift_detected and len(window_data) >= 1:
            Xs    = [w[0] for w in window_data]
            ys    = [w[1] for w in window_data]
            X_win = csr_matrix(vstack(Xs))
            y_win = np.concatenate(ys)
            model = _fresh_model(cfg)
            model.partial_fit(X_win, y_win, classes=np.array([0, 1]))
            energy_ref = model.decision_function(X_win)
            n_updated  = X_win.shape[0]

        m_.update({"month": mk, "method": "ADWIN+Retrain",
                   "updated": n_updated, "total": len(yt),
                   "drift_detected": drift_detected,
                   "ks_stat": round(float(ks_stat), 4),
                   "time_sec": round(time.perf_counter() - t0, 4)})
        results.append(m_)

    return results


# =========================================================
# [Proposed Model + Ablation  (shared inner loop)]
# =========================================================
def _run_selective_update_core(
    X_train: csr_matrix, y_train: np.ndarray,
    month_to_idx: Dict, test_months: List[str],
    rows: List[Dict], y: np.ndarray,
    embedder: SemanticEmbedder, xai: ClusterXAI,
    cfg: Config, rng: np.random.Generator,
    method_name: str,
    use_cluster_selection: bool,
    use_drift_loss: bool,
    use_drift_score: bool = True,
    selection_strategy: str = "random",
    fixed_update_counts: Optional[Dict[str, int]] = None,
    verbose: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Shared inner loop for all ablation variants and the Proposed model.

    Ablation matrix
    ----------------
    method_name                  cluster   drift_score   drift_loss   selection
    "Random+Standard"            False     False         False        random
    "Cluster+Standard"           True      False         False        cluster
    "Random+DriftLoss"           False     True          True         random
    "Cluster+DriftScore"         True      True          False        cluster
    "UncertaintyOnly"            False     False         False        uncertainty
    "ScoreShiftOnly"             False     False         False        score_shift
    "Proposed"                   True      True          True         cluster
    """
    cfg_run = Config(**{k: v for k, v in cfg.__dict__.items()})
    cfg_run.loss_mode = "drift_reg" if use_drift_loss else "standard"
    if use_cluster_selection:
        selection_strategy = "cluster"

    model = _fresh_model(cfg_run)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    if use_cluster_selection:
        cluster_mgr = ClusterManager(cfg_run, xai)
        Z_train     = embedder.transform(X_train)
        cluster_mgr.fit_initial(Z_train, y_train)
    else:
        cluster_mgr = None

    energy_train = model.decision_function(X_train)
    energy_train_mean = float(np.mean(energy_train))
    energy_train_std = max(float(np.std(energy_train)), 1e-6)
    drift_mon = None
    if use_drift_score:
        drift_mon = DriftMonitor(
            energy_train,
            ks_alpha            = cfg_run.ks_alpha,
            ks_min_effect       = cfg_run.ks_min_effect,
            perf_ema_alpha      = cfg_run.perf_ema_alpha,
            burst_threshold     = cfg_run.burst_threshold,
            combined_threshold  = cfg_run.combined_threshold,
            weights             = cfg_run.drift_weights,
            emergency_threshold = cfg_run.emergency_threshold,
        )

    results:       List[Dict] = []
    xai_snapshots: List[Dict] = []

    for t_idx, mk in enumerate(test_months):
        idx_m = month_to_idx.get(mk, [])
        if not idx_m:
            continue

        t0 = time.perf_counter()
        Xt = build_hashed_csr(rows, idx_m, cfg_run.hash_dim)
        yt = y[idx_m]

        y_pred  = model.predict(Xt)
        y_prob  = model.predict_proba(Xt)[:, 1]
        metrics = compute_metrics(yt, y_pred, y_prob)

        loss_before = compute_loss_profile(model, Xt, yt)

        energy_t       = model.decision_function(Xt)
        if drift_mon is not None:
            drift_info = drift_mon.update(energy_t)
        else:
            drift_info = {
                "drift_score": 0.0,
                "ks_stat": float("nan"),
                "ks_score": 0.0,
                "confidence_shift_score": 0.0,
                "novelty_score": 0.0,
                "ks_pvalue": float("nan"),
                "drift_detected": False,
                "emergency_flag": False,
                "drift_flag_applied": False,
            }
        drift_flag     = drift_info["drift_flag_applied"]
        emergency_flag = drift_info.get("emergency_flag", False)
        # drift 또는 emergency 시 예산을 drift_budget_scale 배 확대
        if drift_flag or emergency_flag:
            actual_frac = min(1.0, cfg_run.max_update_frac * cfg_run.drift_budget_scale)
        else:
            actual_frac = cfg_run.max_update_frac
        fixed_budget = None
        if fixed_update_counts is not None:
            fixed_budget = max(0, int(fixed_update_counts.get(mk, 0)))
            actual_frac = fixed_budget / len(yt) if len(yt) else 0.0

        n_new, n_pruned = 0, 0
        assignment      = np.full(len(yt), -1, dtype=np.int32)

        if use_cluster_selection and cluster_mgr is not None:
            Zt              = embedder.transform(Xt)
            member_rows      = [rows[i] for i in idx_m]
            assignment, n_new = cluster_mgr.assign(Zt, t_idx, member_rows=member_rows)
            if drift_mon is not None:
                drift_mon.notify_births(n_new)
            n_pruned = cluster_mgr.prune(t_idx)

            X_sel, y_sel, sel_w, per_cl = cluster_mgr.select_update_samples(
                Xt, yt, assignment, drift_flag,
                actual_frac, rng, exact_budget=fixed_budget)
            if fixed_budget is not None and X_sel.shape[0] < fixed_budget:
                selected_idx = np.array(per_cl.get("__selected_idx", []), dtype=np.int64)
                selected_mask = np.zeros(len(yt), dtype=bool)
                selected_mask[selected_idx] = True
                remaining = np.where(~selected_mask)[0]
                n_fill = min(fixed_budget - X_sel.shape[0], len(remaining))
                if n_fill > 0:
                    filler_idx = rng.choice(remaining, size=n_fill, replace=False)
                    all_idx = np.concatenate([selected_idx, filler_idx])
                    cluster_mgr.update_label_distribution(assignment[filler_idx], yt[filler_idx])
                    K = len(cluster_mgr.centroids)
                    cluster_weights = np.array(
                        [cfg_run.cluster_priority_new if cluster_mgr.is_new[i]
                         else (cfg_run.cluster_priority_drift if drift_flag
                               else cfg_run.cluster_priority_stable)
                         for i in range(K)], dtype=np.float64
                    )
                    X_sel = Xt[all_idx]
                    y_sel = yt[all_idx]
                    sel_w = np.array(
                        [cluster_weights[int(assignment[i])]
                         if 0 <= int(assignment[i]) < K else 1.0
                         for i in all_idx], dtype=np.float64
                    )
                    per_cl["__budget_topup"] = int(n_fill)

            xai_snapshots.extend(cluster_mgr.export_xai_snapshot(mk))
        else:
            if fixed_budget is None:
                budget = max(1, int(len(yt) * actual_frac))
            else:
                budget = min(fixed_budget, len(yt))
            if selection_strategy == "uncertainty":
                uncertainty = 1.0 - 2.0 * np.abs(y_prob - 0.5)
                chosen = np.argsort(uncertainty)[::-1][:min(budget, len(yt))]
            elif selection_strategy == "score_shift":
                score_shift = np.abs((energy_t - drift_mon._mu0) / drift_mon._sigma0) \
                    if drift_mon is not None else np.abs((energy_t - energy_train_mean) / energy_train_std)
                chosen = np.argsort(score_shift)[::-1][:min(budget, len(yt))]
            else:
                chosen  = rng.choice(len(yt), size=budget, replace=False) if budget > 0 \
                    else np.array([], dtype=np.int64)
            X_sel   = Xt[chosen]
            y_sel   = yt[chosen]
            sel_w   = np.ones(len(y_sel), dtype=np.float64)

        _apply_update(model, X_sel, y_sel, sel_w,
                      drift_info["drift_score"], cfg_run)

        loss_after = compute_loss_profile(model, Xt, yt)

        K_now        = cluster_mgr.summary()["n_clusters"] if cluster_mgr else float('nan')
        mal_rate_sel = compute_label_efficiency(y_sel)

        row = {
            "month"            : mk,
            "method"           : method_name,
            "total"            : len(yt),
            "updated"          : X_sel.shape[0],
            "n_clusters"       : K_now,
            "n_new_clusters"   : n_new,
            "n_pruned"         : n_pruned,
            "drift_score"      : drift_info["drift_score"],
            "ks_stat"          : drift_info["ks_stat"],
            "ks_score"         : drift_info["ks_score"],
            "confidence_shift_score": drift_info["confidence_shift_score"],
            "novelty_score"    : drift_info["novelty_score"],
            "ks_pvalue"        : drift_info["ks_pvalue"],
            "drift_detected"   : drift_info["drift_detected"],
            "emergency_flag"   : emergency_flag,
            "drift_applied"    : drift_flag,
            "use_drift_score"  : use_drift_score,
            "selection_strategy": selection_strategy,
            "actual_frac"      : round(actual_frac, 4),
            "loss_mode"        : cfg_run.loss_mode,
            "loss_before"      : round(loss_before["loss_mean"], 6),
            "loss_after"       : round(loss_after["loss_mean"], 6),
            "loss_delta"       : round(loss_after["loss_mean"] - loss_before["loss_mean"], 6),
            "mal_rate_selected": mal_rate_sel,
            "time_sec"         : round(time.perf_counter() - t0, 4),
        }
        row.update(metrics)
        results.append(row)

    return results, xai_snapshots


def run_ablation_baseline(X_train, y_train, month_to_idx, test_months,
                          rows, y, embedder, xai, cfg, rng):
    """Random selection, uniform loss [no cluster, no drift score, no drift loss]."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="Random+Standard",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=False,
        selection_strategy="random",
    )


def run_ablation_cluster_only(X_train, y_train, month_to_idx, test_months,
                              rows, y, embedder, xai, cfg, rng):
    """Cluster-guided selection, uniform loss [cluster only, no drift score/loss]."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="Cluster+Standard",
        use_cluster_selection=True,
        use_drift_loss=False,
        use_drift_score=False,
    )


def run_ablation_drift_loss_only(X_train, y_train, month_to_idx, test_months,
                                 rows, y, embedder, xai, cfg, rng):
    """Random selection, drift-reg loss [drift score/loss only, no cluster]."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="Random+DriftLoss",
        use_cluster_selection=False,
        use_drift_loss=True,
        use_drift_score=True,
        selection_strategy="random",
    )


def run_ablation_cluster_driftscore(X_train, y_train, month_to_idx, test_months,
                                    rows, y, embedder, xai, cfg, rng):
    """Cluster-guided selection with drift-score budget scaling, standard loss."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="Cluster+DriftScore",
        use_cluster_selection=True,
        use_drift_loss=False,
        use_drift_score=True,
    )


def run_uncertainty_update(X_train, y_train, month_to_idx, test_months,
                           rows, y, embedder, xai, cfg, rng):
    """Uncertainty-only active learning baseline at the same label budget."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="UncertaintyOnly",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=False,
        selection_strategy="uncertainty",
    )


def run_score_shift_update(X_train, y_train, month_to_idx, test_months,
                           rows, y, embedder, xai, cfg, rng):
    """Non-cluster selective update ranked by confidence-score shift."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="ScoreShiftOnly",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=False,
        selection_strategy="score_shift",
    )


def run_matched_random_update(X_train, y_train, month_to_idx, test_months,
                              rows, y, embedder, xai, cfg, rng,
                              fixed_update_counts: Dict[str, int]):
    """Random selection with the exact same monthly review counts as Proposed."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="RandomMatchedUpdate",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=False,
        selection_strategy="random",
        fixed_update_counts=fixed_update_counts,
    )


def run_matched_uncertainty_update(X_train, y_train, month_to_idx, test_months,
                                   rows, y, embedder, xai, cfg, rng,
                                   fixed_update_counts: Dict[str, int]):
    """Uncertainty selection with the exact same monthly review counts as Proposed."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="UncertaintyMatchedUpdate",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=False,
        selection_strategy="uncertainty",
        fixed_update_counts=fixed_update_counts,
    )


def run_matched_score_shift_update(X_train, y_train, month_to_idx, test_months,
                                   rows, y, embedder, xai, cfg, rng,
                                   fixed_update_counts: Dict[str, int]):
    """Score-shift selection with the exact same monthly review counts as Proposed."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="ScoreShiftMatchedUpdate",
        use_cluster_selection=False,
        use_drift_loss=False,
        use_drift_score=True,
        selection_strategy="score_shift",
        fixed_update_counts=fixed_update_counts,
    )


def run_proposed(X_train, y_train, month_to_idx, test_months,
                 rows, y, embedder, xai, cfg, rng):
    """Proposed model: cluster-guided selection + drift-reg loss [all contributions]."""
    return _run_selective_update_core(
        X_train, y_train, month_to_idx, test_months,
        rows, y, embedder, xai, cfg, rng,
        method_name="Proposed",
        use_cluster_selection=True,
        use_drift_loss=True,
        use_drift_score=True,
        verbose=True,
    )


# =========================================================
# [Hyperparameter Tuning on Val Set  (P5)]
# =========================================================
def tune_on_val(
    X_train, y_train,
    month_to_idx, val_months,
    rows, y, embedder, xai, cfg,
    rng: np.random.Generator,
) -> Config:
    """
    Grid search over Proposed model hyperparameters on validation set.
    Optimises mean AUC-PR (primary metric for imbalanced data, Arp et al. 2022).
    ks_alpha=0.05 is NOT tuned — direct statistical interpretation (P5).
    """
    best_score = -1.0
    best_cfg   = cfg

    effect_cands = [0.05, 0.10, 0.15]
    thr_cands    = [0.15, 0.25, 0.35]
    lam_cands    = [0.5, 1.0, 2.0]
    dist_cands   = [0.25, 0.35, 0.45]

    print(f"\n  [Tuning] Grid search on val set (loss_mode='{cfg.loss_mode}')")
    print(f"  Candidates: effect×{effect_cands} thr×{thr_cands} "
          f"λ×{lam_cands} dist×{dist_cands}")

    for effect in effect_cands:
        for thr in thr_cands:
            for lam in lam_cands:
                for dist in dist_cands:
                    cfg_tmp = Config(**{k: v for k, v in cfg.__dict__.items()})
                    cfg_tmp.ks_min_effect      = effect
                    cfg_tmp.combined_threshold = thr
                    cfg_tmp.drift_lambda       = lam
                    cfg_tmp.new_cluster_dist   = dist

                    rng_inner = np.random.default_rng(cfg.random_state)
                    res, _    = run_proposed(
                        X_train, y_train, month_to_idx, val_months,
                        rows, y, embedder, xai, cfg_tmp, rng_inner,
                    )
                    if not res:
                        continue
                    score = float(np.nanmean([r["auc_pr"] for r in res]))
                    if score > best_score:
                        best_score = score
                        best_cfg   = cfg_tmp
                        print(f"  ★ New best: effect={effect} thr={thr} "
                              f"λ={lam} dist={dist}  val AUC-PR={score:.4f}")

    print(f"\n  [Tuning] Best val AUC-PR={best_score:.4f} → "
          f"effect={best_cfg.ks_min_effect} thr={best_cfg.combined_threshold} "
          f"λ={best_cfg.drift_lambda} dist={best_cfg.new_cluster_dist}")
    return best_cfg


# =========================================================
# [Experiment Timer — wall time + peak memory per experiment]
# =========================================================
@contextmanager
def ExperimentTimer(exp_name: str, out_dir: str = "./results"):
    """Context manager: measures wall-clock time and peak RSS memory.

    Usage (in each exp's main):
        with ExperimentTimer("baseline_static", cfg.out_dir):
            ... run experiment ...

    Appends one row to results/timing_summary.csv:
        exp_name, wall_time_sec, peak_mem_mb, timestamp
    """
    os.makedirs(out_dir, exist_ok=True)
    timing_path = os.path.join(out_dir, "timing_summary.csv")

    tracemalloc.start()
    if _HAS_PSUTIL:
        proc = _psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / 1024 ** 2  # MB
    t0 = time.perf_counter()

    try:
        yield
    finally:
        wall_sec = time.perf_counter() - t0

        # Peak memory: prefer psutil RSS (includes NumPy/scipy C allocations)
        # tracemalloc only tracks Python-managed allocations — use as fallback
        if _HAS_PSUTIL:
            peak_mem_mb = proc.memory_info().rss / 1024 ** 2 - mem_before
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_mem_mb   = peak_bytes / 1024 ** 2
        tracemalloc.stop()

        row = {
            "exp_name"     : exp_name,
            "wall_time_sec": round(wall_sec, 2),
            "peak_mem_mb"  : round(peak_mem_mb, 1),
            "timestamp"    : time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        write_header = not os.path.exists(timing_path)
        with open(timing_path, "a") as f:
            if write_header:
                f.write("exp_name,wall_time_sec,peak_mem_mb,timestamp\n")
            f.write(f"{row['exp_name']},{row['wall_time_sec']},"
                    f"{row['peak_mem_mb']},{row['timestamp']}\n")

        print(f"\n  [Timer] {exp_name}: "
              f"{wall_sec:.1f}s  peak_mem={peak_mem_mb:.0f}MB")
        print(f"  [Timer] Saved → {timing_path}")


# =========================================================
# [Print Helpers — consistent terminal output across all exps]
# =========================================================
_RESULT_HDR = (f"{'Period':<10} {'Method':<22} {'F1':>7} {'AUC-PR':>8} "
               f"{'FPR@90%':>8} {'Updated':>9} {'Total':>8}")
_RESULT_SEP = "-" * 78
_BOX_W      = 78


def print_results_table(df: pd.DataFrame, title: str = "") -> None:
    """Uniform terminal table for all performance experiments (performance runners)."""
    print(f"\n{'='*_BOX_W}")
    if title:
        print(f"  {title}")
        print(f"{'='*_BOX_W}")
    print(_RESULT_HDR)
    print(_RESULT_SEP)
    for _, row in df.sort_values(["month", "method"]).iterrows():
        updated = (int(row["updated"])
                   if "updated" in row and not pd.isna(row["updated"]) else 0)
        total   = int(row["total"]) if "total" in row else 0
        print(f"{str(row['month']):<10} {str(row['method']):<22} "
              f"{row['f1']:>7.3f} {row['auc_pr']:>8.3f} "
              f"{row['fpr_at_tpr90']:>8.3f} {updated:>9} {total:>8}")
    print(_RESULT_SEP)
    # Mean row
    print(f"{'Mean':<10} {'':<22} "
          f"{df['f1'].mean():>7.3f} {df['auc_pr'].mean():>8.3f} "
          f"{df['fpr_at_tpr90'].mean():>8.3f}")
    print(f"{'='*_BOX_W}")


def print_xai_table(df: pd.DataFrame, title: str = "",
                    kind: str = "drebin") -> None:
    """Uniform terminal table for XAI case-study experiments (evidence runners)."""
    print(f"\n{'='*_BOX_W}")
    if title:
        print(f"  {title}")
        print(f"{'='*_BOX_W}")

    if kind == "drebin":
        hdr = (f"{'CID':>4}  {'N':>7}  {'Dominant Family':<22}  "
               f"{'Purity':>6}  {'Mal%Train':>9}  Top Features")
        print(hdr)
        print("-" * _BOX_W)
        for _, row in df.sort_values("n_drebin_samples", ascending=False).iterrows():
            feat = str(row["top_features"])[:28]
            print(f"{int(row['cluster_id']):>4}  "
                  f"{int(row['n_drebin_samples']):>7}  "
                  f"{str(row['dominant_family']):<22}  "
                  f"{row['family_purity']:>6.3f}  "
                  f"{row['mal_ratio_train']*100:>9.1f}  {feat}")
        mean_p = df["family_purity"].mean()
        n_high = (df["family_purity"] > 0.8).sum()
    else:  # marvin gw/mw
        hdr = (f"{'CID':>4}  {'N':>7}  {'N_MW':>6}  {'N_GW':>6}  "
               f"{'Dominant':<10}  {'Purity':>6}  {'Mal%Train':>9}  Top Features")
        print(hdr)
        print("-" * _BOX_W)
        for _, row in df.sort_values("n_total", ascending=False).iterrows():
            feat = str(row["top_features"])[:22]
            print(f"{int(row['cluster_id']):>4}  "
                  f"{int(row['n_total']):>7}  "
                  f"{int(row['n_malware']):>6}  "
                  f"{int(row['n_goodware']):>6}  "
                  f"{str(row['dominant_label']):<10}  "
                  f"{row['purity']:>6.3f}  "
                  f"{row['mal_ratio_train']*100:>9.1f}  {feat}")
        mean_p = df["purity"].mean()
        n_high = (df["purity"] > 0.8).sum()

    print("-" * _BOX_W)
    print(f"  Mean purity: {mean_p:.3f}  |  "
          f"High-purity clusters (>0.8): {n_high} / {len(df)}")
    print(f"{'='*_BOX_W}")


# =========================================================
# [Plot Helpers]
# =========================================================
COLORS = {
    "Static"             : "#d62728",
    "FullUpdate"         : "#2ca02c",
    "Retraining"         : "#9467bd",
    "Transcend+Adapt"    : "#8c564b",
    "ADWIN+Retrain"      : "#e377c2",
    "Random+Standard"    : "#aec7e8",
    "Cluster+Standard"   : "#ffbb78",
    "Random+DriftLoss"   : "#98df8a",
    "Cluster+DriftScore" : "#c5b0d5",
    "UncertaintyOnly"    : "#17becf",
    "Proposed"           : "#1f77b4",
}
MARKERS = {
    "Static"             : "x",
    "FullUpdate"         : "s",
    "Retraining"         : "D",
    "Transcend+Adapt"    : "^",
    "ADWIN+Retrain"      : "v",
    "Random+Standard"    : "o",
    "Cluster+Standard"   : "o",
    "Random+DriftLoss"   : "o",
    "Cluster+DriftScore" : "o",
    "UncertaintyOnly"    : "s",
    "Proposed"           : "o",
}


def _setup_xaxis(ax, all_months):
    step = max(1, len(all_months) // 12)
    ax.set_xticks(range(0, len(all_months), step))
    ax.set_xticklabels([all_months[i] for i in range(0, len(all_months), step)],
                       rotation=45, ha="right", fontsize=8)


def plot_metric(all_results: List[Dict], metric: str, ylabel: str,
                title: str, path: str, method_subset: Optional[List[str]] = None,
                drift_rows: Optional[List[Dict]] = None) -> None:
    df = pd.DataFrame(all_results)
    if method_subset:
        df = df[df["method"].isin(method_subset)]
    fig, ax = plt.subplots(figsize=(13, 4))
    all_months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(all_months)}

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("month")
        xs  = [midx[m] for m in sub["month"]]
        ax.plot(xs, sub[metric],
                label=method,
                color=COLORS.get(method, "grey"),
                marker=MARKERS.get(method, "o"),
                linewidth=1.5, markersize=4)

    if drift_rows:
        df_d = pd.DataFrame(drift_rows)
        if "drift_detected" in df_d.columns:
            for _, row in df_d[df_d["drift_detected"].fillna(False)].iterrows():
                if row["month"] in midx:
                    ax.axvspan(midx[row["month"]] - 0.4, midx[row["month"]] + 0.4,
                               alpha=0.08, color="red", linewidth=0)

    _setup_xaxis(ax, all_months)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def plot_ablation_bar(ablation_results: List[Dict], path: str) -> None:
    """Bar chart comparison of ablation variants + Proposed (mean ± std)."""
    df      = pd.DataFrame(ablation_results)
    methods = ["Random+Standard", "Cluster+Standard", "Random+DriftLoss",
               "Cluster+DriftScore", "Proposed"]
    metrics = ["f1", "tpr", "auc_pr"]
    labels  = ["Random\nstandard", "Cluster\nstandard",
               "Random\ndrift loss", "Cluster\ndrift score",
               "Proposed\nfull"]

    x     = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, label) in enumerate(zip(methods, labels)):
        sub   = df[df["method"] == method]
        means = [sub[m].mean() for m in metrics]
        stds  = [sub[m].std()  for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               label=label, color=COLORS.get(method, "grey"),
               yerr=stds, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["F1 (macro)", "TPR", "AUC-PR"], fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: contribution of each module")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def plot_label_efficiency(all_results: List[Dict], global_mal_rate: float,
                          path: str) -> None:
    """
    Selected malicious rate over time for methods with selective labelling.
    Compares against global malicious rate baseline.
    """
    df      = pd.DataFrame(all_results)
    methods = ["Random+Standard", "Cluster+Standard", "Random+DriftLoss",
               "Cluster+DriftScore", "Proposed", "UncertaintyOnly"]
    df = df[df["method"].isin(methods) & df["mal_rate_selected"].notna()]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    all_months = sorted(df["month"].unique())
    midx = {m: i for i, m in enumerate(all_months)}

    ax.axhline(global_mal_rate, color="grey", lw=1.5, ls="--",
               label=f"Global mal rate ({global_mal_rate:.3f})")

    for method in methods:
        sub = df[df["method"] == method].sort_values("month")
        if sub.empty:
            continue
        xs = [midx[m] for m in sub["month"] if m in midx]
        ys = sub[sub["month"].isin(midx)]["mal_rate_selected"].values
        ax.plot(xs, ys, label=method,
                color=COLORS.get(method, "grey"),
                marker=MARKERS.get(method, "o"),
                linewidth=1.5, markersize=4)

    _setup_xaxis(ax, all_months)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Selected malicious rate")
    ax.set_title("Label Efficiency: malicious rate of selected samples over time")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def plot_cluster_drift_linkage(proposed_results: List[Dict], path: str) -> None:
    """Combined drift score / 3-signal decomposition / cluster lifecycle."""
    df = pd.DataFrame(proposed_results)
    required = {"drift_score", "ks_score", "confidence_shift_score", "novelty_score",
                "n_clusters", "n_new_clusters", "n_pruned", "drift_detected"}
    if not required.issubset(df.columns):
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    xs   = range(len(df))
    tks  = list(xs)[::max(1, len(df) // 12)]
    xlbl = df["month"].iloc[tks].tolist()

    colors = ["#d62728" if d else "#aec7e8"
              for d in df["drift_detected"].fillna(False)]
    axes[0].bar(xs, df["drift_score"], color=colors, alpha=0.85)
    axes[0].axhline(df["drift_score"].mean(), color="navy", lw=0.8,
                    ls="--", label="mean")
    axes[0].set_ylabel("Combined drift score")
    axes[0].set_title("(a) Combined drift score — red bars = drift detected")
    axes[0].legend(fontsize=8)

    axes[1].plot(xs, df["ks_score"],      "#1f77b4", lw=1.5, label="KS score (S1)")
    axes[1].plot(xs, df["confidence_shift_score"], "#ff7f0e", lw=1.2, ls="--",
                 label="Confidence shift score (S2)")
    axes[1].plot(xs, df["novelty_score"], "#9467bd", lw=1.0, ls=":",
                 label="Novelty score (S3)")
    axes[1].set_ylabel("Signal score [0,1]")
    axes[1].set_title("(b) 3-signal decomposition: KS / Confidence shift / Novelty")
    axes[1].legend(fontsize=8)

    axes[2].plot(xs, df["n_clusters"], "#2ca02c", lw=1.5, label="K (total)")
    axes[2].bar(xs, df["n_new_clusters"], color="#ff7f0e", alpha=0.7, label="births")
    axes[2].bar(xs, [-v for v in df["n_pruned"]], color="#d62728", alpha=0.7,
                label="deaths")
    axes[2].set_ylabel("Cluster count")
    axes[2].set_title("(c) Cluster lifecycle — births and deaths per month")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(tks)
        ax.set_xticklabels(xlbl, rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def plot_xai_evolution(xai_snapshots: List[Dict], path: str) -> None:
    df = pd.DataFrame(xai_snapshots)
    if df.empty:
        return
    counts  = df.groupby("cluster_id")["month"].count()
    stable  = counts[counts >= 6].index.tolist() or counts.nlargest(10).index.tolist()
    mean_n  = df[df["cluster_id"].isin(stable)].groupby(
        "cluster_id")["n_samples"].mean()
    top_ids = mean_n.nlargest(min(10, len(mean_n))).index.tolist()

    fig, ax = plt.subplots(figsize=(13, 5))
    cmap    = plt.get_cmap("tab10")
    all_months = sorted(df["month"].unique())
    for j, cid in enumerate(top_ids):
        sub = df[df["cluster_id"] == cid].sort_values("month")
        feat_label = ""
        if not sub.empty and sub["top_features"].iloc[0]:
            first = sub["top_features"].iloc[0].split(";")[0].split("(")[0].strip()
            feat_label = f" [{first[:20]}]"
        xs = [all_months.index(m) for m in sub["month"] if m in all_months]
        ax.plot(xs, sub["mal_ratio"].values,
                label=f"C{cid:03d}{feat_label}",
                color=cmap(j % 10), lw=1.4, marker="o", markersize=3)

    _setup_xaxis(ax, all_months)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Malicious ratio")
    ax.set_title("XAI (C2): Cluster malicious ratio over time — top clusters by size")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def save_summary(all_results: List[Dict], path: str) -> None:
    df      = pd.DataFrame(all_results)
    metrics = ["f1", "tpr", "fpr", "fpr_at_tpr90", "auc_roc", "auc_pr"]
    df.groupby("method")[metrics].agg(["mean", "std"]).round(4).to_csv(path)
    print(f"  -> {path}")

    print("\n" + "=" * 77)
    print("  PERFORMANCE SUMMARY  (mean ± std over test months)")
    print("=" * 77)
    method_order = [
        "Static", "FullUpdate", "Retraining",
        "Transcend+Adapt", "ADWIN+Retrain",
        "Ablation-Baseline", "Ablation-+Cluster",
        "Ablation-+DriftLoss", "Proposed",
    ]
    print(f"{'Method':<22}  {'F1':>12}  {'AUC-PR':>12}  {'FPR@90%':>12}")
    print("-" * 65)
    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        print(f"{method:<22}  "
              f"{sub['f1'].mean():.3f}±{sub['f1'].std():.3f}  "
              f"{sub['auc_pr'].mean():.3f}±{sub['auc_pr'].std():.3f}  "
              f"{sub['fpr_at_tpr90'].mean():.3f}±{sub['fpr_at_tpr90'].std():.3f}")

    sec = df[df["method"] == "Proposed"]
    if len(sec):
        tot = sec["total"].sum()
        upd = sec["updated"].sum()
        if tot > 0:
            print(f"\n  Proposed label cost: {upd:,}/{tot:,} = "
                  f"{upd/tot*100:.1f}%  (reduction: {(1-upd/tot)*100:.1f}%)")
    print("=" * 77)
