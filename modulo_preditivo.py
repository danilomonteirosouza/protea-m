# -*- coding: utf-8 -*-
import os, json, time, argparse, warnings, math, itertools, random
import numpy as np, pandas as pd
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import comb
from scipy.special import erf
from scipy.stats import ttest_rel, wilcoxon, shapiro, t as student_t
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, balanced_accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support, roc_curve, auc, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed, parallel_backend
from typing import Optional

# tenta StratifiedGroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF_Avail

    SGKF_AVAILABLE = True
except Exception:
    SGKF_AVAILABLE = False

# ======================
# CONFIG (padrões)
# ======================
RANDOM_STATE_BASE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_MODELOS_ENSEMBLE = 25
TOPK_LIST = [1, 3]
ALVO = "PlanoIntervencao"
ALVO_GUIA = "PlanoGuia"
ID_COL = "PessoaID"
TEXT_COL = "DescricaoComportamento"
LABEL_SMOOTH_EPS = 0.05

CATEG_COLS = [
    "PlanoGuia", "NivelComunicacao", "FuncaoComunicativa", "Ambiente",
    "ParceiroComunicacional", "SuporteNecessario", "FormaPreferida"
]


# ======== Performance helpers ========
def _effective_n_jobs(n_jobs):
    if n_jobs is None or n_jobs == 0:
        return 0
    if n_jobs < 0:
        try:
            cpu = os.cpu_count() or 1
            if n_jobs == -1:
                return cpu
            return max(1, cpu + 1 + n_jobs)
        except Exception:
            return -1
    return int(max(1, n_jobs))


def set_num_threads(blas_threads: Optional[int]):
    """
    Se blas_threads > 0: limita threads de BLAS/OpenMP.
    Se blas_threads <= 0 ou None: não força limites (deixa usar tudo).
    """
    if blas_threads is None or blas_threads <= 0:
        return
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(int(blas_threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("MKL_DYNAMIC", "TRUE")
    try:
        import mkl;
        mkl.set_num_threads(int(blas_threads))
    except Exception:
        pass
    try:
        import numexpr as ne;
        ne.set_num_threads(int(blas_threads))
    except Exception:
        pass


# ======================
# PATH / IO helpers
# ======================
def _common_candidates(script_dir: str, name: str):
    return [os.path.abspath(name),
            os.path.abspath(os.path.join(script_dir, name)),
            os.path.abspath(os.path.join(script_dir, "dados", name)),
            os.path.abspath(os.path.join(script_dir, "..", "dados", name)),
            os.path.abspath(os.path.join(script_dir, "..", "..", "dados", name))]


def resolve_path(csv_arg: str, default_basename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_arg and os.path.isabs(csv_arg) and os.path.exists(csv_arg): return csv_arg
    candidates = _common_candidates(script_dir, csv_arg if csv_arg else default_basename)
    for c in candidates:
        if os.path.exists(c): return c
    if csv_arg:
        bn = os.path.basename(csv_arg)
        for c in _common_candidates(script_dir, bn):
            if os.path.exists(c): return c
    raise FileNotFoundError("Arquivo não encontrado. Testados:\n - " + "\n - ".join(candidates))


import pandas as pd

import pandas as pd

def read_csv_auto(path: str) -> pd.DataFrame:
    def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).replace("\ufeff", "").strip().rstrip(",") for c in df.columns]
        return df

    def looks_like_wrong_sep(df: pd.DataFrame, expected_delim: str) -> bool:
        # Se só veio 1 coluna, provavelmente separador errado
        if df.shape[1] != 1:
            return False
        h = str(df.columns[0])
        # Se o header tem MUITAS ocorrências do outro delimitador, é sinal claro
        if expected_delim == ";":
            return h.count(",") > 5   # era para ser ',' e não ';'
        if expected_delim == ",":
            return h.count(";") > 5   # era para ser ';' e não ','
        return False

    # 1) Tenta inferir automaticamente
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
        df = clean_cols(df)
        # Se por acaso inferiu errado e colou header com ';', rejeita
        if any(";" in c for c in df.columns):
            raise ValueError("Inferência falhou: header contém ';'.")
        return df
    except Exception:
        pass

    # 2) Tenta ';' MAS valida
    try:
        df = pd.read_csv(path, sep=";", engine="c", encoding="utf-8-sig", low_memory=False)
        df = clean_cols(df)
        if looks_like_wrong_sep(df, expected_delim=";"):
            raise ValueError("Leitura com ';' parece errada (ficou 1 coluna, header tem muitas vírgulas).")
        return df
    except Exception:
        pass

    # 3) Fallback final: ','
    df = pd.read_csv(path, sep=",", engine="c", encoding="utf-8-sig", low_memory=False)
    df = clean_cols(df)
    return df

# ======================
# BUILD SBERT + *_freq
# ======================
def frequency_encode(series: pd.Series) -> pd.Series:
    counts = series.astype(str).value_counts(dropna=False)
    freqs = counts / counts.sum()
    return series.astype(str).map(freqs).astype(np.float32)


def _pick_device(device_flag: str) -> str:
    device_flag = str(device_flag).lower().strip()
    if device_flag in ("cuda", "cpu"): return device_flag
    try:
        import torch;
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_sbert_base(csv_original: str, out_csv: str, model_name: str,
                     batch_size: int = 64, device_flag: str = "auto") -> str:
    print(f"[BUILD] Lendo base original: {csv_original}")
    df = read_csv_auto(csv_original)
    print("[DEBUG] Colunas lidas:", list(df.columns))
    for c in [ID_COL, ALVO, ALVO_GUIA, TEXT_COL]:
        assert c in df.columns, f"Coluna obrigatória não encontrada: {c}"

    # acrescenta *_freq (não vaza alvo)
    cat_cols = CATEG_COLS
    for c in cat_cols:
        if c in df.columns: df[c + "_freq"] = frequency_encode(df[c])

    device = _pick_device(device_flag)
    print(f"[BUILD] SBERT={model_name} | device={device} | batch={batch_size}")
    model = SentenceTransformer(model_name, device=device)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    emb = model.encode(texts, batch_size=int(batch_size), show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    emb_cols = [f"SBERT_{i:03d}" for i in range(emb.shape[1])]
    out_df = pd.concat(
        [df[[ID_COL, ALVO, ALVO_GUIA, TEXT_COL] + [c for c in df.columns if c.endswith("_freq")] + cat_cols],
         pd.DataFrame(emb, columns=emb_cols, index=df.index)], axis=1)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[BUILD] ✅ Base SBERT gerada: {out_csv}")
    return out_csv


def load_precomputed_base(csv_path, emb_prefix="SBERT_", alvo=ALVO, id_col=ID_COL, text_col=TEXT_COL):
    df = read_csv_auto(csv_path)
    assert alvo in df.columns and id_col in df.columns
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    freq_cols = [c for c in df.columns if c.endswith("_freq")]
    X_emb = df[emb_cols].astype("float32").to_numpy()
    X_freq = df[freq_cols].astype("float32").to_numpy() if freq_cols else None
    X_all = np.hstack([X_freq, X_emb]).astype("float32") if X_freq is not None else X_emb.astype("float32")
    y_text = df[alvo].astype(str).values
    groups = df[id_col].astype(str).values
    texts = df[text_col].fillna("").astype(str).values if text_col in df.columns else np.array([""] * len(df))
    return df, X_all, y_text, groups, emb_cols, freq_cols, texts


# ======================
# ELM
# ======================
def orthogonal_matrix(rows, cols, rng):
    if rows >= cols:
        A = rng.normal(size=(rows, cols));
        Q, _ = np.linalg.qr(A);
        return Q[:, :cols]
    A = rng.normal(size=(cols, rows));
    Q, _ = np.linalg.qr(A);
    return Q[:, :rows].T


def he_scale(d):     return np.sqrt(2.0 / max(1, d))


def xavier_scale(d): return np.sqrt(1.0 / max(1, d))


class ELM:
    def __init__(self, input_size, hidden_size, output_size,
                 activation="relu", random_state=None, reg=1e-2):
        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.output_size = output_size
        self.activation = activation;
        self.random_state = random_state;
        self.reg = reg
        rng = np.random.RandomState(self.random_state)
        W_raw = orthogonal_matrix(self.input_size, self.hidden_size, rng)
        a = str(self.activation).lower()
        scale = he_scale(self.input_size) if a in ("relu", "leaky_relu", "elu", "selu", "gelu", "mish", "swish",
                                                   "softplus") \
            else xavier_scale(self.input_size) if a in ("tanh", "sigmoid", "softsign") \
            else 1.0 / np.sqrt(max(1, self.input_size))
        self.W = W_raw * scale
        self.b = rng.normal(scale=0.5, size=(self.hidden_size,))
        self.beta = None

    def _sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def _softplus(self, X):
        return np.log1p(np.exp(-np.abs(X))) + np.maximum(X, 0)

    def _gelu(self, X):
        return 0.5 * X * (1.0 + erf(X / np.sqrt(2.0)))

    def _mish(self, X):
        return X * np.tanh(self._softplus(X))

    def _selu(self, X):
        lam = 1.0507009873554805;
        alpha = 1.6732632423543772
        return lam * np.where(X > 0, X, alpha * (np.exp(X) - 1.0))

    def _activation_fn(self, X):
        a = str(self.activation).lower()
        if a == "sigmoid":    return self._sigmoid(X)
        if a == "tanh":       return np.tanh(X)
        if a == "relu":       return np.maximum(0, X)
        if a == "leaky_relu": return np.where(X >= 0, X, 0.01 * X)
        if a == "elu":        return np.where(X >= 0, X, 1.0 * (np.exp(X) - 1.0))
        if a == "selu":       return self._selu(X)
        if a == "softplus":   return self._softplus(X)
        if a == "swish":      return X * self._sigmoid(1.0 * X)
        if a == "gelu":       return self._gelu(X)
        if a == "mish":       return self._mish(X)
        if a == "softsign":   return X / (1.0 + np.abs(X))
        if a == "hardtanh":   return np.clip(X, -1.0, 1.0)
        if a in ("linear", "identity"): return X
        raise ValueError(f"Ativação não suportada: {self.activation}")

    def _hidden(self, X):
        H = X.dot(self.W) if hasattr(X, "dot") else np.dot(X, self.W)
        H = self._activation_fn(H + self.b)
        return H - H.mean(axis=0, keepdims=True)

    def fit(self, X, Y, sample_weight=None):
        H = self._hidden(X)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1, 1)
            Hw = H * np.sqrt(w);
            Yw = Y * np.sqrt(w)
            HtH = Hw.T @ Hw;
            A = HtH + self.reg * np.eye(HtH.shape[0])
            self.beta = np.linalg.solve(A, Hw.T @ Yw)
        else:
            HtH = H.T @ H;
            A = HtH + self.reg * np.eye(HtH.shape[0])
            self.beta = np.linalg.solve(A, H.T @ Y)

    def predict_logits(self, X):
        H = self._hidden(X)
        return H.dot(self.beta)

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits);
        return exp / np.sum(exp, axis=1, keepdims=True)


# ======================
# Métricas / utilitários
# ======================
def topk_accuracy_score(y_true, proba, k=1):
    topk = np.argsort(-proba, axis=1)[:, :k]
    return np.mean([yt in row for yt, row in zip(y_true, topk)])


def brier_score_multiclass(y_true_int, proba, n_classes):
    Y = np.eye(n_classes, dtype=float)[y_true_int]
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def expected_calibration_error(y_true_int, proba, n_bins=15):
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true_int).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if np.any(m):
            bin_acc = np.mean(correct[m])
            bin_conf = np.mean(conf[m])
            ece += (np.sum(m) / len(conf)) * abs(bin_conf - bin_acc)
    return float(ece)


def plot_reliability_diagram(y_true_int, proba, out_path, n_bins=15, title="Reliability Diagram (Top-1)"):
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true_int).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if np.any(m):
            xs.append(np.mean(conf[m]))
            ys.append(np.mean(correct[m]))
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    if len(xs) > 0:
        plt.scatter(xs, ys)
        plt.plot(xs, ys)
    plt.xlabel("Confiança média (bin)")
    plt.ylabel("Acurácia média (bin)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _roc_pr_micro_macro(y_true_int, y_prob, out_png_roc, out_png_pr, title_prefix, n_classes=None):
    """
    Gera ROC/PR micro. Evita warnings quando só há 1 classe em y_true.
    Usa binarização com TODAS as classes (0..n_classes-1) para que exista negativo,
    mesmo quando só há uma classe presente no fold.
    """
    try:
        if y_true_int.size == 0:
            return  # nada a fazer

        # se não vier, infere pelo y_prob
        if n_classes is None:
            n_classes = y_prob.shape[1]

        # binariza com todas as classes
        all_labels = np.arange(n_classes)
        y_bin = label_binarize(y_true_int, classes=all_labels)

        # micro-ROC/PR só faz sentido se y_bin.ravel() tiver 0 e 1
        y_flat = y_bin.ravel()
        if not (np.any(y_flat == 0) and np.any(y_flat == 1)):
            # pula com segurança
            return

        # seleciona as mesmas colunas de y_prob
        P = y_prob[:, all_labels] if y_prob.shape[1] >= n_classes else y_prob

        # ROC (micro)
        fpr_micro, tpr_micro, _ = roc_curve(y_flat, P.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # AUC macro pode falhar quando há coluna sem positivos → trate via try
        try:
            auc_macro = roc_auc_score(y_bin, P, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = np.nan

        plt.figure()
        plt.plot(fpr_micro, tpr_micro, label=f"micro AUC = {auc_micro:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR");
        plt.ylabel("TPR");
        plt.title(f"{title_prefix} — ROC (macro AUC={auc_macro:.3f})")
        plt.legend(loc="lower right");
        plt.tight_layout()
        plt.savefig(out_png_roc, bbox_inches="tight");
        plt.close()

        # PR (micro)
        precision_micro, recall_micro, _ = precision_recall_curve(y_flat, P.ravel())
        ap_micro = average_precision_score(y_bin, P, average="micro")
        try:
            ap_macro = average_precision_score(y_bin, P, average="macro")
        except Exception:
            ap_macro = np.nan

        plt.figure()
        plt.plot(recall_micro, precision_micro, label=f"micro AP = {ap_micro:.3f}")
        plt.xlabel("Recall");
        plt.ylabel("Precision");
        plt.title(f"{title_prefix} — PR (macro AP={ap_macro:.3f})")
        plt.legend(loc="lower left");
        plt.tight_layout()
        plt.savefig(out_png_pr, bbox_inches="tight");
        plt.close()

    except Exception:
        pass


def _bars_triple(values_macro, values_micro, values_weighted, ylabel, title, out_path):
    vals = [values_macro, values_micro, values_weighted]
    labels = ["macro", "micro", "weighted"]
    plt.figure()
    xs = np.arange(len(vals))
    plt.bar(xs, vals)
    plt.xticks(xs, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _save_confusion(y_true_int, y_pred_int, classes, out_dir, prefix):
    """Salva CSV e PNG da matriz de confusão (contagens e normalizada)."""
    os.makedirs(out_dir, exist_ok=True)
    labels_idx = np.arange(len(classes))
    cm = confusion_matrix(y_true_int, y_pred_int, labels=labels_idx)
    # CSV
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"), sep=";", encoding="utf-8-sig"
    )
    # PNG (contagens)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(include_values=True, cmap="Blues", xticks_rotation=45, colorbar=True)
    plt.title(f"Matriz de Confusão — {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    # PNG (normalizada por verdade)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # evita divisão por zero quando uma linha tiver soma 0
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    disp_norm.plot(include_values=True, cmap="Blues", xticks_rotation=45, colorbar=True,
                   values_format=".2f")
    plt.title(f"Matriz de Confusão (Normalizada) — {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix_normalized.png"), bbox_inches="tight")
    plt.close()


def _save_confusion_minimal(cm, out_path, annotate=True, fmt="d", fontsize=14, figsize=(10, 10), dpi=200):
    """
    Salva uma imagem *minimalista* da matriz de confusão:
    - sem título, sem eixos, sem colorbar, só o heatmap central
    - opcionalmente com anotações (valores) no centro de cada célula.
    - robusto a recortes agressivos do Matplotlib.
    """
    import numpy as _np
    fig = plt.figure(figsize=figsize)
    # ocupa todo o canvas (evita corte quando axes estão ocultos)
    ax = fig.add_axes([0, 0, 1, 1])
    # garantir tipo numérico simples
    cm_plot = _np.asarray(cm, dtype=float)
    im = ax.imshow(cm_plot, aspect="equal", cmap="Blues", interpolation="nearest")

    if annotate:
        n_rows, n_cols = cm_plot.shape
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, format(int(cm[i, j]), fmt), ha="center", va="center", fontsize=fontsize)

    ax.set_xticks([]);
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # nada de tight/pad; isso evita recorte do conteúdo
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def avaliar_conjunto(y_true, y_pred, y_prob, classes, out_dir, prefix, topk_list=(1, 3)):
    os.makedirs(out_dir, exist_ok=True)
    prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    pd.DataFrame(
        {"classe": classes, "precision": prec_cls, "recall": rec_cls, "f1": f1_cls, "support": support}).to_csv(
        os.path.join(out_dir, f"{prefix}_per_class_metrics.csv"), sep=";", index=False, encoding="utf-8-sig"
    )
    report_txt = classification_report(y_true, y_pred, labels=np.arange(len(classes)), target_names=classes,
                                       zero_division=0)
    with open(os.path.join(out_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                                         zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro",
                                                                         zero_division=0)
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                                            zero_division=0)
    rows_topk = [{"k": k, "topk_accuracy": topk_accuracy_score(y_true, y_prob, k)} for k in topk_list]
    pd.DataFrame(rows_topk).to_csv(os.path.join(out_dir, f"{prefix}_topk_accuracy.csv"), sep=";", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame([{
        "accuracy": acc, "balanced_accuracy": bacc, "kappa": kappa,
        "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
        "precision_micro": prec_micro, "recall_micro": rec_micro, "f1_micro": f1_micro,
        "precision_weighted": prec_weight, "recall_weighted": rec_weight, "f1_weighted": f1_weight
    }]).to_csv(os.path.join(out_dir, f"{prefix}_aggregate_metrics.csv"), sep=";", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out_dir, f"{prefix}_matriz_confusao.csv"),
                                                            sep=";", encoding="utf-8-sig")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.title(f"Matriz de Confusão - {prefix}");
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_matriz_confusao.png"), bbox_inches="tight");
    plt.close()

    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    try:
        roc_auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc_macro = np.nan
    try:
        roc_auc_micro = roc_auc_score(y_true_bin, y_prob, average="micro", multi_class="ovr")
    except Exception:
        roc_auc_micro = np.nan
    try:
        pr_ap_macro = average_precision_score(y_true_bin, y_prob, average="macro")
    except Exception:
        pr_ap_macro = np.nan
    try:
        pr_ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")
    except Exception:
        pr_ap_micro = np.nan
    pd.DataFrame([
        {"tipo": "roc_auc_macro", "valor": roc_auc_macro},
        {"tipo": "roc_auc_micro", "valor": roc_auc_micro},
        {"tipo": "pr_ap_macro", "valor": pr_ap_macro},
        {"tipo": "pr_ap_micro", "valor": pr_ap_micro},
    ]).to_csv(os.path.join(out_dir, f"{prefix}_roc_pr_macro_micro.csv"), sep=";", index=False, encoding="utf-8-sig")

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro_curve = auc(fpr_micro, tpr_micro)
        plt.figure();
        plt.plot(fpr_micro, tpr_micro, label=f"micro AUC = {roc_auc_micro_curve:.3f}")
        plt.plot([0, 1], [0, 1], "k--");
        plt.xlabel("FPR");
        plt.ylabel("TPR");
        plt.title(f"ROC micro - {prefix}")
        plt.legend(loc="lower right");
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve_micro.png"), bbox_inches="tight");
        plt.close()
    except Exception:
        pass

    return {"accuracy": acc, "balanced_accuracy": bacc, "kappa": kappa,
            "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
            "precision_micro": prec_micro, "recall_micro": rec_micro, "f1_micro": f1_micro,
            "precision_weighted": prec_weight, "recall_weighted": rec_weight, "f1_weighted": f1_weight}, cm


def salvar_predicoes(ids, y_true, y_pred, y_prob, classes, out_csv):
    col_probs = {f"prob_{cls}": y_prob[:, i] for i, cls in enumerate(classes)}
    df_pred = pd.DataFrame({"PessoaID": ids, "PlanoIntervencao_Real": classes[y_true],
                            "PlanoIntervencao_Predita": classes[y_pred], **col_probs})
    df_pred.to_csv(out_csv, sep=";", index=False, encoding="utf-8-sig")


def _align_proba(P, classes_fit, n_classes):
    """
    Garante que as probabilidades tenham todas as colunas (0..n_classes-1),
    alinhando as saídas quando o classificador foi treinado sem algumas classes.
    """
    classes_fit = np.asarray(classes_fit, dtype=int)
    out = np.zeros((P.shape[0], n_classes), dtype=np.float32)
    out[:, classes_fit] = P
    return out


# ===== Helpers extras: salvar/plotar e testes =====
def _safe_json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_kv_csv(d: dict, path_csv: str):
    rows = [{"chave": k, "valor": v} for k, v in d.items()]
    pd.DataFrame(rows).to_csv(path_csv, sep=";", index=False, encoding="utf-8-sig")


def _bar_plot(vals, labels, title, ylabel, out_path):
    plt.figure()
    xs = np.arange(len(vals))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def fmt_p(p):
    if p is None: return "NA"
    try:
        return "< 1e-300" if p < 1e-300 else f"{p:.6e}"
    except Exception:
        return str(p)


def mcnemar_test(b, c, exact=True):
    b = int(b);
    c = int(c);
    n = b + c
    if n == 0:
        return {"stat": 0.0, "pvalue": 1.0, "method": "indeterminado (sem discordâncias)"}
    if exact and n < 25:
        k = min(b, c)
        p_cum = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2.0 * p_cum)
        return {"stat": None, "pvalue": float(p), "method": f"exato (binomial bilateral), n={n}"}
    num = (abs(b - c) - 1) ** 2
    den = b + c if (b + c) > 0 else 1.0
    chi2 = num / den
    try:
        from scipy.stats import chi2 as _chi2
        p = 1.0 - _chi2.cdf(chi2, df=1)
    except Exception:
        z = (abs(b - c) - 1) / math.sqrt(b + c) if (b + c) > 0 else 0.0
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / np.sqrt(2))))
    return {"stat": float(chi2), "pvalue": float(p), "method": f"qui-quadrado (correção), n={n}"}


def paired_ttest_safe(a, b):
    a = np.asarray(a, dtype=float);
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, "Entrada do t-test deve ser vetores 1-D de mesmo tamanho."
    d = a - b;
    n = d.size
    if n < 2:
        return {"t": np.nan, "p": np.nan, "p_fmt": "nan", "n": n, "deltas": d}
    if np.allclose(d, 0.0):
        return {"t": 0.0, "p": 1.0, "p_fmt": f"{1.0:.6e}", "n": n, "deltas": d}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_std = ttest_rel(a, b, nan_policy="omit")
    try:
        df = n - 1
        p_sf = 2.0 * float(student_t.sf(np.abs(t_stat), df))
        p_sf = float(max(0.0, min(1.0, p_sf)))
        p_final = p_sf if (np.isnan(p_std) or p_std == 0.0 or p_sf < p_std) else float(p_std)
    except Exception:
        p_final = float(p_std)
    return {"t": float(t_stat), "p": p_final, "p_fmt": f"{p_final:.6e}", "n": n, "deltas": d}


def effect_sizes_and_ci(deltas, alpha=0.05):
    d = np.asarray(deltas, dtype=float)
    n = d.size
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float('nan')
    dz = float(m / sd) if (n > 1 and sd > 0) else float('inf' if m > 0 else (-float('inf') if m < 0 else 0.0))
    # IC da média (t)
    if n > 1 and np.isfinite(sd):
        tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
        se = sd / math.sqrt(n)
        ci_lo = m - tcrit * se
        ci_hi = m + tcrit * se
    else:
        ci_lo, ci_hi = float('nan'), float('nan')
    return {
        "mean_delta": m, "sd_delta": sd, "n": n,
        "cohen_dz": dz, "ci_mean_lo": ci_lo, "ci_mean_hi": ci_hi
    }


def paired_permutation_test(deltas, reps_max=100000, seed=123):
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n == 0:
        return {"p": np.nan, "p_fmt": "nan", "method": "permutation sign-flip", "note": "N=0"}
    abs_mean = abs(np.mean(d))
    # Número de permutações exatas possíveis (sign-flips)
    total = 2 ** n
    rng = np.random.RandomState(seed)
    if total <= reps_max:
        # exato
        cnt = 0
        for mask in range(total):
            signs = np.where(((np.arange(n) >> 0) & 0) == 0, 1.0, 1.0)  # dummy init
            # construir sinais de forma eficiente:
            s = np.fromiter(((1 if ((mask >> k) & 1) == 1 else -1) for k in range(n)), dtype=float, count=n)
            m = abs(np.mean(s * d))
            if m >= abs_mean - 1e-15:
                cnt += 1
        p = min(1.0, cnt / total)
        return {"p": float(p), "p_fmt": fmt_p(float(p)), "method": f"permutation sign-flip (exato, {total} perms)"}
    else:
        # Monte Carlo
        cnt = 0
        for _ in range(reps_max):
            s = rng.choice([-1.0, 1.0], size=n)
            m = abs(np.mean(s * d))
            if m >= abs_mean - 1e-15:
                cnt += 1
        p = min(1.0, cnt / reps_max)
        return {"p": float(p), "p_fmt": fmt_p(float(p)),
                "method": f"permutation sign-flip (Monte Carlo, {reps_max} perms)"}


# ======= Temperature Scaling (pós-hoc em probabilidades) =======
def temperature_scale_probs(probs, T=1.0, eps=1e-12):
    if T == 1.0: return probs
    p = np.clip(probs, eps, 1.0)
    pT = p ** (1.0 / float(T))
    pT = pT / np.sum(pT, axis=1, keepdims=True)
    return pT


def _nll_from_probs(p, y_true):
    p = np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return -float(np.mean(np.log(p)))


def fit_temperature_grid(probs_cal, y_cal, grid=None):
    if grid is None:
        grid = [0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
    best = {"T": 1.0, "nll": _nll_from_probs(probs_cal, y_cal)}
    for T in grid:
        pT = temperature_scale_probs(probs_cal, T)
        nll = _nll_from_probs(pT, y_cal)
        if nll < best["nll"]:
            best = {"T": float(T), "nll": float(nll)}
    return best["T"]


# ======================
# ENSEMBLE helpers
# ======================
def _train_one(seed, input_dim, n_classes, hidden_size, activation, reg,
               X_fit, Y_fit_soft, sample_weight):
    elm = ELM(input_dim, int(hidden_size), n_classes,
              activation=str(activation), reg=float(reg),
              random_state=seed)
    elm.fit(X_fit, Y_fit_soft, sample_weight=sample_weight)
    return elm


def treinar_ensemble(X_fit, y_fit_int, Y_fit_soft, n_classes, best_cfg, class_weights=None,
                     n_modelos=N_MODELOS_ENSEMBLE, seed_base=1000, n_jobs=1):
    sample_weight = np.array([class_weights[c] for c in y_fit_int], dtype=float) if class_weights is not None else None
    seeds = [seed_base + k for k in range(n_modelos)]
    with parallel_backend("loky"):
        modelos = Parallel(n_jobs=_effective_n_jobs(n_jobs))(
            delayed(_train_one)(seed, X_fit.shape[1], n_classes,
                                best_cfg["hidden_size"], best_cfg["activation"], best_cfg["reg"],
                                X_fit, Y_fit_soft, sample_weight)
            for seed in seeds
        )
    return modelos


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def ensemble_predict(modelos, X, mode="prob"):
    if mode == "prob":
        with parallel_backend("loky"):
            preds = Parallel(n_jobs=min(len(modelos), _effective_n_jobs(-1)))(
                delayed(lambda m, X_: m.predict_proba(X_))(m, X) for m in modelos
            )
        return np.mean(preds, axis=0)
    else:  # logit mean
        with parallel_backend("loky"):
            logits = Parallel(n_jobs=min(len(modelos), _effective_n_jobs(-1)))(
                delayed(lambda m, X_: m.predict_logits(X_))(m, X) for m in modelos
            )
        return _softmax(np.mean(logits, axis=0))


# ======================
# PCA helper
# ======================
def fit_transform_pca(X_train, X_test, pca_dim: int, random_state=RANDOM_STATE_BASE):
    if not pca_dim or pca_dim <= 0:
        return X_train, X_test
    pca_dim = int(min(pca_dim, X_train.shape[1] - 1))  # garante válido
    if pca_dim <= 0:
        return X_train, X_test
    svd = TruncatedSVD(n_components=int(pca_dim), random_state=random_state)
    Xtr = svd.fit_transform(X_train);
    Xte = svd.transform(X_test)
    return Xtr.astype("float32"), Xte.astype("float32")


# ======================
# LABEL SMOOTHING
# ======================
def make_soft_targets(y_int, n_classes, eps=LABEL_SMOOTH_EPS):
    if n_classes <= 1:
        Y = np.ones((len(y_int), max(1, n_classes)), dtype="float32")
        return Y
    Y = np.full((len(y_int), n_classes), eps / (n_classes - 1), dtype="float32")
    Y[np.arange(len(y_int)), y_int] = 1.0 - eps
    return Y


# ======================
# Hierarquia + Fusão calibrável (ELM/PG)
# ======================
def build_map_interv_to_pg(df_train, alvo=ALVO, guia=ALVO_GUIA):
    tab = (df_train[[alvo, guia]].astype(str).value_counts().reset_index(name="count"))
    mapa = {}
    for interv, grp in tab.groupby(alvo):
        plano = grp.sort_values("count", ascending=False).iloc[0][guia]
        mapa[str(interv)] = str(plano)
    return mapa


def _softmax_logits(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits);
    return exp / np.sum(exp, axis=1, keepdims=True)


def temperature_scale(probs, T=1.0, eps=1e-12):
    return temperature_scale_probs(probs, T, eps)


def prior_blend(probs, priors, gamma=0.0):
    if gamma <= 0: return probs
    return (1.0 - float(gamma)) * probs + float(gamma) * priors.reshape(1, -1)


def _logit_adjust(probs, class_priors, tau=0.0, eps=1e-12):
    if tau <= 0: return probs
    p = np.clip(probs, eps, 1.0);
    logits = np.log(p)
    logits_adj = logits - float(tau) * np.log(np.clip(class_priors, eps, 1.0))
    return _softmax_logits(logits_adj)


def shrink_uniform(probs, eps_u=0.0):
    if eps_u <= 0: return probs
    nC = probs.shape[1]
    uni = np.full((1, nC), 1.0 / nC, dtype=probs.dtype)
    return (1.0 - float(eps_u)) * probs + float(eps_u) * uni


def fuse_with_pg(P_interv, labels_interv, P_pg, labels_pg, mapa_interv_to_pg,
                 alpha=1.0, lam=0.5, tau=0.0, class_priors=None,
                 T_int=1.0, gamma=0.0, eps_u=0.0):
    P_int = temperature_scale(P_interv, T=T_int)
    P_int = _logit_adjust(P_int, class_priors, tau=tau) if class_priors is not None else P_int
    P_int = prior_blend(P_int, class_priors, gamma=gamma) if class_priors is not None else P_int
    P_int = shrink_uniform(P_int, eps_u=eps_u)

    idx_pg = {pg: i for i, pg in enumerate(labels_pg)}
    mult = P_int.copy()
    for j, interv in enumerate(labels_interv):
        pg = mapa_interv_to_pg.get(str(interv), None)
        if pg is not None and pg in idx_pg:
            mult[:, j] = mult[:, j] * (P_pg[:, idx_pg[pg]] ** float(alpha))
    s = mult.sum(axis=1, keepdims=True);
    s[s == 0] = 1.0
    mult = mult / s
    return (1.0 - float(lam)) * P_int + float(lam) * mult


def fuse_with_pg_soft(P_interv, labels_interv, P_pg, labels_pg, M_interv_given_pg,
                      alpha=1.0, lam=0.5, tau=0.0, class_priors=None,
                      T_int=1.0, gamma=0.0, eps_u=0.0):
    # 1) calibrar ELM (igual à versão hard)
    P_int = temperature_scale(P_interv, T=T_int)
    P_int = _logit_adjust(P_int, class_priors, tau=tau) if class_priors is not None else P_int
    P_int = prior_blend(P_int, class_priors, gamma=gamma) if class_priors is not None else P_int
    P_int = shrink_uniform(P_int, eps_u=eps_u)

    # 2) PG soft → “afinidade” por intervenção via soma sobre todos os PGs
    #    S = (P_pg @ M) ** alpha
    S = np.clip(P_pg @ M_interv_given_pg, 1e-12, 1.0)
    S = S ** float(alpha)

    # 3) mistura multiplicativa e convexa (mesma lógica anterior)
    mult = P_int * S
    s = mult.sum(axis=1, keepdims=True);
    s[s == 0] = 1.0
    mult = mult / s
    return (1.0 - float(lam)) * P_int + float(lam) * mult


# ======================
# Pós-processamento: bias por classe
# ======================

def _match_scales_to_P(P, scales):
    s = np.asarray(scales, dtype=float).ravel()
    C = P.shape[1]
    if s.size == C:
        return s
    out = np.ones(C, dtype=float)
    out[:min(C, s.size)] = s[:min(C, s.size)]
    return out


def apply_class_bias(P, scales):
    s = _match_scales_to_P(P, scales)
    P2 = P * s.reshape(1, -1)
    P2 = P2 / np.clip(np.sum(P2, axis=1, keepdims=True), 1e-12, None)
    return P2


def tune_class_bias(P_cal, y_cal, n_classes, grid_scales, n_rounds=2, metric="bacc"):
    C = P_cal.shape[1]
    scales = np.ones((C,), dtype=float)
    if metric == "accuracy":
        best = accuracy_score(y_cal, np.argmax(apply_class_bias(P_cal, scales), axis=1))
        scorer = accuracy_score
    else:
        best = balanced_accuracy_score(y_cal, np.argmax(apply_class_bias(P_cal, scales), axis=1))
        scorer = balanced_accuracy_score
    for _ in range(int(max(1, n_rounds))):
        improved = False
        for c in range(C):
            best_local = scales[c];
            best_local_score = best
            for s in grid_scales:
                tmp = scales.copy();
                tmp[c] = float(s)
                y_pred = np.argmax(apply_class_bias(P_cal, tmp), axis=1)
                sc = scorer(y_cal, y_pred)
                if sc > best_local_score + 1e-6:
                    best_local_score = sc;
                    best_local = float(s)
            if best_local != scales[c]:
                scales[c] = best_local;
                best = best_local_score;
                improved = True
        if not improved: break
    return scales, best


# ======================
# TF-IDF helper
# ======================
def python_min(a, b): return a if a < b else b


def fit_tfidf_svd(train_texts, test_texts, dim=256, min_df=2, max_df=0.95, ngram_max=2, random_state=RANDOM_STATE_BASE):
    if dim is None or dim <= 0:
        return None, None
    vect = TfidfVectorizer(ngram_range=(1, int(ngram_max)), min_df=int(min_df), max_df=float(max_df), lowercase=True)
    Xtr_tfidf = vect.fit_transform(train_texts)
    Xte_tfidf = vect.transform(test_texts)
    if dim < Xtr_tfidf.shape[1]:
        svd = TruncatedSVD(n_components=int(python_min(dim, Xtr_tfidf.shape[1] - 1)), random_state=random_state)
        Xtr = svd.fit_transform(Xtr_tfidf);
        Xte = svd.transform(Xte_tfidf)
    else:
        Xtr = Xtr_tfidf.toarray();
        Xte = Xte_tfidf.toarray()
    return Xtr.astype("float32"), Xte.astype("float32")


# ======================
# Regras por Combinação (fold-safe)
# ======================
def _tuple_from_row(row, cols):
    return tuple(str(row[c]) for c in cols)


def build_combo_tables(df_train: pd.DataFrame, alvo=ALVO, cat_cols=CATEG_COLS, le_y: LabelEncoder = None):
    y = le_y.transform(df_train[alvo].astype(str))
    n_classes = len(le_y.classes_)
    priors = np.bincount(y, minlength=n_classes).astype(np.float64)
    priors = priors / priors.sum()

    combo_counts = {}
    for _, row in df_train.iterrows():
        combo = _tuple_from_row(row, cat_cols)
        c = int(le_y.transform([str(row[alvo])])[0])
        combo_counts.setdefault(combo, np.zeros(n_classes, dtype=np.float64))
        combo_counts[combo][c] += 1.0

    alpha = 1.0
    combo_probs = {}
    for combo, cnt in combo_counts.items():
        p = (cnt + alpha) / (cnt.sum() + alpha * n_classes)
        combo_probs[combo] = p

    guia_counts = {}
    for pg, gdf in df_train.groupby(ALVO_GUIA):
        y_pg = le_y.transform(gdf[alvo].astype(str))
        cnt = np.bincount(y_pg, minlength=n_classes).astype(np.float64)
        guia_counts[pg] = (cnt + alpha) / (cnt.sum() + alpha * n_classes)

    return combo_probs, guia_counts, priors


def predict_combo(df_part: pd.DataFrame, combo_probs, guia_probs, priors, le_y: LabelEncoder, cat_cols=CATEG_COLS):
    n = len(df_part);
    n_classes = len(le_y.classes_)
    P = np.zeros((n, n_classes), dtype=np.float32)
    seen_flags = np.zeros(n, dtype=np.int32)
    for i, (_, row) in enumerate(df_part.iterrows()):
        combo = _tuple_from_row(row, cat_cols)
        if combo in combo_probs:
            P[i, :] = combo_probs[combo];
            seen_flags[i] = 1
        else:
            pg = str(row[ALVO_GUIA])
            if pg in guia_probs:
                P[i, :] = guia_probs[pg]
            else:
                P[i, :] = priors
    return P, seen_flags


# ======================
# BLEND helpers
# ======================
def convex_grid_weights(n_models, step=0.1):
    if n_models == 1:
        yield np.array([1.0], dtype=float);
        return
    steps = int(round(1.0 / step))

    def rec(curr, k, remaining):
        if k == n_models - 1:
            yield np.array(curr + [remaining * step], dtype=float)
        else:
            for i in range(remaining + 1):
                yield from rec(curr + [i * step], k + 1, remaining - i)

    for w in rec([], 0, steps):
        if abs(sum(w) - 1.0) < 1e-8: yield w


def blend_calibrate(P_list_cal, y_cal, metric="bacc", step=0.1):
    n_models = len(P_list_cal)
    best = {"score": -np.inf, "weights": [1.0] + [0.0] * (n_models - 1)}
    for w in convex_grid_weights(n_models, step=step):
        P = np.zeros_like(P_list_cal[0])
        for wi, Pi in zip(w, P_list_cal):
            P += wi * Pi
        y_pred = np.argmax(P, axis=1)
        sc = balanced_accuracy_score(y_cal, y_pred) if metric == "bacc" else accuracy_score(y_cal, y_pred)
        if sc > best["score"]:
            best = {"score": sc, "weights": w.tolist()}
    return np.array(best["weights"], dtype=float), best["score"]


# ======================
# MAIN
# ======================
def run_pipeline(args):
    t0_all = time.time()
    set_num_threads(args.blas_threads)

    # Caminhos
    csv_original = resolve_path(args.csv_original, "matriz_caa_ajustada.csv")
    try:
        csv_sbert = resolve_path(args.csv_sbert, "base_embeddings_sbert.csv");
        _sbert_exists = True
    except FileNotFoundError:
        csv_sbert = os.path.join(os.path.dirname(csv_original), "base_embeddings_sbert.csv");
        _sbert_exists = False

    if _sbert_exists:
        try:
            tmp = read_csv_auto(csv_sbert).head(1)
            if not any([c.startswith(args.emb_prefix) for c in tmp.columns]): _sbert_exists = False
        except Exception:
            _sbert_exists = False
    if not _sbert_exists:
        print("[AUTO-BUILD] Gerando base SBERT...");
        build_sbert_base(csv_original, csv_sbert, args.sbert_model,
                         batch_size=args.sbert_batch, device_flag=args.device)

    out_root = os.path.join(os.path.dirname(csv_sbert), "resultado_v98_combo_blend")
    os.makedirs(out_root, exist_ok=True)

    df_all, X_all_raw, y_text_all, groups_all, emb_cols, freq_cols, texts_all = load_precomputed_base(
        csv_sbert, emb_prefix=args.emb_prefix)

    # Labels
    le_y = LabelEncoder();
    y_int_all = le_y.fit_transform(y_text_all)
    classes_interv = le_y.classes_;
    n_classes = len(classes_interv)
    le_pg = LabelEncoder();
    y_pg_all = le_pg.fit_transform(df_all[ALVO_GUIA].astype(str).values)
    classes_pg = le_pg.classes_;
    n_pg = len(classes_pg)

    # CV
    if SGKF_AVAILABLE:
        print("[CV] StratifiedGroupKFold (outer/inner).")
        OuterCV = _SGKF_Avail(n_splits=args.outer_folds, shuffle=True, random_state=RANDOM_STATE_BASE)
        InnerCV = _SGKF_Avail(n_splits=args.inner_folds, shuffle=True, random_state=RANDOM_STATE_BASE)
    else:
        print("[CV] GroupKFold (outer/inner).")
        OuterCV = GroupKFold(n_splits=args.outer_folds);
        InnerCV = GroupKFold(n_splits=args.inner_folds)

    outer_train_accs, outer_test_accs_final, outer_test_accs_plain = [], [], []
    # Acurácias de comparadores (Test)
    outer_test_accs_svm, outer_test_accs_knn, outer_test_accs_dt = [], [], []
    # McNemar (por fold) FINAL vs comparadores
    mcn_b_svm_list, mcn_c_svm_list = [], []
    mcn_b_knn_list, mcn_c_knn_list = [], []
    mcn_b_dt_list, mcn_c_dt_list = [], []

    outer_val_bacc_means = []
    outer_test_topk = {k: [] for k in TOPK_LIST}
    global_cfg_scores = {}
    chosen_fusion_params = []
    chosen_ens_mode = []  # prob / logit / mix@eta
    chosen_stack_weights = []
    combo_seen_rates = []
    blend_weights_list = []

    # Novos coletores: tempos por fold
    train_times, test_times = [], []
    infer_samples_per_sec = []

    # Brier/ECE por fold (treino e teste) e para diagramas
    fold_brier_train, fold_brier_test = [], []
    fold_ece_train, fold_ece_test = [], []
    all_test_probs, all_test_true = [], []
    all_train_probs, all_train_true = [], []

    # Métricas adicionais agregadas por fold (test)
    fold_metrics_train = []
    fold_metrics_test = []

    # --- NOVO: coletores por método p/ gráficos comparativos ---
    methods_all = ["FINAL", "PLAIN", "SVM", "KNN", "DT"]
    train_times_by_method = {m: [] for m in methods_all}
    test_times_by_method = {m: [] for m in methods_all}
    fold_metrics_train_by_method = {m: [] for m in methods_all}
    fold_metrics_test_by_method = {m: [] for m in methods_all}
    all_test_probs_by_method = {m: [] for m in methods_all}
    y_test_true_by_method = {m: [] for m in methods_all}

    # McNemar agregado
    mcn_b_list, mcn_c_list = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(OuterCV.split(df_all, y_int_all, groups_all), start=1):
        fold_dir = os.path.join(out_root, f"fold_{fold_idx}")
        grid_dir = os.path.join(fold_dir, "grid")
        treino_dir = os.path.join(fold_dir, "treino")
        teste_dir = os.path.join(fold_dir, "teste")

        for d in [fold_dir, grid_dir, treino_dir, teste_dir]: os.makedirs(d, exist_ok=True)

        # Split
        X_tr_all = X_all_raw[tr_idx].copy();
        X_te_all = X_all_raw[te_idx].copy()
        y_tr = y_int_all[tr_idx];
        y_te = y_int_all[te_idx]
        ypg_tr = y_pg_all[tr_idx];
        ypg_te = y_pg_all[te_idx]
        df_tr = df_all.iloc[tr_idx].copy();
        df_te = df_all.iloc[te_idx].copy()
        groups_tr = groups_all[tr_idx]
        texts_tr = texts_all[tr_idx];
        texts_te = texts_all[te_idx]

        # >>> INSERT: Re-encoder *_freq por fold (sem vazamento)
        # Recalcula frequências por coluna categórica com base APENAS no TREINO
        def _freq_from_train(col_tr, col_te):
            vc = col_tr.astype(str).value_counts(dropna=False)
            freq = (vc / vc.sum()).to_dict()
            ftr = col_tr.astype(str).map(freq).astype("float32")
            fte = col_te.astype(str).map(freq).fillna(0.0).astype("float32")
            return ftr.values.reshape(-1, 1), fte.values.reshape(-1, 1)

        freq_tr_list, freq_te_list = [], []
        used_cats = []
        for c in CATEG_COLS:
            if c in df_tr.columns:
                ftr, fte = _freq_from_train(df_tr[c], df_te[c])
                freq_tr_list.append(ftr);
                freq_te_list.append(fte)
                used_cats.append(c)

        if freq_tr_list:
            X_tr_freq = np.hstack(freq_tr_list).astype("float32")
            X_te_freq = np.hstack(freq_te_list).astype("float32")
        else:
            X_tr_freq = np.empty((len(df_tr), 0), dtype="float32")
            X_te_freq = np.empty((len(df_te), 0), dtype="float32")

        # Extrai SOMENTE os embeddings SBERT do X_all_raw (estão no final)
        n_emb = len(emb_cols)
        X_tr_emb = X_all_raw[tr_idx][:, -n_emb:].astype("float32")
        X_te_emb = X_all_raw[te_idx][:, -n_emb:].astype("float32")

        # Reconstrói X_tr_all / X_te_all sem usar *_freq pré-computados do CSV
        X_tr_all = np.hstack([X_tr_freq, X_tr_emb]).astype("float32")
        X_te_all = np.hstack([X_te_freq, X_te_emb]).astype("float32")

        # Guardamos o número de colunas de frequência DO FOLD para o scaler
        d_freq_fold = X_tr_freq.shape[1]

        # TF-IDF -> SVD -> concat (ON por padrão)
        if args.tfidf_dim and args.tfidf_dim > 0:
            Xtr_tfidf, Xte_tfidf = fit_tfidf_svd(
                texts_tr, texts_te, dim=int(args.tfidf_dim),
                min_df=max(1, int(args.tfidf_min_df)),
                max_df=float(args.tfidf_max_df),
                ngram_max=int(args.tfidf_ngram_max),
                random_state=RANDOM_STATE_BASE + fold_idx
            )
            X_tr_all = np.hstack([X_tr_all, Xtr_tfidf]).astype("float32")
            X_te_all = np.hstack([X_te_all, Xte_tfidf]).astype("float32")

        # Escalonamento *_freq + peso adicional
        if d_freq_fold > 0:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr_all[:, :d_freq_fold])
            X_tr_all[:, :d_freq_fold] = scaler.transform(X_tr_all[:, :d_freq_fold]) * float(args.freq_weight)
            X_te_all[:, :d_freq_fold] = scaler.transform(X_te_all[:, :d_freq_fold]) * float(args.freq_weight)

        # PCA (opcional)
        if args.pca_dim and args.pca_dim > 0:
            X_tr_full, X_te = fit_transform_pca(X_tr_all, X_te_all, args.pca_dim, random_state=RANDOM_STATE_BASE)
            pca_info = {"use_pca": True, "dim": int(args.pca_dim)}
        else:
            X_tr_full, X_te = X_tr_all, X_te_all
            pca_info = {"use_pca": False, "dim": 0}

        # Soft targets
        Y_tr_soft = make_soft_targets(y_tr, n_classes, eps=LABEL_SMOOTH_EPS)

        # ======== GRID (ELM Intervenção) ========
        GRID_HIDDEN_SIZES = args.hidden_sizes
        GRID_ACTIVATIONS = args.activations
        GRID_REGS = args.regs
        resultados = [];
        inner_splits = list(InnerCV.split(df_tr, y_tr, groups_tr))
        cfgs = [(h, act, reg) for h in GRID_HIDDEN_SIZES for act in GRID_ACTIVATIONS for reg in GRID_REGS]
        with parallel_backend("loky"):
            out = Parallel(n_jobs=_effective_n_jobs(args.n_jobs))(
                delayed(_eval_config_one)(h, act, reg, X_tr_full, Y_tr_soft, y_tr, inner_splits, TOPK_LIST)
                for (h, act, reg) in cfgs
            )
        for result, _stats in out: resultados.append(result)

        df_grid = pd.DataFrame(resultados)
        if df_grid.empty:
            raise RuntimeError("Grid interno vazio — verifique se há splits válidos no inner CV.")

        if "val_top3_mean" in df_grid.columns:
            df_grid["score_sel"] = df_grid["val_bacc_mean"] + 0.2 * df_grid["val_top3_mean"]
        else:
            df_grid["score_sel"] = df_grid["val_bacc_mean"]

        df_grid = df_grid.sort_values(by=["score_sel", "val_bacc_mean"], ascending=False)
        df_grid.to_csv(os.path.join(grid_dir, "grid_results_inner_interv.csv"), sep=";", index=False,
                       encoding="utf-8-sig")

        # top-K configs + pesos com temperatura
        topK = int(max(1, min(args.inner_top_k, len(df_grid))))
        top_cfg_rows = df_grid.iloc[:topK].to_dict(orient="records")
        score_series = df_grid.iloc[:topK]["score_sel"].fillna(df_grid.iloc[:topK]["val_bacc_mean"]).astype(float)
        score_arr = score_series.to_numpy(dtype=float)
        score_arr = score_arr - np.min(score_arr);
        score_arr = np.maximum(score_arr, 1e-8)
        logits = score_arr / max(1e-8, float(args.sel_temp));
        logits = logits - np.max(logits)
        w = np.exp(logits);
        stack_weights = w / np.sum(w)
        chosen_stack_weights.append(stack_weights.tolist())

        best_row = top_cfg_rows[0]
        for _, r in df_grid.iterrows():
            key = (int(r["hidden_size"]), str(r["activation"]), float(r["reg"]))
            global_cfg_scores.setdefault(key, []).append(float(r["val_bacc_mean"]))
        with open(os.path.join(grid_dir, "best_config_interv.json"), "w", encoding="utf-8") as f:
            json.dump({"best_by": "score_sel", **best_row, **pca_info, "topK_used": topK,
                       "stack_weights": stack_weights.tolist()}, f, ensure_ascii=False, indent=2)

        # PlanoGuia (único)
        H_PG = max(min(int(best_row["hidden_size"]), 800), 200)
        cfg_pg = {"hidden_size": int(H_PG), "activation": "gelu", "reg": float(min(0.01, best_row["reg"]))}

        # Pesos de classe
        uniq, cnt = np.unique(y_tr, return_counts=True)
        class_weights = {c: (len(y_tr) / (len(uniq) * cnt_i)) for c, cnt_i in dict(zip(uniq, cnt)).items()}
        uniq_pg, cnt_pg = np.unique(ypg_tr, return_counts=True)
        class_weights_pg = {c: (len(ypg_tr) / (len(uniq_pg) * cnt_i)) for c, cnt_i in
                            dict(zip(uniq_pg, cnt_pg)).items()}

        # ======= Treino =======
        t_train0 = time.perf_counter()
        # Intervenção — empilhado topK
        t_train_plain0 = time.perf_counter()  # --- NOVO: tempo de treino PLAIN (ELM empilhado)
        modelos_list = []
        n_each = max(1, args.ensemble_n // topK)
        for k_idx, cfg in enumerate(top_cfg_rows):
            modelos_k = treinar_ensemble(
                X_tr_full, y_tr, make_soft_targets(y_tr, n_classes, LABEL_SMOOTH_EPS),
                n_classes, cfg, class_weights,
                n_modelos=n_each, seed_base=1230 + fold_idx * 100 + 13 * k_idx, n_jobs=args.n_jobs
            )
            modelos_list.append(modelos_k)
        t_train_plain1 = time.perf_counter()  # --- NOVO
        train_times_by_method["PLAIN"].append(t_train_plain1 - t_train_plain0)  # --- NOVO

        # PlanoGuia (único)
        modelos_pg = treinar_ensemble(
            X_tr_full, ypg_tr, make_soft_targets(ypg_tr, n_pg, LABEL_SMOOTH_EPS),
            n_pg, cfg_pg, class_weights_pg,
            n_modelos=max(10, args.ensemble_n // 2), seed_base=777 + fold_idx * 50, n_jobs=args.n_jobs
        )
        t_train1 = time.perf_counter()
        train_times.append(t_train1 - t_train0)
        train_times_by_method["FINAL"].append(t_train1 - t_train0)  # --- NOVO: tempo de treino FINAL

        # ======= Predições ELM/PG =======
        def stacked_predict(modelos_list, X, mode, weights=None):
            Ps = [ensemble_predict(m, X, mode=mode) for m in modelos_list]  # [K][n,C]
            Ps = np.stack(Ps, axis=0)  # [K, n, C]
            if weights is None: return np.mean(Ps, axis=0)
            w = weights.reshape(-1, 1, 1);
            return np.sum(w * Ps, axis=0)

        # ======= Fusão ELM com PlanoGuia calibrada =======
        mapa_interv_to_pg = build_map_interv_to_pg(df_tr, alvo=ALVO, guia=ALVO_GUIA)
        priors = np.bincount(y_tr, minlength=n_classes).astype(np.float64);
        priors = priors / priors.sum()

        test_size = min(max(0.1, float(args.calib_holdout)), 0.3)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE_BASE + fold_idx)
        calib_idx = next(gss.split(X_tr_full, y_tr, groups_tr))[1]
        y_calib = y_tr[calib_idx]

        # Predições para calibração (sem bias, sem blend)
        P_tr_pg = ensemble_predict(modelos_pg, X_tr_full, mode="prob")
        P_tr_int_prob = stacked_predict(modelos_list, X_tr_full, mode="prob", weights=stack_weights)
        P_tr_int_logit = stacked_predict(modelos_list, X_tr_full, mode="logit", weights=stack_weights)
        P_tr_int_modes = {"prob": P_tr_int_prob, "logit": P_tr_int_logit}
        for eta in [float(x) for x in args.mix_etas]:
            P_tr_int_modes[f"mix@{eta:.2f}"] = eta * P_tr_int_prob + (1.0 - eta) * P_tr_int_logit

        P_calib_pg = P_tr_pg[calib_idx]
        P_calib_int_modes = {m: P_tr_int_modes[m][calib_idx] for m in P_tr_int_modes.keys()}

        # ======= Regras-Combo (fold-safe) + matriz M (PG→Intervenção) =======
        combo_probs, guia_probs, priors_vec = build_combo_tables(df_tr, alvo=ALVO, cat_cols=CATEG_COLS, le_y=le_y)

        M = np.zeros((len(classes_pg), len(classes_interv)), dtype=float)
        for i, pg in enumerate(classes_pg):
            M[i, :] = guia_probs.get(pg, np.full(len(classes_interv), 1.0 / len(classes_interv), dtype=float))

        grid_alpha = [0.75, 1.0, 1.5, 2.0]
        grid_lambda = [0.20, 0.40, 0.60, 0.80, 1.0]
        grid_tau = [0.0, 0.25, 0.5]
        grid_T = [0.80, 1.0, 1.20]
        grid_gamma = [0.0, 0.05, 0.10, 0.15]
        grid_epsu = [0.0, 0.02, 0.05]

        best_cal = {"bacc": -np.inf}
        for m in list(P_tr_int_modes.keys()):
            for a in grid_alpha:
                for l in grid_lambda:
                    for t in grid_tau:
                        for T_ in grid_T:
                            for g in grid_gamma:
                                for eU in grid_epsu:
                                    # hard (atual)
                                    P_cal_hard = fuse_with_pg(P_calib_int_modes[m], classes_interv,
                                                              P_calib_pg, classes_pg,
                                                              mapa_interv_to_pg,
                                                              alpha=a, lam=l, tau=t, class_priors=priors,
                                                              T_int=T_, gamma=g, eps_u=eU)
                                    bacc_hard = balanced_accuracy_score(y_calib, np.argmax(P_cal_hard, axis=1))

                                    # soft (novo)
                                    P_cal_soft = fuse_with_pg_soft(P_calib_int_modes[m], classes_interv,
                                                                   P_calib_pg, classes_pg, M,
                                                                   alpha=a, lam=l, tau=t, class_priors=priors,
                                                                   T_int=T_, gamma=g, eps_u=eU)
                                    bacc_soft = balanced_accuracy_score(y_calib, np.argmax(P_cal_soft, axis=1))

                                    if bacc_soft > best_cal["bacc"]:
                                        best_cal = {"bacc": bacc_soft, "alpha": a, "lambda": l, "tau": t, "T": T_,
                                                    "gamma": g, "eps_u": eU, "mode": m, "pg_mode": "soft"}
                                    if bacc_hard > best_cal["bacc"]:
                                        best_cal = {"bacc": bacc_hard, "alpha": a, "lambda": l, "tau": t, "T": T_,
                                                    "gamma": g, "eps_u": eU, "mode": m, "pg_mode": "hard"}

        chosen_fusion_params.append(
            {k: best_cal[k] for k in ["alpha", "lambda", "tau", "T", "gamma", "eps_u", "pg_mode"]})
        chosen_ens_mode.append(best_cal["mode"])

        # Probabilidades TREINO para fusão e viés
        P_tr_int = P_tr_int_modes[best_cal["mode"]]
        if best_cal.get("pg_mode", "hard") == "soft":
            P_tr_fused = fuse_with_pg_soft(
                P_tr_int, classes_interv,
                P_tr_pg, classes_pg, M,
                alpha=best_cal["alpha"], lam=best_cal["lambda"],
                tau=best_cal["tau"], class_priors=priors, T_int=best_cal["T"],
                gamma=best_cal["gamma"], eps_u=best_cal["eps_u"]
            )
        else:
            P_tr_fused = fuse_with_pg(
                P_tr_int, classes_interv,
                P_tr_pg, classes_pg, mapa_interv_to_pg,
                alpha=best_cal["alpha"], lam=best_cal["lambda"],
                tau=best_cal["tau"], class_priors=priors, T_int=best_cal["T"],
                gamma=best_cal["gamma"], eps_u=best_cal["eps_u"]
            )

        # ======= Classificador Regras-Combo (fold-safe) =======
        combo_probs, guia_probs, priors_vec = build_combo_tables(df_tr, alvo=ALVO, cat_cols=CATEG_COLS, le_y=le_y)
        # matriz M[pg_idx, interv_idx] = P(intervenção | PG=labels_pg[pg_idx])
        M = np.zeros((len(classes_pg), len(classes_interv)), dtype=float)
        for i, pg in enumerate(classes_pg):
            # guia_probs[pg] já vem na ordem de classes do le_y
            M[i, :] = guia_probs.get(pg, np.full(len(classes_interv), 1.0 / len(classes_interv), dtype=float))

        P_rule_train, _seen_tr = predict_combo(df_tr, combo_probs, guia_probs, priors_vec, le_y, CATEG_COLS)
        P_rule_test, seen_te = predict_combo(df_te, combo_probs, guia_probs, priors_vec, le_y, CATEG_COLS)
        combo_seen_rate = float(np.mean(seen_te));
        combo_seen_rates.append(combo_seen_rate)

        # ======= TUNING DE VIÉS POR CLASSE (holdout; métrica = bacc) =======
        bias_grid = [float(x) for x in args.class_bias_search]
        scales_elmfused, _ = tune_class_bias(P_tr_fused[calib_idx], y_calib, n_classes, bias_grid,
                                             n_rounds=int(args.class_bias_rounds), metric="bacc")
        P_tr_elmbias = apply_class_bias(P_tr_fused, scales_elmfused)
        scales_rule, _ = tune_class_bias(P_rule_train[calib_idx], y_calib, n_classes, bias_grid,
                                         n_rounds=1, metric="bacc")
        P_tr_rulebias = apply_class_bias(P_rule_train, scales_rule)

        # ======= BLEND (RULE + ELM) calibrado por BACC no holdout =======
        P_list_cal = [P_tr_rulebias[calib_idx], P_tr_elmbias[calib_idx]]
        blend_w, blend_score = blend_calibrate(P_list_cal, y_calib, metric="bacc", step=float(args.blend_grid_step))
        blend_weights_list.append(blend_w.tolist())

        # aplica no treino
        P_tr_final_raw = blend_w[0] * P_tr_rulebias + blend_w[1] * P_tr_elmbias

        # ======= Temperature Scaling (pós-hoc) — fit em holdout =======
        T_final = fit_temperature_grid(P_tr_final_raw[calib_idx], y_calib)
        P_tr_final = temperature_scale_probs(P_tr_final_raw, T_final)

        # --- NOVO: métricas de treino por método ---
        # PLAIN (ELM empilhado, sem PG/bias/blend/TS)
        y_pred_tr_plain = np.argmax(P_tr_int, axis=1)
        m_tr_plain, _ = avaliar_conjunto(y_tr, y_pred_tr_plain, P_tr_int, classes_interv,
                                         treino_dir, "treino_elm_plain", TOPK_LIST)
        fold_metrics_train_by_method["PLAIN"].append(m_tr_plain)

        # ======= TESTE: construir tudo e cronometrar com perf_counter() + repetições =======
        # --- NOVO: tempo de teste PLAIN (ELM empilhado)
        t_plain0 = time.perf_counter()
        P_te_int_prob = stacked_predict(modelos_list, X_te, mode="prob", weights=stack_weights)
        P_te_int_logit = stacked_predict(modelos_list, X_te, mode="logit", weights=stack_weights)
        P_te_int_modes = {"prob": P_te_int_prob, "logit": P_te_int_logit}
        for eta in [float(x) for x in args.mix_etas]:
            P_te_int_modes[f"mix@{eta:.2f}"] = eta * P_te_int_prob + (1.0 - eta) * P_te_int_logit
        P_te_int = P_te_int_modes[best_cal["mode"]]
        y_pred_te_plain = np.argmax(P_te_int, axis=1)
        t_plain1 = time.perf_counter()
        test_times_by_method["PLAIN"].append(t_plain1 - t_plain0)

        t_rep = []

        for _ in range(int(max(1, args.infer_reps))):
            t0 = time.perf_counter()
            P_te_pg = ensemble_predict(modelos_pg, X_te, mode="prob")
            P_te_int_prob = stacked_predict(modelos_list, X_te, mode="prob", weights=stack_weights)
            P_te_int_logit = stacked_predict(modelos_list, X_te, mode="logit", weights=stack_weights)
            P_te_int_modes = {"prob": P_te_int_prob, "logit": P_te_int_logit}
            for eta in [float(x) for x in args.mix_etas]:
                P_te_int_modes[f"mix@{eta:.2f}"] = eta * P_te_int_prob + (1.0 - eta) * P_te_int_logit
            P_te_int = P_te_int_modes[best_cal["mode"]]
            if best_cal.get("pg_mode", "hard") == "soft":
                P_te_fused = fuse_with_pg_soft(
                    P_te_int, classes_interv,
                    P_te_pg, classes_pg, M,
                    alpha=best_cal["alpha"], lam=best_cal["lambda"],
                    tau=best_cal["tau"], class_priors=priors, T_int=best_cal["T"],
                    gamma=best_cal["gamma"], eps_u=best_cal["eps_u"]
                )
            else:
                P_te_fused = fuse_with_pg(
                    P_te_int, classes_interv,
                    P_te_pg, classes_pg, mapa_interv_to_pg,
                    alpha=best_cal["alpha"], lam=best_cal["lambda"],
                    tau=best_cal["tau"], class_priors=priors, T_int=best_cal["T"],
                    gamma=best_cal["gamma"], eps_u=best_cal["eps_u"]
                )
            P_te_elmbias = apply_class_bias(P_te_fused, scales_elmfused)
            P_te_rulebias = apply_class_bias(P_rule_test, scales_rule)
            P_te_final = temperature_scale_probs(blend_w[0] * P_te_rulebias + blend_w[1] * P_te_elmbias, T_final)
            y_pred_te_final = np.argmax(P_te_final, axis=1)
            t1 = time.perf_counter()
            t_rep.append(t1 - t0)

        t_mean = float(np.mean(t_rep));
        t_std = float(np.std(t_rep))
        test_times.append(t_mean)
        infer_samples_per_sec.append(len(X_te) / t_mean if t_mean > 0 else float('inf'))
        test_times_by_method["FINAL"].append(t_mean)  # --- NOVO: tempo de teste FINAL

        # ======= AVALIAÇÕES =======
        # OUTER-TRAIN (com TS)
        y_pred_tr = np.argmax(P_tr_final, axis=1)
        salvar_predicoes(df_tr[ID_COL].values, y_tr, y_pred_tr, P_tr_final, classes_interv,
                         os.path.join(treino_dir, "predicoes_treino_final.csv"))
        m_tr, _ = avaliar_conjunto(y_tr, y_pred_tr, P_tr_final, classes_interv, treino_dir, "treino_final", TOPK_LIST)
        fold_metrics_train.append(m_tr)

        # FINAL
        fold_metrics_train_by_method["FINAL"].append(m_tr)

        # Métricas de calibração (treino)
        brier_tr = brier_score_multiclass(y_tr, P_tr_final, n_classes)
        ece_tr = expected_calibration_error(y_tr, P_tr_final, n_bins=15)
        fold_brier_train.append(brier_tr);
        fold_ece_train.append(ece_tr)
        all_train_probs.append(P_tr_final);
        all_train_true.append(y_tr)

        # OUTER-TEST (com TS)
        salvar_predicoes(df_te[ID_COL].values, y_te, y_pred_te_final, P_te_final, classes_interv,
                         os.path.join(teste_dir, "predicoes_teste_final.csv"))
        m_te_final, cm_te = avaliar_conjunto(y_te, y_pred_te_final, P_te_final, classes_interv, teste_dir,
                                             "teste_final", TOPK_LIST)
        fold_metrics_test.append(m_te_final)

        # --- NOVO: guardar por método (FINAL) ---
        fold_metrics_test_by_method["FINAL"].append(m_te_final)
        all_test_probs_by_method["FINAL"].append(P_te_final)
        y_test_true_by_method["FINAL"].append(y_te)

        # baseline (ELM plain) — sem RULE/PG/bias/TS, apenas melhor modo da pilha
        y_pred_te_plain = np.argmax(P_te_int, axis=1)
        m_te_plain, _ = avaliar_conjunto(y_te, y_pred_te_plain, P_te_int, classes_interv, teste_dir, "teste_elm_plain",
                                         TOPK_LIST)

        # --- NOVO: guardar por método (PLAIN) ---
        fold_metrics_test_by_method["PLAIN"].append(m_te_plain)
        all_test_probs_by_method["PLAIN"].append(P_te_int)
        y_test_true_by_method["PLAIN"].append(y_te)

        # Calibração (teste)
        brier_te = brier_score_multiclass(y_te, P_te_final, n_classes)
        ece_te = expected_calibration_error(y_te, P_te_final, n_bins=15)
        fold_brier_test.append(brier_te);
        fold_ece_test.append(ece_te)
        all_test_probs.append(P_te_final);
        all_test_true.append(y_te)

        for k in TOPK_LIST: outer_test_topk[k].append(topk_accuracy_score(y_te, P_te_final, k))
        outer_train_accs.append(m_tr["accuracy"])
        outer_test_accs_final.append(m_te_final["accuracy"])
        outer_test_accs_plain.append(m_te_plain["accuracy"])
        outer_val_bacc_means.append(float(df_grid.iloc[0]["val_bacc_mean"]))

        # ======= Classificadores adicionais: SVM/KNN/DT (probabilísticos) =======
        # SVM (LinearSVC calibrado com standardization)
        svm_base = make_pipeline(
            StandardScaler(with_mean=True),
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE_BASE + fold_idx)
        )
        svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")

        # --- NOVO: tempo de treino SVM ---
        t0_svm_fit = time.perf_counter()
        svm.fit(X_tr_full, y_tr)
        t1_svm_fit = time.perf_counter()
        train_times_by_method["SVM"].append(t1_svm_fit - t0_svm_fit)

        # --- NOVO: métricas de treino SVM ---
        P_tr_svm_raw = svm.predict_proba(X_tr_full)
        P_tr_svm = _align_proba(P_tr_svm_raw, svm.classes_, n_classes)
        y_pred_tr_svm = np.argmax(P_tr_svm, axis=1)
        m_tr_svm, _ = avaliar_conjunto(y_tr, y_pred_tr_svm, P_tr_svm, classes_interv, treino_dir, "treino_svm",
                                       TOPK_LIST)
        fold_metrics_train_by_method["SVM"].append(m_tr_svm)

        # --- NOVO: tempo de teste SVM ---
        t0_svm_pred = time.perf_counter()
        P_te_svm_raw = svm.predict_proba(X_te)
        t1_svm_pred = time.perf_counter()
        test_times_by_method["SVM"].append(t1_svm_pred - t0_svm_pred)

        P_te_svm = _align_proba(P_te_svm_raw, svm.classes_, n_classes)
        y_pred_te_svm = np.argmax(P_te_svm, axis=1)
        salvar_predicoes(df_te[ID_COL].values, y_te, y_pred_te_svm, P_te_svm, classes_interv,
                         os.path.join(teste_dir, "predicoes_teste_svm.csv"))
        m_te_svm, _ = avaliar_conjunto(y_te, y_pred_te_svm, P_te_svm, classes_interv, teste_dir, "teste_svm", TOPK_LIST)

        # --- NOVO: guardar por método (SVM) ---
        fold_metrics_test_by_method["SVM"].append(m_te_svm)
        all_test_probs_by_method["SVM"].append(P_te_svm)
        y_test_true_by_method["SVM"].append(y_te)

        # KNN (probabilístico) com standardization
        knn = make_pipeline(
            StandardScaler(with_mean=True),
            KNeighborsClassifier(n_neighbors=int(args.knn_k), weights="distance")
        )

        # --- NOVO: tempo de treino KNN ---
        t0_knn_fit = time.perf_counter()
        knn.fit(X_tr_full, y_tr)
        t1_knn_fit = time.perf_counter()
        train_times_by_method["KNN"].append(t1_knn_fit - t0_knn_fit)

        # --- NOVO: métricas de treino KNN ---
        P_tr_knn_raw = knn.predict_proba(X_tr_full)
        knn_classes = knn.named_steps['kneighborsclassifier'].classes_
        P_tr_knn = _align_proba(P_tr_knn_raw, knn_classes, n_classes)
        y_pred_tr_knn = np.argmax(P_tr_knn, axis=1)
        m_tr_knn, _ = avaliar_conjunto(y_tr, y_pred_tr_knn, P_tr_knn, classes_interv, treino_dir, "treino_knn",
                                       TOPK_LIST)
        fold_metrics_train_by_method["KNN"].append(m_tr_knn)

        # --- NOVO: tempo de teste KNN ---
        t0_knn_pred = time.perf_counter()
        P_te_knn_raw = knn.predict_proba(X_te)
        t1_knn_pred = time.perf_counter()
        test_times_by_method["KNN"].append(t1_knn_pred - t0_knn_pred)

        P_te_knn = _align_proba(P_te_knn_raw, knn_classes, n_classes)
        y_pred_te_knn = np.argmax(P_te_knn, axis=1)
        salvar_predicoes(df_te[ID_COL].values, y_te, y_pred_te_knn, P_te_knn, classes_interv,
                         os.path.join(teste_dir, "predicoes_teste_knn.csv"))
        m_te_knn, _ = avaliar_conjunto(y_te, y_pred_te_knn, P_te_knn, classes_interv, teste_dir, "teste_knn", TOPK_LIST)

        # --- NOVO: guardar por método (KNN) ---
        fold_metrics_test_by_method["KNN"].append(m_te_knn)
        all_test_probs_by_method["KNN"].append(P_te_knn)
        y_test_true_by_method["KNN"].append(y_te)

        # Árvore de Decisão (probabilística)
        max_depth_dt = None if int(args.dt_max_depth) <= 0 else int(args.dt_max_depth)
        dt = DecisionTreeClassifier(max_depth=max_depth_dt, class_weight="balanced",
                                    random_state=RANDOM_STATE_BASE + fold_idx)

        # --- NOVO: tempo de treino DT ---
        t0_dt_fit = time.perf_counter()
        dt.fit(X_tr_full, y_tr)
        t1_dt_fit = time.perf_counter()
        train_times_by_method["DT"].append(t1_dt_fit - t0_dt_fit)

        # --- NOVO: métricas de treino DT ---
        P_tr_dt_raw = dt.predict_proba(X_tr_full)
        P_tr_dt = _align_proba(P_tr_dt_raw, dt.classes_, n_classes)
        y_pred_tr_dt = np.argmax(P_tr_dt, axis=1)
        m_tr_dt, _ = avaliar_conjunto(y_tr, y_pred_tr_dt, P_tr_dt, classes_interv, treino_dir, "treino_dt", TOPK_LIST)
        fold_metrics_train_by_method["DT"].append(m_tr_dt)

        # --- NOVO: tempo de teste DT ---
        t0_dt_pred = time.perf_counter()
        P_te_dt_raw = dt.predict_proba(X_te)
        t1_dt_pred = time.perf_counter()
        test_times_by_method["DT"].append(t1_dt_pred - t0_dt_pred)

        P_te_dt = _align_proba(P_te_dt_raw, dt.classes_, n_classes)
        y_pred_te_dt = np.argmax(P_te_dt, axis=1)
        salvar_predicoes(df_te[ID_COL].values, y_te, y_pred_te_dt, P_te_dt, classes_interv,
                         os.path.join(teste_dir, "predicoes_teste_dt.csv"))
        m_te_dt, _ = avaliar_conjunto(y_te, y_pred_te_dt, P_te_dt, classes_interv, teste_dir, "teste_dt", TOPK_LIST)

        # --- NOVO: guardar por método (DT) ---
        fold_metrics_test_by_method["DT"].append(m_te_dt)
        all_test_probs_by_method["DT"].append(P_te_dt)
        y_test_true_by_method["DT"].append(y_te)

        # Coleta por fold
        outer_test_accs_svm.append(m_te_svm["accuracy"])
        outer_test_accs_knn.append(m_te_knn["accuracy"])
        outer_test_accs_dt.append(m_te_dt["accuracy"])

        # McNemar por fold (FINAL vs cada comparador)
        def _mcn_append(pred_other, b_list, c_list):
            final_correct = (y_pred_te_final == y_te)
            other_correct = (pred_other == y_te)
            b_list.append(int(np.sum((~final_correct) & (other_correct))))  # FINAL erra, outro acerta
            c_list.append(int(np.sum((final_correct) & (~other_correct))))  # FINAL acerta, outro erra

        _mcn_append(y_pred_te_svm, mcn_b_svm_list, mcn_c_svm_list)
        _mcn_append(y_pred_te_knn, mcn_b_knn_list, mcn_c_knn_list)
        _mcn_append(y_pred_te_dt, mcn_b_dt_list, mcn_c_dt_list)

        # ====== Pasta de análise do fold ======
        analise_dir = os.path.join(fold_dir, "analise")
        os.makedirs(analise_dir, exist_ok=True)

        # Reliability (train/test) no fold
        plot_reliability_diagram(y_tr, P_tr_final,
                                 os.path.join(analise_dir, "reliability_train.png"),
                                 n_bins=15, title=f"Reliability — Train (fold {fold_idx})")
        plot_reliability_diagram(y_te, P_te_final,
                                 os.path.join(analise_dir, "reliability_test.png"),
                                 n_bins=15, title=f"Reliability — Test (fold {fold_idx})")

        # ROC/PR micro/macro por fold — Train e Test
        _roc_pr_micro_macro(y_tr, P_tr_final,
                            os.path.join(analise_dir, "roc_micro_macro_train.png"),
                            os.path.join(analise_dir, "pr_micro_macro_train.png"),
                            f"Fold {fold_idx} Train")
        _roc_pr_micro_macro(y_te, P_te_final,
                            os.path.join(analise_dir, "roc_micro_macro_test.png"),
                            os.path.join(analise_dir, "pr_micro_macro_test.png"),
                            f"Fold {fold_idx} Test")

        # Barras: P/R/F1 (macro, micro, weighted) — Train e Test
        _bars_triple(m_tr["precision_macro"], m_tr["precision_micro"], m_tr["precision_weighted"],
                     "Precision", f"Fold {fold_idx} — Precision (Train)",
                     os.path.join(analise_dir, "bars_precision_triple_train.png"))
        _bars_triple(m_tr["recall_macro"], m_tr["recall_micro"], m_tr["recall_weighted"],
                     "Recall", f"Fold {fold_idx} — Recall (Train)",
                     os.path.join(analise_dir, "bars_recall_triple_train.png"))
        _bars_triple(m_tr["f1_macro"], m_tr["f1_micro"], m_tr["f1_weighted"],
                     "F1", f"Fold {fold_idx} — F1 (Train)",
                     os.path.join(analise_dir, "bars_f1_triple_train.png"))

        _bars_triple(m_te_final["precision_macro"], m_te_final["precision_micro"], m_te_final["precision_weighted"],
                     "Precision", f"Fold {fold_idx} — Precision (Test)",
                     os.path.join(analise_dir, "bars_precision_triple_test.png"))
        _bars_triple(m_te_final["recall_macro"], m_te_final["recall_micro"], m_te_final["recall_weighted"],
                     "Recall", f"Fold {fold_idx} — Recall (Test)",
                     os.path.join(analise_dir, "bars_recall_triple_test.png"))
        _bars_triple(m_te_final["f1_macro"], m_te_final["f1_micro"], m_te_final["f1_weighted"],
                     "F1", f"Fold {fold_idx} — F1 (Test)",
                     os.path.join(analise_dir, "bars_f1_triple_test.png"))

        # Barras: acurácias
        _bar_plot(
            [m_tr["accuracy"], m_te_final["accuracy"], m_te_plain["accuracy"],
             m_te_svm["accuracy"], m_te_knn["accuracy"], m_te_dt["accuracy"]],
            ["Train FINAL", "Test FINAL", "Test PLAIN", "SVM", "KNN", "DT"],
            f"Acurácias — Fold {fold_idx}",
            "Acurácia",
            os.path.join(analise_dir, "bar_accuracies.png")
        )

        # Barras: tempos
        _bar_plot(
            [train_times[-1], test_times[-1]],
            ["Treino (s)", "Teste (s)"],
            f"Tempos — Fold {fold_idx}",
            "Segundos",
            os.path.join(analise_dir, "bar_tempos.png")
        )

        # CSV resumo de métricas (train/test) por fold
        pd.DataFrame([{
            "fold": fold_idx,
            "train_accuracy": m_tr["accuracy"],
            "train_balanced_accuracy": m_tr["balanced_accuracy"],
            "train_precision_macro": m_tr["precision_macro"],
            "train_precision_micro": m_tr["precision_micro"],
            "train_precision_weighted": m_tr["precision_weighted"],
            "train_recall_macro": m_tr["recall_macro"],
            "train_recall_micro": m_tr["recall_micro"],
            "train_recall_weighted": m_tr["recall_weighted"],
            "train_f1_macro": m_tr["f1_macro"],
            "train_f1_micro": m_tr["f1_micro"],
            "train_f1_weighted": m_tr["f1_weighted"],
            "test_accuracy": m_te_final["accuracy"],
            "test_balanced_accuracy": m_te_final["balanced_accuracy"],
            "test_precision_macro": m_te_final["precision_macro"],
            "test_precision_micro": m_te_final["precision_micro"],
            "test_precision_weighted": m_te_final["precision_weighted"],
            "test_recall_macro": m_te_final["recall_macro"],
            "test_recall_micro": m_te_final["recall_micro"],
            "test_recall_weighted": m_te_final["recall_weighted"],
            "test_f1_macro": m_te_final["f1_macro"],
            "test_f1_micro": m_te_final["f1_micro"],
            "test_f1_weighted": m_te_final["f1_weighted"],
            "tempo_treino_s": train_times[-1],
            "tempo_teste_s": test_times[-1],
            "samples_per_sec": infer_samples_per_sec[-1],
            "brier_train": brier_tr,
            "ece_train": ece_tr,
            "brier_test": brier_te,
            "ece_test": ece_te
        }]).to_csv(os.path.join(analise_dir, "metrics_train_test_summary.csv"),
                   sep=";", index=False, encoding="utf-8-sig")

        # McNemar (pareado) no fold — FINAL vs PLAIN
        final_correct = (y_pred_te_final == y_te)
        plain_correct = (y_pred_te_plain == y_te)
        b_fold = int(np.sum((~final_correct) & (plain_correct)))  # FINAL erra, PLAIN acerta
        c_fold = int(np.sum((final_correct) & (~plain_correct)))  # FINAL acerta, PLAIN erra
        mcn_fold = mcnemar_test(b_fold, c_fold, exact=True)

        # Guardar para agregado
        mcn_b_list.append(b_fold);
        mcn_c_list.append(c_fold)

        # Resumos do fold
        resumo_fold = {
            "fold": fold_idx,
            "acc_train_final": float(m_tr["accuracy"]),
            "acc_test_final": float(m_te_final["accuracy"]),
            "acc_test_plain": float(m_te_plain["accuracy"]),
            "delta_final_minus_plain_acc": float(m_te_final["accuracy"] - m_te_plain["accuracy"]),
            "tempo_treino_s": float(train_times[-1]),
            "tempo_teste_s": float(test_times[-1]),
            "samples_per_sec": float(infer_samples_per_sec[-1]),
            "brier_train": float(brier_tr),
            "ece_train": float(ece_tr),
            "brier_test": float(brier_te),
            "ece_test": float(ece_te),
            "mcnemar_b": int(b_fold),
            "mcnemar_c": int(c_fold),
            "mcnemar_method": mcn_fold["method"],
            "mcnemar_stat": None if mcn_fold["stat"] is None else float(mcn_fold["stat"]),
            "mcnemar_pvalue": float(mcn_fold["pvalue"]),
            "temperature_T_final": float(T_final),
            "blend_weights_rule_elm": blend_w.tolist(),
            "bias_scales_elm": [float(x) for x in scales_elmfused],
            "bias_scales_rule": [float(x) for x in scales_rule],
        }
        _safe_json_dump(resumo_fold, os.path.join(analise_dir, "resumo_fold.json"))
        _save_kv_csv(resumo_fold, os.path.join(analise_dir, "resumo_fold.csv"))

        with open(os.path.join(analise_dir, "estatisticas_fold.txt"), "w", encoding="utf-8") as f_est:
            f_est.write(f"Fold {fold_idx}\n")
            f_est.write(
                f"Acurácias: Train FINAL={m_tr['accuracy']:.4f} | Test FINAL={m_te_final['accuracy']:.4f} | Test PLAIN={m_te_plain['accuracy']:.4f}\n")
            f_est.write(
                f"Tempos (s): Treino={train_times[-1]:.4f} | Teste={test_times[-1]:.4f} | Samples/s={infer_samples_per_sec[-1]:.2f}\n")
            f_est.write(
                f"Calibração: Brier(train/test)={brier_tr:.4f}/{brier_te:.4f} | ECE(train/test)={ece_tr:.4f}/{ece_te:.4f}\n")
            f_est.write(
                f"McNemar (FINAL vs PLAIN): b={b_fold}, c={c_fold} | método={mcn_fold['method']} | estatística={mcn_fold['stat']} | p-valor={fmt_p(mcn_fold['pvalue'])}\n")
            # Interpretação compacta por fold:
            direc = "FINAL corrige mais que PLAIN" if c_fold > b_fold else (
                "PLAIN corrige mais que FINAL" if b_fold > c_fold else "Sem diferença nos discordantes")
            alfa = float(args.alpha)
            decisao = "REJEITA H0" if (
                        mcn_fold["pvalue"] is not None and mcn_fold["pvalue"] <= alfa) else "NÃO REJEITA H0"
            f_est.write(f"Interpretação McNemar: {decisao} a α={alfa:.2f}; direção: {direc}.\n")
            f_est.write(f"Temperature Scaling: T*={T_final:.3f}\n")
            f_est.write("Obs.: t-test/Wilcoxon/Shapiro/Permutação e gráficos agregados no diretório 'agregado/'.\n")

        with open(os.path.join(fold_dir, "relatorio_fold.txt"), "w", encoding="utf-8") as f:
            f.write(f"FOLD {fold_idx} – SBERT v9.8\n")
            f.write(
                f"  PCA={pca_info['dim']} | n_jobs={args.n_jobs} | freq_w={args.freq_weight} | tfidf_dim={args.tfidf_dim}\n")
            f.write(f"  Top-{topK} Interv cfgs:\n")
            for r in top_cfg_rows: f.write("    " + json.dumps(r, ensure_ascii=False) + "\n")
            f.write(f"  Pesos do stack (temp={args.sel_temp}): {stack_weights.tolist()}\n")
            f.write(
                f"  Fusão ELM+PG: {{'alpha':{best_cal['alpha']}, 'lambda':{best_cal['lambda']}, 'tau':{best_cal['tau']}, 'T':{best_cal['T']}, 'gamma':{best_cal['gamma']}, 'eps_u':{best_cal['eps_u']}, 'mode':'{best_cal['mode']}', 'pg_mode':'{best_cal.get('pg_mode', 'hard')}'}}\n")
            f.write(f"  Bias scales ELM: {scales_elmfused.tolist()}\n")
            f.write(f"  Bias scales RULE: {scales_rule.tolist()}\n")
            f.write(f"  Blend weights (RULE, ELM) calibrados BACC: {blend_w.tolist()} (score_cal={blend_score:.4f})\n")
            f.write(f"  Cobertura RULE (combos vistos no teste): {combo_seen_rate * 100:.2f}%\n")
            f.write(
                f"  ACC Treino final={m_tr['accuracy']:.4f} | ACC Teste ELMplain={m_te_plain['accuracy']:.4f} | ACC Teste FINAL={m_te_final['accuracy']:.4f}\n")
            f.write(
                f"  Tempo Treino(s)={train_times[-1]:.4f} | Tempo Teste(s)={test_times[-1]:.4f} | Samples/s={infer_samples_per_sec[-1]:.2f}\n")
            f.write(
                f"  Brier (train/test)={brier_tr:.4f}/{brier_te:.4f} | ECE (train/test)={ece_tr:.4f}/{ece_te:.4f}\n")
            f.write(f"  Temperature Scaling T*={T_final:.3f}\n")

    # ========= Agregado =========
    agg_dir = os.path.join(out_root, "agregado");
    os.makedirs(agg_dir, exist_ok=True)
    methods_dir = os.path.join(agg_dir, "comparacao_metodos")
    os.makedirs(methods_dir, exist_ok=True)

    # === Contagem exata (Nested CV) e esforço de treino ===
    n_outer = len(outer_test_accs_final)
    n_inner = int(args.inner_folds)
    grid_total_cfgs = len(args.hidden_sizes) * len(args.activations) * len(args.regs)
    total_inner_fits = n_outer * n_inner * grid_total_cfgs  # ELMs treinados no inner (grid)

    # Modelos finais (por fold): intervenção = (ensemble_n // topK) * topK; PG = max(10, ensemble_n//2)
    topKs = [len(w) for w in chosen_stack_weights]  # um comprimento por fold
    elm_models_interv = sum((args.ensemble_n // max(1, tk)) * max(1, tk) for tk in topKs)
    elm_models_pg = n_outer * max(10, args.ensemble_n // 2)
    total_elm_models_finais = elm_models_interv + elm_models_pg

    # Comparadores (um treino por outer)
    total_svm_treinos = n_outer
    total_knn_treinos = n_outer
    total_dt_treinos = n_outer

    # melhor cfg global (Intervenção, média val_bacc_mean no inner)
    best_cfg, best_score = None, -np.inf
    for key, vals in global_cfg_scores.items():
        sc = float(np.mean(vals))
        if sc > best_score:
            best_score = sc
            best_cfg = {"hidden_size": key[0], "activation": key[1], "reg": key[2]}
    with open(os.path.join(agg_dir, "best_global_config_interv.json"), "w", encoding="utf-8") as f:
        json.dump({"best_global_config": best_cfg, "mean_val_bacc": best_score}, f, ensure_ascii=False, indent=2)

    # ===== Matriz predita (estrutura idêntica à de entrada) — AGREGADO =====
    # Cria uma planilha igual à de entrada, porém com PlanoIntervencao predito
    # e uma coluna adicional PlanoIntervencao_real ao lado.
    try:
        mpred_dir = os.path.join(agg_dir, "matriz_predita")
        os.makedirs(mpred_dir, exist_ok=True)

        # --- Split interno para calibração (no conjunto de treino) ---
        test_size = min(max(0.1, float(args.calib_holdout)), 0.3)
        gss_full = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE_BASE + 999)
        # usamos TODA a base e separamos um holdout para calibração
        calib_idx = next(gss_full.split(df_all, y_int_all, groups_all))[1]
        mask_tr = np.ones(len(df_all), dtype=bool);
        mask_tr[calib_idx] = False
        tr_idx = np.where(mask_tr)[0]

        df_tr_best = df_all.iloc[tr_idx].copy()
        texts_tr_best = texts_all[tr_idx]
        groups_tr_best = groups_all[tr_idx]

        # --- *_freq: mapeia frequências do TREINO para TREINO+TODOS ---
        def _freq_from_train_full(col_tr, col_full):
            vc = col_tr.astype(str).value_counts(dropna=False)
            freq = (vc / vc.sum()).to_dict()
            f_tr = col_tr.astype(str).map(freq).astype("float32").values.reshape(-1, 1)
            f_full = col_full.astype(str).map(freq).fillna(0.0).astype("float32").values.reshape(-1, 1)
            return f_tr, f_full

        freq_tr_list, freq_all_list = [], []
        for c in CATEG_COLS:
            if c in df_tr_best.columns:
                ftr, ffull = _freq_from_train_full(df_tr_best[c], df_all[c])
                freq_tr_list.append(ftr);
                freq_all_list.append(ffull)

        X_tr_freq = np.hstack(freq_tr_list).astype("float32") if freq_tr_list else np.empty((len(df_tr_best), 0),
                                                                                            dtype="float32")
        X_all_freq = np.hstack(freq_all_list).astype("float32") if freq_all_list else np.empty((len(df_all), 0),
                                                                                               dtype="float32")

        # --- SBERT embeddings (últimas colunas em X_all_raw) ---
        n_emb = len(emb_cols)
        X_tr_emb = X_all_raw[tr_idx][:, -n_emb:].astype("float32")
        X_all_emb = X_all_raw[:, -n_emb:].astype("float32")

        # --- Concatena blocos ---
        X_tr_all = np.hstack([X_tr_freq, X_tr_emb]).astype("float32")
        X_all_all = np.hstack([X_all_freq, X_all_emb]).astype("float32")
        d_freq = X_tr_freq.shape[1]

        # --- TF-IDF + SVD (opcional) ---
        if args.tfidf_dim and int(args.tfidf_dim) > 0:
            Xtr_tfidf, Xall_tfidf = fit_tfidf_svd(
                texts_tr_best, texts_all,
                dim=int(args.tfidf_dim),
                min_df=max(1, int(args.tfidf_min_df)),
                max_df=float(args.tfidf_max_df),
                ngram_max=int(args.tfidf_ngram_max),
                random_state=RANDOM_STATE_BASE + 999
            )
            X_tr_all = np.hstack([X_tr_all, Xtr_tfidf]).astype("float32")
            X_all_all = np.hstack([X_all_all, Xall_tfidf]).astype("float32")

        # --- Escalonamento e peso do bloco *_freq ---
        if d_freq > 0:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr_all[:, :d_freq])
            X_tr_all[:, :d_freq] = scaler.transform(X_tr_all[:, :d_freq]) * float(args.freq_weight)
            X_all_all[:, :d_freq] = scaler.transform(X_all_all[:, :d_freq]) * float(args.freq_weight)

        # --- PCA (opcional) ---
        if args.pca_dim and int(args.pca_dim) > 0:
            X_tr_full, X_all_full = fit_transform_pca(X_tr_all, X_all_all, int(args.pca_dim),
                                                      random_state=RANDOM_STATE_BASE)
        else:
            X_tr_full, X_all_full = X_tr_all, X_all_all

        # --- Targets no treino ---
        y_tr_best = le_y.transform(df_tr_best[ALVO].astype(str).values)
        ypg_tr_best = le_pg.transform(df_tr_best[ALVO_GUIA].astype(str).values)

        # --- Pesos de classe (balanceamento) ---
        uniq, cnt = np.unique(y_tr_best, return_counts=True)
        class_weights = {c: (len(y_tr_best) / (len(uniq) * cnt_i)) for c, cnt_i in dict(zip(uniq, cnt)).items()}
        uniq_pg, cnt_pg = np.unique(ypg_tr_best, return_counts=True)
        class_weights_pg = {c: (len(ypg_tr_best) / (len(uniq_pg) * cnt_i)) for c, cnt_i in
                            dict(zip(uniq_pg, cnt_pg)).items()}

        # --- Configurações finais a partir do best_cfg global ---
        cfg_int = {"hidden_size": int(best_cfg["hidden_size"]), "activation": str(best_cfg["activation"]),
                   "reg": float(best_cfg["reg"])}
        H_PG = max(min(int(best_cfg["hidden_size"]), 800), 200)
        cfg_pg = {"hidden_size": int(H_PG), "activation": "gelu", "reg": float(min(0.01, float(best_cfg["reg"])))}

        # --- Treino dos ensembles finais ---
        modelos_int = treinar_ensemble(
            X_tr_full, y_tr_best, make_soft_targets(y_tr_best, len(classes_interv), LABEL_SMOOTH_EPS),
            len(classes_interv), cfg_int, class_weights,
            n_modelos=int(args.ensemble_n), seed_base=4100, n_jobs=args.n_jobs
        )
        modelos_pg = treinar_ensemble(
            X_tr_full, ypg_tr_best, make_soft_targets(ypg_tr_best, len(classes_pg), LABEL_SMOOTH_EPS),
            len(classes_pg), cfg_pg, class_weights_pg,
            n_modelos=max(10, int(args.ensemble_n) // 2), seed_base=4200, n_jobs=args.n_jobs
        )

        # --- Sub-holdout para calibração (a partir do treino) ---
        gss_tr = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE_BASE + 1000)
        calib_idx_tr = next(gss_tr.split(X_tr_full, y_tr_best, groups_tr_best))[1]
        y_calib = y_tr_best[calib_idx_tr]

        # --- Predições para calibração (modos prob/logit/mix) ---
        P_tr_int_prob = ensemble_predict(modelos_int, X_tr_full, mode="prob")
        P_tr_int_logit = ensemble_predict(modelos_int, X_tr_full, mode="logit")
        P_tr_modes = {"prob": P_tr_int_prob, "logit": P_tr_int_logit}
        for eta in [float(x) for x in args.mix_etas]:
            P_tr_modes[f"mix@{eta:.2f}"] = eta * P_tr_int_prob + (1.0 - eta) * P_tr_int_logit

        P_tr_pg = ensemble_predict(modelos_pg, X_tr_full, mode="prob")

        # --- Regras-Combo (fold-safe em todo o treino) + matriz M (PG->Intervenção) ---
        combo_probs_best, guia_probs_best, _priors_vec = build_combo_tables(df_tr_best, alvo=ALVO, cat_cols=CATEG_COLS,
                                                                            le_y=le_y)
        M = np.zeros((len(classes_pg), len(classes_interv)), dtype=float)
        for i, pg in enumerate(classes_pg):
            M[i, :] = guia_probs_best.get(pg, np.full(len(classes_interv), 1.0 / len(classes_interv), dtype=float))

        mapa_interv_to_pg = build_map_interv_to_pg(df_tr_best, alvo=ALVO, guia=ALVO_GUIA)
        priors = np.bincount(y_tr_best, minlength=len(classes_interv)).astype(np.float64)
        priors = priors / np.clip(priors.sum(), 1.0, None)

        # --- Preparar calibração ---
        P_calib_pg = P_tr_pg[calib_idx_tr]
        P_calib_modes = {m: P_tr_modes[m][calib_idx_tr] for m in P_tr_modes}

        # --- Busca grossa de fusão e calibração ---
        grid_alpha = [0.75, 1.0, 1.5, 2.0]
        grid_lambda = [0.20, 0.40, 0.60, 0.80, 1.0]
        grid_tau = [0.0, 0.25, 0.5]
        grid_T = [0.80, 1.0, 1.20]
        grid_gamma = [0.0, 0.05, 0.10, 0.15]
        grid_epsu = [0.0, 0.02, 0.05]

        best_cal = {"bacc": -np.inf}
        for m in list(P_calib_modes.keys()):
            for a in grid_alpha:
                for l in grid_lambda:
                    for t in grid_tau:
                        for T_ in grid_T:
                            for g in grid_gamma:
                                for eU in grid_epsu:
                                    # HARD
                                    P_ch = fuse_with_pg(P_calib_modes[m], classes_interv, P_calib_pg, classes_pg,
                                                        mapa_interv_to_pg,
                                                        alpha=a, lam=l, tau=t, class_priors=priors, T_int=T_, gamma=g,
                                                        eps_u=eU)
                                    bacc_h = balanced_accuracy_score(y_calib, np.argmax(P_ch, axis=1))
                                    if bacc_h > best_cal["bacc"]:
                                        best_cal = {"bacc": bacc_h, "alpha": a, "lambda": l, "tau": t, "T": T_,
                                                    "gamma": g, "eps_u": eU, "mode": m, "pg_mode": "hard"}
                                    # SOFT
                                    P_cs = fuse_with_pg_soft(P_calib_modes[m], classes_interv, P_calib_pg, classes_pg,
                                                             M,
                                                             alpha=a, lam=l, tau=t, class_priors=priors, T_int=T_,
                                                             gamma=g, eps_u=eU)
                                    bacc_s = balanced_accuracy_score(y_calib, np.argmax(P_cs, axis=1))
                                    if bacc_s > best_cal["bacc"]:
                                        best_cal = {"bacc": bacc_s, "alpha": a, "lambda": l, "tau": t, "T": T_,
                                                    "gamma": g, "eps_u": eU, "mode": m, "pg_mode": "soft"}

        # --- Probabilidades no treino, já com a melhor fusão ---
        P_tr_int_best = P_tr_modes[best_cal["mode"]]
        if best_cal.get("pg_mode", "hard") == "soft":
            P_tr_fused = fuse_with_pg_soft(P_tr_int_best, classes_interv, P_tr_pg, classes_pg, M,
                                           alpha=best_cal["alpha"], lam=best_cal["lambda"], tau=best_cal["tau"],
                                           class_priors=priors, T_int=best_cal["T"], gamma=best_cal["gamma"],
                                           eps_u=best_cal["eps_u"])
        else:
            P_tr_fused = fuse_with_pg(P_tr_int_best, classes_interv, P_tr_pg, classes_pg, mapa_interv_to_pg,
                                      alpha=best_cal["alpha"], lam=best_cal["lambda"], tau=best_cal["tau"],
                                      class_priors=priors, T_int=best_cal["T"], gamma=best_cal["gamma"],
                                      eps_u=best_cal["eps_u"])

        # --- Probabilidades de REGRAS (Intervenção) para treino e TODOS ---
        P_rule_tr, _ = predict_combo(df_tr_best, combo_probs_best, guia_probs_best, _priors_vec, le_y, CATEG_COLS)
        P_rule_all, _ = predict_combo(df_all, combo_probs_best, guia_probs_best, _priors_vec, le_y, CATEG_COLS)

        # --- Tuning de viés por classe (holdout) ---
        bias_grid = [float(x) for x in args.class_bias_search]
        scales_elm, _ = tune_class_bias(P_tr_fused[calib_idx_tr], y_calib, len(classes_interv),
                                        grid_scales=bias_grid, n_rounds=int(args.class_bias_rounds), metric="bacc")
        scales_rule, _ = tune_class_bias(P_rule_tr[calib_idx_tr], y_calib, len(classes_interv),
                                         grid_scales=bias_grid, n_rounds=1, metric="bacc")

        P_tr_elm_bias = apply_class_bias(P_tr_fused, scales_elm)
        P_rule_tr_bias = apply_class_bias(P_rule_tr, scales_rule)

        # --- BLEND calibrado no holdout ---
        blend_w, _ = blend_calibrate([P_rule_tr_bias[calib_idx_tr], P_tr_elm_bias[calib_idx_tr]],
                                     y_calib, metric="bacc", step=float(args.blend_grid_step))

        # --- Temperature Scaling (fit em holdout) ---
        P_tr_final_raw = blend_w[0] * P_rule_tr_bias + blend_w[1] * P_tr_elm_bias
        T_final = fit_temperature_grid(P_tr_final_raw[calib_idx_tr], y_calib)

        # --- Predições para TODOS ---
        P_all_pg = ensemble_predict(modelos_pg, X_all_full, mode="prob")
        P_all_int_prob = ensemble_predict(modelos_int, X_all_full, mode="prob")
        P_all_int_logit = ensemble_predict(modelos_int, X_all_full, mode="logit")
        if best_cal["mode"] == "prob":
            P_all_int = P_all_int_prob
        elif best_cal["mode"] == "logit":
            P_all_int = P_all_int_logit
        elif best_cal["mode"].startswith("mix@"):
            try:
                eta = float(best_cal["mode"].split("@")[1])
            except Exception:
                eta = 0.5
            P_all_int = eta * P_all_int_prob + (1.0 - eta) * P_all_int_logit
        else:
            P_all_int = P_all_int_prob

        # fusão ELM + PG (para TODOS)
        if best_cal.get("pg_mode", "hard") == "soft":
            P_all_fused = fuse_with_pg_soft(P_all_int, classes_interv, P_all_pg, classes_pg, M,
                                            alpha=best_cal["alpha"], lam=best_cal["lambda"], tau=best_cal["tau"],
                                            class_priors=priors, T_int=best_cal["T"], gamma=best_cal["gamma"],
                                            eps_u=best_cal["eps_u"])
        else:
            P_all_fused = fuse_with_pg(P_all_int, classes_interv, P_all_pg, classes_pg, mapa_interv_to_pg,
                                       alpha=best_cal["alpha"], lam=best_cal["lambda"], tau=best_cal["tau"],
                                       class_priors=priors, T_int=best_cal["T"], gamma=best_cal["gamma"],
                                       eps_u=best_cal["eps_u"])

        # aplica viés e blend
        P_all_elm_bias = apply_class_bias(P_all_fused, scales_elm)
        P_rule_all_bias = apply_class_bias(P_rule_all, scales_rule)
        P_all_final_raw = blend_w[0] * P_rule_all_bias + blend_w[1] * P_all_elm_bias
        P_all_final = temperature_scale_probs(P_all_final_raw, T_final)

        y_pred_all = np.argmax(P_all_final, axis=1)
        pred_labels = classes_interv[y_pred_all]

        # --- Constrói a matriz predita mantendo colunas e nomes originais ---
        df_pred_matrix = read_csv_auto(csv_original)  # usa a matriz original (texto claro, sem SBERT)
        pos_alvo = df_pred_matrix.columns.get_loc(ALVO)
        df_pred_matrix.insert(pos_alvo + 1, f"{ALVO}_real", df_pred_matrix[ALVO].astype(str).values)
        df_pred_matrix[ALVO] = pd.Series(pred_labels, index=df_pred_matrix.index).astype(str)

        input_name = os.path.basename(csv_original)
        out_pred_csv = os.path.join(mpred_dir, input_name)
        df_pred_matrix.to_csv(out_pred_csv, sep=";", index=False, encoding="utf-8-sig")
        print(f"[AGG] Matriz predita salva em: {out_pred_csv}")
        # === Cópia da matriz_predita para diretório configurável ===
        try:
            import shutil
            os.makedirs(args.copy_dir, exist_ok=True)
            dest_path = os.path.join(args.copy_dir, os.path.basename(out_pred_csv))
            shutil.copy2(out_pred_csv, dest_path)
            print(f"[AGG] Cópia concluída: {out_pred_csv} -> {dest_path}")
        except Exception as e:
            print(f"[AGG][WARN] Falha ao copiar matriz_predita para '{args.copy_dir}': {e}")

    except Exception as e:
        import traceback
        print(f"[AGG][ERRO] Falha ao gerar a matriz predita: {e}")
        traceback.print_exc()

    # testes estatísticos (ACC final vs ACC ELM plain, pareados por fold)
    def _wilcox(a, b):
        a = np.asarray(a);
        b = np.asarray(b)
        dif = a - b;
        dif_nz = dif[dif != 0]
        if len(dif_nz) < 2:
            return np.nan, np.nan, "pares_nao_zero_insuficientes" if len(dif_nz) > 0 else "todas_diferencas_zero"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = wilcoxon(a, b, zero_method="wilcox", correction=False, mode="auto")
        return float(r.statistic), float(r.pvalue), None

    deltas = np.array(outer_test_accs_final, dtype=float) - np.array(outer_test_accs_plain, dtype=float)

    # t-test pareado (robusto) + decisão por alfa
    t_res = paired_ttest_safe(np.array(outer_test_accs_final, dtype=float),
                              np.array(outer_test_accs_plain, dtype=float))
    reject_t = (t_res["p"] <= float(args.alpha)) if not np.isnan(t_res["p"]) else False

    # Wilcoxon
    w_bp, p_w_bp, nota_bp = _wilcox(outer_test_accs_final, outer_test_accs_plain)
    reject_w = (p_w_bp <= float(args.alpha)) if not np.isnan(p_w_bp) else False

    # Shapiro nos deltas
    sh_ok = False
    if len(deltas) >= 3:
        sh = shapiro(deltas)
        sh_ok = sh.pvalue > float(args.alpha)
    else:
        sh = None

    # Permutação (sign-flip) da média dos deltas
    perm = paired_permutation_test(deltas, reps_max=100000, seed=RANDOM_STATE_BASE)
    reject_perm = (perm["p"] <= float(args.alpha)) if not np.isnan(perm["p"]) else False

    # Efeitos e IC
    eff = effect_sizes_and_ci(deltas, alpha=float(args.alpha))

    # ===== Testes adicionais: FINAL vs {SVM, KNN, DT} (ACC por fold) =====
    comparators = {
        "SVM": np.array(outer_test_accs_svm, dtype=float),
        "KNN": np.array(outer_test_accs_knn, dtype=float),
        "DT": np.array(outer_test_accs_dt, dtype=float),
    }
    cmp_tests_summary = {}  # <--- NOVO: para reaproveitar no resumo agregado

    for name, arr in comparators.items():
        deltas_cmp = np.array(outer_test_accs_final, dtype=float) - arr
        t_res_cmp = paired_ttest_safe(np.array(outer_test_accs_final, dtype=float), arr)
        w_stat, p_w, _ = _wilcox(outer_test_accs_final, arr)
        perm_cmp = paired_permutation_test(deltas_cmp, reps_max=100000, seed=RANDOM_STATE_BASE + 1)
        eff_cmp = effect_sizes_and_ci(deltas_cmp, alpha=float(args.alpha))

        # NOVO: guardar tudo para escrever no resumo_agregado.txt
        cmp_tests_summary[name] = {
            "t": t_res_cmp["t"], "p_t": t_res_cmp["p"], "p_t_fmt": t_res_cmp["p_fmt"], "n": t_res_cmp["n"],
            "W": w_stat, "p_w": p_w,
            "perm_method": perm_cmp["method"], "perm_p": perm_cmp["p"], "perm_p_fmt": perm_cmp["p_fmt"],
            "mean_delta": eff_cmp["mean_delta"], "sd_delta": eff_cmp["sd_delta"],
            "ci_lo": eff_cmp["ci_mean_lo"], "ci_hi": eff_cmp["ci_mean_hi"],
            "cohen_dz": eff_cmp["cohen_dz"]
        }

        with open(os.path.join(agg_dir, f"tests_FINAL_vs_{name}.txt"), "w", encoding="utf-8") as fcmp:
            fcmp.write(f"FINAL vs {name} — ACC por fold\n")
            fcmp.write(f"t-test pareado: t={t_res_cmp['t']:.6f}, p={t_res_cmp['p_fmt']}, n={t_res_cmp['n']}\n")
            fcmp.write(f"Wilcoxon: W={w_stat:.6f}, p={fmt_p(p_w)}\n")
            fcmp.write(f"Permutação (sign-flip): {perm_cmp['method']}, p={perm_cmp['p_fmt']}\n")
            fcmp.write(f"Média delta = {eff_cmp['mean_delta']:.6f}, DP={eff_cmp['sd_delta']:.6f}, n={eff_cmp['n']}\n")
            fcmp.write(f"IC95% da média = [{eff_cmp['ci_mean_lo']:.6f}, {eff_cmp['ci_mean_hi']:.6f}]\n")
            fcmp.write(f"Cohen d_z = {eff_cmp['cohen_dz']:.6f}\n")

    # tabela por fold
    df_folds = pd.DataFrame({
        "fold": list(range(1, len(outer_test_accs_final) + 1)),
        "outer_train_acc_final": outer_train_accs[:len(outer_test_accs_final)],
        "outer_test_acc_elm_plain": outer_test_accs_plain[:len(outer_test_accs_final)],
        "outer_test_acc_final": outer_test_accs_final[:len(outer_test_accs_final)],
        "delta_final_minus_plain_acc": deltas[:len(outer_test_accs_final)],
        "rule_combo_seen_rate": combo_seen_rates[:len(outer_test_accs_final)],
        "blend_weights_rule_elm": [json.dumps(w) for w in blend_weights_list],
        "fusion_params_elm_pg": [json.dumps(p) for p in chosen_fusion_params],
        "ensemble_mode_elm": chosen_ens_mode,
        "stack_weights_elm": [json.dumps(w) for w in chosen_stack_weights],
        "train_time_s": train_times[:len(outer_test_accs_final)],
        "test_time_s": test_times[:len(outer_test_accs_final)],
        "samples_per_sec": infer_samples_per_sec[:len(outer_test_accs_final)],
        "brier_train": fold_brier_train[:len(outer_test_accs_final)],
        "brier_test": fold_brier_test[:len(outer_test_accs_final)],
        "ece_train": fold_ece_train[:len(outer_test_accs_final)],
        "ece_test": fold_ece_test[:len(outer_test_accs_final)],
    })
    df_fold_extra_train = pd.DataFrame(fold_metrics_train)
    df_fold_extra_test = pd.DataFrame(fold_metrics_test)
    df_fold_extra_test.to_csv(os.path.join(agg_dir, "fold_metrics_test_aggregate.csv"), sep=";", index=False,
                              encoding="utf-8-sig")
    df_fold_extra_train.to_csv(os.path.join(agg_dir, "fold_metrics_train_aggregate.csv"), sep=";", index=False,
                               encoding="utf-8-sig")
    df_folds.to_csv(os.path.join(agg_dir, "outer_folds_scores_final_vs_plain.csv"), sep=";", index=False,
                    encoding="utf-8-sig")

    # === CSV com todos os métodos (Test) ===
    df_methods = pd.DataFrame({
        "fold": list(range(1, len(outer_test_accs_final) + 1)),
        "acc_test_FINAL": outer_test_accs_final[:len(outer_test_accs_final)],
        "acc_test_PLAIN": outer_test_accs_plain[:len(outer_test_accs_final)],
        "acc_test_SVM": outer_test_accs_svm[:len(outer_test_accs_final)],
        "acc_test_KNN": outer_test_accs_knn[:len(outer_test_accs_final)],
        "acc_test_DT": outer_test_accs_dt[:len(outer_test_accs_final)],
    })
    df_methods.to_csv(os.path.join(methods_dir, "outer_folds_scores_methods.csv"),
                      sep=";", index=False, encoding="utf-8-sig")

    # === Tabela completa por método e fold (inclui DT) ===
    records = []
    for meth in methods_all:  # ["FINAL","PLAIN","SVM","KNN","DT"]
        Ps = all_test_probs_by_method[meth]
        Ys = y_test_true_by_method[meth]
        for i, (P, y) in enumerate(zip(Ps, Ys), start=1):
            m = fold_metrics_test_by_method[meth][i - 1]  # contém acc, bacc, kappa, P/R/F1 (macro/micro/weighted)
            rec = {"fold": i, "method": meth, **m}
            # Top-k
            rec["top1"] = topk_accuracy_score(y, P, k=1)
            rec["top3"] = topk_accuracy_score(y, P, k=min(3, P.shape[1]))
            # Calibração
            rec["brier"] = brier_score_multiclass(y, P, n_classes)
            rec["ece"] = expected_calibration_error(y, P, n_bins=15)
            # AUC/AP micro/macro
            yb = label_binarize(y, classes=np.arange(n_classes))
            try:
                rec["roc_auc_macro"] = roc_auc_score(yb, P, average="macro", multi_class="ovr")
            except Exception:
                rec["roc_auc_macro"] = np.nan
            try:
                rec["roc_auc_micro"] = roc_auc_score(yb, P, average="micro", multi_class="ovr")
            except Exception:
                rec["roc_auc_micro"] = np.nan
            try:
                rec["ap_macro"] = average_precision_score(yb, P, average="macro")
                rec["ap_micro"] = average_precision_score(yb, P, average="micro")
            except Exception:
                rec["ap_macro"] = np.nan;
                rec["ap_micro"] = np.nan
            records.append(rec)

    df_methods_full = pd.DataFrame(records)
    df_methods_full.to_csv(os.path.join(methods_dir, "outer_folds_metrics_methods_full.csv"),
                           sep=";", index=False, encoding="utf-8-sig")

    # === Resumo por método (médias ± dp) para leitura rápida (inclui DT) ===
    agg_rows = []
    cols_to_summarize = [
        "accuracy", "balanced_accuracy", "kappa",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_micro", "recall_micro", "f1_micro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "top1", "top3", "roc_auc_macro", "roc_auc_micro", "ap_macro", "ap_micro", "brier", "ece"
    ]
    for meth in methods_all:
        sub = df_methods_full[df_methods_full["method"] == meth]
        for col in cols_to_summarize:
            mu = float(np.mean(sub[col])) if not sub.empty else np.nan
            sd = float(np.std(sub[col], ddof=1)) if len(sub) > 1 else np.nan
            agg_rows.append({"method": meth, "metric": col, "mean": mu, "std": sd})
    pd.DataFrame(agg_rows).to_csv(os.path.join(methods_dir, "methods_metric_summary.csv"),
                                  sep=";", index=False, encoding="utf-8-sig")

    # === Matrizes de confusão agregadas por método (inclui DT) ===
    for meth in methods_all:
        if len(all_test_probs_by_method[meth]) == 0: continue
        Pm = np.vstack(all_test_probs_by_method[meth])
        ym = np.concatenate(y_test_true_by_method[meth])
        y_pred_m = np.argmax(Pm, axis=1)
        _save_confusion(ym, y_pred_m, classes_interv, methods_dir, f"{meth.lower()}_test_aggregate")

    # Barras (média ± EP) de acurácia por método (Test)
    methods_keys = ["acc_test_FINAL", "acc_test_PLAIN", "acc_test_SVM", "acc_test_KNN", "acc_test_DT"]
    labels_methods = ["FINAL", "PLAIN", "SVM", "KNN", "DT"]
    means = [float(np.mean(df_methods[k])) for k in methods_keys]
    ses = [float(np.std(df_methods[k], ddof=1) / math.sqrt(len(df_methods))) for k in methods_keys]
    xs = np.arange(len(methods_keys))
    plt.figure();
    plt.bar(xs, means, yerr=ses, capsize=5);
    plt.xticks(xs, labels_methods)
    plt.title("Acurácia média por método (Test) ± EP");
    plt.tight_layout()
    plt.savefig(os.path.join(methods_dir, "bars_methods_means_se.png"), bbox_inches="tight");
    plt.close()

    # Boxplot de acurácia por método (Test)
    plt.figure()

    data = [df_methods[k].astype(float).values for k in methods_keys]
    means = [float(np.mean(d)) for d in data]

    bp = plt.boxplot(
        data,
        labels=labels_methods,  # se seu matplotlib for antigo, troque por plt.xticks depois
        showfliers=True
    )

    # Troca a linha central (mediana) pela média
    for i, med_line in enumerate(bp["medians"]):
        m = means[i]
        med_line.set_ydata([m, m])  # desenha a "mediana" na altura da média

    plt.title("Boxplot — Acurácia por método (Test) — Linha central Média")
    plt.tight_layout()
    plt.savefig(os.path.join(methods_dir, "boxplot_acc_by_method.png"), bbox_inches="tight")
    plt.close()

    # -------- NOVO: boxplots de TEMPOS por método --------
    def _box_by_method_dict(dct, labels, title, filename):
        data = [list(map(float, dct[m])) for m in labels]
        plt.figure()
        try:
            plt.boxplot(data, tick_labels=labels)
        except TypeError:
            plt.boxplot(data);
            plt.xticks(np.arange(1, len(labels) + 1), labels)
        plt.title(title);
        plt.tight_layout()
        plt.savefig(os.path.join(methods_dir, filename), bbox_inches="tight");
        plt.close()

    _box_by_method_dict(train_times_by_method, labels_methods,
                        "Boxplot — Tempo de Treino por método (s)",
                        "boxplot_train_time_by_method.png")
    _box_by_method_dict(test_times_by_method, labels_methods,
                        "Boxplot — Tempo de Teste por método (s)",
                        "boxplot_test_time_by_method.png")

    # -------- NOVO: boxplots de MÉTRICAS (Train/Test) por método --------
    def _box_metric_by_method(metric_key, split, filename_prefix):
        src = fold_metrics_train_by_method if split == "train" else fold_metrics_test_by_method
        data = [[m[metric_key] for m in src[meth]] for meth in labels_methods]
        plt.figure()
        try:
            plt.boxplot(data, tick_labels=labels_methods)
        except TypeError:
            plt.boxplot(data);
            plt.xticks(np.arange(1, len(labels_methods) + 1), labels_methods)
        plt.title(f"Boxplot — {metric_key} ({split}) por método")
        plt.tight_layout()
        plt.savefig(os.path.join(methods_dir, f"{filename_prefix}_{split}_by_method.png"), bbox_inches="tight");
        plt.close()

    # todas as métricas disponíveis no avaliar_conjunto
    _metrics = ["accuracy", "balanced_accuracy", "kappa",
                "precision_macro", "recall_macro", "f1_macro",
                "precision_micro", "recall_micro", "f1_micro",
                "precision_weighted", "recall_weighted", "f1_weighted"]
    for mk in _metrics:
        _box_metric_by_method(mk, "train", f"boxplot_{mk}")
        _box_metric_by_method(mk, "test", f"boxplot_{mk}")

    # -------- NOVO: ROC/PR agregados comparando métodos --------
    try:
        methods_plot = labels_methods  # ["FINAL","PLAIN","SVM","KNN","DT"]
        # ROC micro por método (uma curva por método)
        plt.figure()
        for meth in methods_plot:
            if len(all_test_probs_by_method[meth]) == 0: continue
            Pm = np.vstack(all_test_probs_by_method[meth])
            ym = np.concatenate(y_test_true_by_method[meth])
            yb = label_binarize(ym, classes=np.arange(n_classes)).ravel()
            fpr, tpr, _ = roc_curve(yb, Pm.ravel())
            auc_m = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{meth} (AUC={auc_m:.3f})")
        plt.plot([0, 1], [0, 1], "k--");
        plt.xlabel("FPR");
        plt.ylabel("TPR")
        plt.title("ROC micro por método (agregado Test)")
        plt.legend(loc="lower right");
        plt.tight_layout()
        plt.savefig(os.path.join(methods_dir, "roc_micro_by_method.png"), bbox_inches="tight");
        plt.close()
    except Exception:
        pass

    try:
        # PR micro por método (uma curva por método)
        plt.figure()
        for meth in methods_plot:
            if len(all_test_probs_by_method[meth]) == 0: continue
            Pm = np.vstack(all_test_probs_by_method[meth])
            ym = np.concatenate(y_test_true_by_method[meth])
            yb = label_binarize(ym, classes=np.arange(n_classes)).ravel()
            prec, rec, _ = precision_recall_curve(yb, Pm.ravel())
            ap_m = average_precision_score(label_binarize(ym, classes=np.arange(n_classes)), Pm, average="micro")
            plt.plot(rec, prec, label=f"{meth} (AP={ap_m:.3f})")
        plt.xlabel("Recall");
        plt.ylabel("Precision")
        plt.title("Precision-Recall micro por método (agregado Test)")
        plt.legend(loc="lower left");
        plt.tight_layout()
        plt.savefig(os.path.join(methods_dir, "pr_micro_by_method.png"), bbox_inches="tight");
        plt.close()
    except Exception:
        pass

    # Concat para reliability/ROC/PR agregados
    if len(all_test_probs) > 0:
        P_test_all = np.vstack(all_test_probs)
        y_test_all = np.concatenate(all_test_true)
        y_test_bin = label_binarize(y_test_all, classes=np.arange(n_classes))

        # ===== Matriz de Confusão — AGREGADO (TEST) =====
        try:
            y_pred_test_all = np.argmax(P_test_all, axis=1)
            _save_confusion(y_test_all, y_pred_test_all, classes_interv, agg_dir, "test_aggregate")
            cm_test = confusion_matrix(y_test_all, y_pred_test_all, labels=np.arange(len(classes_interv)))

            out_min_png = os.path.join(agg_dir, "test_aggregate_confusion_matrix_minimal.png")
            print(f"[AGG] Salvando matriz de confusão minimalista em: {out_min_png}")
            _save_confusion_minimal(cm_test, out_min_png, annotate=True, fmt="d", fontsize=18, figsize=(12, 12))
            print("[AGG] Matriz minimalista salva com sucesso.")
        except Exception as e:
            print(f"[AGG][ERRO] Falhou ao salvar matriz minimalista: {e}")
            raise

        # Reliability
        plot_reliability_diagram(y_test_all, P_test_all,
                                 os.path.join(agg_dir, "reliability_diagram_test.png"),
                                 n_bins=15, title="Reliability Diagram — Test (Top-1)")

        # ROC micro/macro
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), P_test_all.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            auc_macro = roc_auc_score(y_test_bin, P_test_all, average="macro", multi_class="ovr")
            plt.figure()
            plt.plot(fpr_micro, tpr_micro, label=f"micro AUC = {auc_micro:.3f}")
            plt.plot([0, 1], [0, 1], "k--");
            plt.xlabel("FPR");
            plt.ylabel("TPR");
            plt.title(f"ROC — Micro & Macro (Test; macro AUC={auc_macro:.3f})")
            plt.legend(loc="lower right");
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "roc_micro_macro.png"), bbox_inches="tight");
            plt.close()
        except Exception:
            pass

        # PR micro/macro
        try:
            precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), P_test_all.ravel())
            ap_micro = average_precision_score(y_test_bin, P_test_all, average="micro")
            ap_macro = average_precision_score(y_test_bin, P_test_all, average="macro")
            plt.figure()
            plt.plot(recall_micro, precision_micro, label=f"micro AP = {ap_micro:.3f}")
            plt.xlabel("Recall");
            plt.ylabel("Precision");
            plt.title(f"Precision-Recall — Micro & Macro (Test; macro AP={ap_macro:.3f})")
            plt.legend(loc="lower left");
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "pr_micro_macro.png"), bbox_inches="tight");
            plt.close()
        except Exception:
            pass

    if len(all_train_probs) > 0:
        P_train_all = np.vstack(all_train_probs)
        y_train_all = np.concatenate(all_train_true)
        plot_reliability_diagram(y_train_all, P_train_all,
                                 os.path.join(agg_dir, "reliability_diagram_train.png"),
                                 n_bins=15, title="Reliability Diagram — Train (Top-1)")

    # Boxplots adicionais de métricas (Test por fold)
    def _box_triple(keys, title, out_png):
        data = [[m[k] for m in fold_metrics_test] for k in keys]
        try:
            plt.figure();
            plt.boxplot(data, tick_labels=keys)
        except TypeError:
            plt.figure();
            plt.boxplot(data);
            plt.xticks(np.arange(1, len(keys) + 1), keys)
        plt.title(title + " (Test)");
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, out_png), bbox_inches="tight");
        plt.close()

    _box_triple(["precision_macro", "precision_micro", "precision_weighted"],
                "Boxplot — Precision", "boxplot_precision_macro_micro_weighted.png")
    _box_triple(["recall_macro", "recall_micro", "recall_weighted"],
                "Boxplot — Recall", "boxplot_recall_macro_micro_weighted.png")
    _box_triple(["f1_macro", "f1_micro", "f1_weighted"],
                "Boxplot — F1", "boxplot_f1_macro_micro_weighted.png")

    # Barras com médias + erro-padrão (ACC, BACC, Macro-F1)
    def _bars_means_se(keys, labels, filename):
        means = [float(np.mean([m[k] for m in fold_metrics_test])) for k in keys]
        ses = [float(np.std([m[k] for m in fold_metrics_test], ddof=1) / math.sqrt(len(fold_metrics_test))) for k in
               keys]
        xs = np.arange(len(keys))
        plt.figure()
        plt.bar(xs, means, yerr=ses, capsize=5)
        plt.xticks(xs, labels)
        plt.title("Médias por fold ± EP (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, filename), bbox_inches="tight");
        plt.close()

    _bars_means_se(["accuracy", "balanced_accuracy", "f1_macro"],
                   ["ACC", "BACC", "F1_macro"], "bars_metrics_means_se.png")

    # ===== McNemar (agregado em todos os folds) =====
    b_sum = int(np.sum(mcn_b_list))
    c_sum = int(np.sum(mcn_c_list))
    mcn_agg = mcnemar_test(b_sum, c_sum, exact=True)

    _safe_json_dump(
        {"b_total": b_sum, "c_total": c_sum,
         "method": mcn_agg["method"], "stat": mcn_agg["stat"], "pvalue": fmt_p(mcn_agg["pvalue"])},
        os.path.join(agg_dir, "mcnemar_agregado.json")
    )
    _save_kv_csv(
        {"b_total": b_sum, "c_total": c_sum, "method": mcn_agg["method"],
         "stat": mcn_agg["stat"], "pvalue": fmt_p(mcn_agg["pvalue"])},
        os.path.join(agg_dir, "mcnemar_agregado.csv")
    )

    # ===== McNemar agregado: FINAL vs {SVM, KNN, DT} =====
    mcn_cmp_summary = {}  # <--- NOVO

    for meth, b_list, c_list in [
        ("SVM", mcn_b_svm_list, mcn_c_svm_list),
        ("KNN", mcn_b_knn_list, mcn_c_knn_list),
        ("DT", mcn_b_dt_list, mcn_c_dt_list),
    ]:
        b_tot = int(np.sum(b_list));
        c_tot = int(np.sum(c_list))
        mcn = mcnemar_test(b_tot, c_tot, exact=True)

        # NOVO: guardar para o resumo agregado
        mcn_cmp_summary[meth] = {
            "b_total": b_tot, "c_total": c_tot,
            "method": mcn["method"], "stat": mcn["stat"], "pvalue": mcn["pvalue"]
        }

        _safe_json_dump(
            {"b_total": b_tot, "c_total": c_tot, "method": mcn["method"],
             "stat": mcn["stat"], "pvalue": fmt_p(mcn["pvalue"])},
            os.path.join(agg_dir, f"mcnemar_agregado_final_vs_{meth.lower()}.json")
        )
        _save_kv_csv(
            {"b_total": b_tot, "c_total": c_tot, "method": mcn["method"],
             "stat": mcn["stat"], "pvalue": fmt_p(mcn["pvalue"])},
            os.path.join(agg_dir, f"mcnemar_agregado_final_vs_{meth.lower()}.csv")
        )

    # resumo agregado
    t_total = time.time() - t0_all
    with open(os.path.join(agg_dir, "resumo_agregado.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Resumo Nested CV – SBERT v9.8 (Regra-Combo fold-safe + ELM | blend calibrado + bias tuning | TS pós-hoc)\n\n")
        f.write(f"Tempo total de execução (s): {t_total:.2f}\n")
        f.write(f"Tempo total (min): {t_total / 60.0:.2f}\n\n")
        f.write(f"Folds (outer, inner): {len(outer_test_accs_final)}, {int(args.inner_folds)}\n")
        f.write(f"Ensemble ELM por fold: N={int(N_MODELOS_ENSEMBLE)} (dividido por topK)\n")
        f.write(
            f"Embeddings: {len(emb_cols)} SBERT + *_freq (w={float(args.freq_weight)}) + TFIDF-SVD(dim={int(args.tfidf_dim)})\n")
        f.write(f"PCA final: {'ON, dim=' + str(int(args.pca_dim)) if args.pca_dim and args.pca_dim > 0 else 'OFF'}\n")
        f.write(f"Label smoothing: {LABEL_SMOOTH_EPS}\n")
        f.write(f"n_jobs={args.n_jobs} | blas_threads={args.blas_threads} | device={args.device}\n\n")
        f.write("Contagem exata (Nested CV):\n")
        f.write(f"  Outer (executados): {n_outer}\n")
        f.write(f"  Inner por outer   : {n_inner}\n")
        f.write(
            f"  Grid interno (ELM): {grid_total_cfgs} cfgs = {len(args.hidden_sizes)}×{len(args.activations)}×{len(args.regs)}\n")
        f.write(f"  Treinos do grid   : {total_inner_fits}\n")
        f.write(
            f"  Modelos ELM finais (TOTAL): {total_elm_models_finais}  (Intervenção≈{elm_models_interv}, PG={elm_models_pg})\n")
        f.write(f"  Treinos SVM/KNN/DT: {total_svm_treinos}/{total_knn_treinos}/{total_dt_treinos}\n\n")

        mean_tr = np.mean(outer_train_accs);
        std_tr = np.std(outer_train_accs)
        mean_te = np.mean(outer_test_accs_final);
        std_te = np.std(outer_test_accs_final)

        f.write("ACURÁCIA (médias ± dp):\n")
        f.write(f"  Train (FINAL): {mean_tr:.4f} ± {std_tr:.4f}\n")
        f.write(f"  Test  (FINAL): {mean_te:.4f} ± {std_te:.4f}\n\n")

        f.write("TEMPOS por fold (s):\n")
        f.write(f"  Treino: mean={np.mean(train_times):.4f} ± {np.std(train_times):.4f}\n")
        f.write(f"  Teste : mean={np.mean(test_times):.4f} ± {np.std(test_times):.4f}\n")
        f.write(
            f"  Samples/s (Test): mean={np.mean(infer_samples_per_sec):.2f} ± {np.std(infer_samples_per_sec):.2f}\n\n")

        f.write("CALIBRAÇÃO (Brier / ECE, com Temperature Scaling pós-hoc):\n")
        f.write(
            f"  Train — Brier={np.mean(fold_brier_train):.4f} ± {np.std(fold_brier_train):.4f} | ECE={np.mean(fold_ece_train):.4f} ± {np.std(fold_ece_train):.4f}\n")
        f.write(
            f"  Test  — Brier={np.mean(fold_brier_test):.4f} ± {np.std(fold_brier_test):.4f} | ECE={np.mean(fold_ece_test):.4f} ± {np.std(fold_ece_test):.4f}\n")
        f.write("  Figuras: reliability_diagram_train.png, reliability_diagram_test.png\n")

        f.write("\nDetalhe por fold (ACC treino/teste e tempos):\n")
        for i in range(len(outer_test_accs_final)):
            btr = fold_brier_train[i] if i < len(fold_brier_train) else float('nan')
            etr = fold_ece_train[i] if i < len(fold_ece_train) else float('nan')
            bte = fold_brier_test[i] if i < len(fold_brier_test) else float('nan')
            ete = fold_ece_test[i] if i < len(fold_ece_test) else float('nan')
            sps = infer_samples_per_sec[i] if i < len(infer_samples_per_sec) else float('nan')
            f.write(
                "  Fold {idx}: Train={tr:.4f} | Test(FINAL)={tef:.4f} | Test(PLAIN)={tep:.4f} | "
                "Tempos(s) Treino={ttr:.2f} Teste={tte:.4f} | Samples/s={sps:.2f} | "
                "Brier(train/test)={btr:.4f}/{bte:.4f} | ECE(train/test)={etr:.4f}/{ete:.4f}\n"
                .format(
                    idx=i + 1,
                    tr=outer_train_accs[i],
                    tef=outer_test_accs_final[i],
                    tep=outer_test_accs_plain[i],
                    ttr=train_times[i],
                    tte=test_times[i],
                    sps=sps, btr=btr, bte=bte, etr=etr, ete=ete
                )
            )

        f.write("\nTestes estatísticos — FINAL vs ELM plain (ACC, pareado por fold):\n")
        alfa = float(args.alpha)

        # Shapiro (deltas)
        if sh is not None:
            concl = "compatível com normalidade" if sh.pvalue > alfa else "não compatível com normalidade"
            f.write(f"  Shapiro (deltas): stat={sh.statistic:.6f}, p={sh.pvalue:.6e} → {concl} a α={alfa:.2f}\n")
        else:
            f.write("  Shapiro (deltas): N insuficiente para um teste de normalidade confiável.\n")

        # t-test pareado
        decisao_t = "REJEITA H0 (diferença significativa)" if reject_t else "NÃO REJEITA H0"
        f.write(
            f"  t-test pareado  : t={t_res['t']:.6f}, p={t_res['p_fmt']} (n={t_res['n']}), α={alfa:.2f} → {decisao_t}.\n")
        f.write(
            "    Interpretação: comparam-se as médias dos deltas (FINAL−PLAIN). Se p ≤ α, há evidência de que a média dos deltas difere de 0.\n")

        # Wilcoxon
        decisao_w = "REJEITA H0 (diferença significativa)" if reject_w else "NÃO REJEITA H0"
        nota_str = "" if nota_bp is None else f" | nota: {nota_bp}"
        f.write(
            f"  Wilcoxon        : W={w_bp:.6f}, p={fmt_p(p_w_bp)} (bilateral){nota_str} → {decisao_w} a α={alfa:.2f}.\n")
        f.write(
            "    Interpretação: teste não paramétrico sobre os deltas; com N pequeno o p é discreto (pode ficar pouco abaixo/acima de 0.05).\n")

        # Permutação pareada
        decisao_perm = "REJEITA H0 (diferença significativa)" if reject_perm else "NÃO REJEITA H0"
        f.write(
            f"  Permutação (sign-flip): método={perm['method']}, p={perm['p_fmt']} → {decisao_perm} a α={alfa:.2f}.\n")
        f.write("    Interpretação: embaralha sinais dos deltas para avaliar a média observada sob H0 (média=0).\n")

        # Efeitos e IC
        f.write("\nEfeito e incerteza (deltas de ACC = FINAL−PLAIN):\n")
        f.write(f"  Média dos deltas = {eff['mean_delta']:.6f}  | DP = {eff['sd_delta']:.6f}  | n = {eff['n']}\n")
        f.write(f"  IC95% da média (t) = [{eff['ci_mean_lo']:.6f}, {eff['ci_mean_hi']:.6f}]\n")
        f.write(f"  Cohen’s d_z (pareado) = {eff['cohen_dz']:.6f}\n")

        # McNemar agregado + interpretação
        direc_agg = "FINAL corrige mais erros do que o PLAIN" if c_sum > b_sum else (
            "PLAIN corrige mais erros do que o FINAL" if b_sum > c_sum else "sem assimetria nos discordantes")
        decisao_m = "REJEITA H0 (diferença nas proporções de acertos por amostra)" if (
                    mcn_agg["pvalue"] is not None and mcn_agg["pvalue"] <= alfa) else "NÃO REJEITA H0"
        f.write("\nTeste de McNemar (agregado, FINAL vs PLAIN — pares discordantes somados em todos os folds):\n")
        f.write(
            f"  b(total)={b_sum}, c(total)={c_sum} | método={mcn_agg['method']} | estatística={mcn_agg['stat']} | p-valor={fmt_p(mcn_agg['pvalue'])}\n")
        f.write(f"  Decisão: {decisao_m} a α={alfa:.2f}. Interpretação: {direc_agg} (diferença refletida em c−b).\n")

        # ===== NOVO: Médias por método (ACC no Test) =====
        f.write("\nMédias de acurácia por método (Test) ± DP:\n")
        for label, key in zip(labels_methods, methods_keys):
            f.write(f"  {label}: {np.mean(df_methods[key]):.4f} ± {np.std(df_methods[key], ddof=1):.4f}\n")

        f.write("\nResumos específicos por método:\n")

        # Garante que temos a lista de métodos (já definida antes como labels_methods)
        # labels_methods = ["FINAL","PLAIN","SVM","KNN","DT"]

        for meth in labels_methods:
            f.write(f"\nResumo específico — {meth}:\n")
            sub = df_methods_full[df_methods_full["method"] == meth]
            if sub.empty:
                f.write("  (sem registros)\n")
                continue

            acc = sub["accuracy"].mean()
            bacc = sub["balanced_accuracy"].mean()
            f1_macro = sub["f1_macro"].mean()
            kappa = sub["kappa"].mean()
            top1 = sub["top1"].mean()
            top3 = sub["top3"].mean()
            auc_micro = sub["roc_auc_micro"].mean()
            auc_macro = sub["roc_auc_macro"].mean()
            ap_micro = sub["ap_micro"].mean()
            ap_macro = sub["ap_macro"].mean()
            brier = sub["brier"].mean()
            ece = sub["ece"].mean()

            f.write(f"  ACC={acc:.4f} | BACC={bacc:.4f} | Macro-F1={f1_macro:.4f} | Kappa={kappa:.4f}\n")
            f.write(f"  Top-1={top1:.4f} | Top-3={top3:.4f}\n")
            f.write(
                f"  AUC micro/macro={auc_micro:.4f}/{auc_macro:.4f} | AP micro/macro={ap_micro:.4f}/{ap_macro:.4f}\n")
            f.write(f"  Calibração: Brier={brier:.4f} | ECE={ece:.4f}\n")

            # caminho da matriz de confusão agregada para cada método
            cm_png = f"comparacao_metodos/{meth.lower()}_test_aggregate_confusion_matrix.png"
            f.write(f"  Ver matrizes agregadas: {cm_png} (e versão normalizada).\n")

        # ===== NOVO: Comparações com técnicas de referência (ACC no Test, pareado por fold) =====
        f.write("\nComparações com técnicas de referência (ACC no Test, pareado por fold):\n")
        for name in ["SVM", "KNN", "DT"]:
            if name in cmp_tests_summary:
                ct = cmp_tests_summary[name]
                decisao_t = "REJEITA H0 (diferença significativa)" if (
                            not np.isnan(ct["p_t"]) and ct["p_t"] <= alfa) else "NÃO REJEITA H0"
                decisao_w = "REJEITA H0 (diferença significativa)" if (
                            not np.isnan(ct["p_w"]) and ct["p_w"] <= alfa) else "NÃO REJEITA H0"
                decisao_perm = "REJEITA H0 (diferença significativa)" if (
                            not np.isnan(ct["perm_p"]) and ct["perm_p"] <= alfa) else "NÃO REJEITA H0"
                direc_delta = "FINAL > " + name if ct["mean_delta"] > 0 else (
                    "FINAL < " + name if ct["mean_delta"] < 0 else "sem diferença média")

                f.write(f"\n  FINAL vs {name}:\n")
                f.write(f"    t-test pareado : t={ct['t']:.6f}, p={ct['p_t_fmt']} (n={ct['n']}) → {decisao_t}\n")
                f.write(f"    Wilcoxon       : W={ct['W']:.6f}, p={fmt_p(ct['p_w'])} → {decisao_w}\n")
                f.write(f"    Permutação     : {ct['perm_method']}, p={ct['perm_p_fmt']} → {decisao_perm}\n")
                f.write(
                    f"    ΔACC médio = {ct['mean_delta']:.6f} (DP={ct['sd_delta']:.6f}, n={ct['n']}) | IC95% = [{ct['ci_lo']:.6f}, {ct['ci_hi']:.6f}] | d_z = {ct['cohen_dz']:.6f} → direção: {direc_delta}\n")

                # McNemar agregado por comparador
                if name in mcn_cmp_summary:
                    mc = mcn_cmp_summary[name]
                    decisao_mcn = "REJEITA H0 (diferença nas proporções)" if (
                                mc["pvalue"] is not None and mc["pvalue"] <= alfa) else "NÃO REJEITA H0"
                    direc_mcn = "FINAL corrige mais" if mc["c_total"] > mc["b_total"] else (
                        "Comparador corrige mais" if mc["b_total"] > mc[
                            "c_total"] else "sem assimetria nos discordantes")
                    f.write(
                        f"    McNemar (agregado): b={mc['b_total']}, c={mc['c_total']} | método={mc['method']} | estatística={mc['stat']} | p-valor={fmt_p(mc['pvalue'])} → {decisao_mcn}; direção: {direc_mcn}\n")

        f.write("\nGráficos agregados (Test):\n")
        f.write(
            "  - ROC/PR: roc_micro_macro.png, pr_micro_macro.png, roc_micro_by_method.png, pr_micro_by_method.png\n")
        f.write("  - Boxplots de métricas por método (train/test):\n")
        f.write(
            "      boxplot_accuracy_train_by_method.png, boxplot_accuracy_test_by_method.png (nomes gerados como boxplot_<métrica>_{train|test}_by_method.png)\n")
        f.write(
            "      boxplot_precision_macro_{train|test}_by_method.png, boxplot_recall_macro_{train|test}_by_method.png, boxplot_f1_macro_{train|test}_by_method.png, ...\n")
        f.write("  - Tempos por método: boxplot_train_time_by_method.png, boxplot_test_time_by_method.png\n")
        f.write(
            "  - Resumos: boxplot_acc_by_method.png, bars_methods_means_se.png, boxplot_precision_macro_micro_weighted.png,\n")
        f.write(
            "             boxplot_recall_macro_micro_weighted.png, boxplot_f1_macro_micro_weighted.png, bars_metrics_means_se.png\n\n")

        f.write("\nMelhor configuração GLOBAL (Intervenção; média val_bacc_mean no inner):\n")
        f.write(
            json.dumps({"config": best_cfg, "mean_inner_val_bacc": float(best_score)}, ensure_ascii=False, indent=2))
        f.write("\n")

        # ---------- NOVOS BOXPLOTS: ACC (Train vs Test), Tempos (Treino vs Teste) e Throughput ----------
        def _box_series(series_list, labels, title, filename):
            # Gera um único boxplot com múltiplas séries
            data = [list(map(float, s)) for s in series_list]  # garante float
            plt.figure()
            try:
                plt.boxplot(data, tick_labels=labels)
            except TypeError:
                plt.boxplot(data)
                plt.xticks(np.arange(1, len(labels) + 1), labels)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, filename), bbox_inches="tight")
            plt.close()

        # Boxplot de acurácia FINAL por fold — Train vs Test
        _box_series(
            [outer_train_accs, outer_test_accs_final],
            ["Train FINAL", "Test FINAL"],
            "Boxplot — Acurácia (FINAL) por fold",
            "boxplot_acc_train_test_final.png"
        )

        # Boxplot de tempos por fold — Treino vs Teste
        _box_series(
            [train_times, test_times],
            ["Treino (s)", "Teste (s)"],
            "Boxplot — Tempo por fold",
            "boxplot_time_train_test.png"
        )

        # Boxplot de throughput (Samples/s) no Teste
        _box_series(
            [infer_samples_per_sec],
            ["Samples/s (Test)"],
            "Boxplot — Throughput no Teste",
            "boxplot_samples_per_sec_test.png"
        )

    print("✅ Pipeline v9.8 (testes com interpretações + efeitos + CI + permutação) concluído.")
    print("📂 Resultados em :", out_root)


# ======================
# GRID helper (inner)
# ======================
def _eval_config_one(h, act, reg, X_tr_full, Y_tr_soft, y_tr, inner_splits, topk_list):
    inner_tr_accs = [];
    inner_val_accs = [];
    inner_tr_bacc = [];
    inner_val_bacc = []
    inner_tr_topk = {k: [] for k in topk_list};
    inner_val_topk = {k: [] for k in topk_list}
    inner_fit = [];
    inner_pred = []
    for tr_in, va_in in inner_splits:
        X_tr_in = X_tr_full[tr_in];
        X_va_in = X_tr_full[va_in]
        Y_tr_in = Y_tr_soft[tr_in];
        y_tr_in = y_tr[tr_in];
        y_va_in = y_tr[va_in]
        uniq, cnt = np.unique(y_tr_in, return_counts=True)
        cw = {c: (len(y_tr_in) / (len(uniq) * cnt_i)) for c, cnt_i in dict(zip(uniq, cnt)).items()}
        sw = np.array([cw[c] for c in y_tr_in], dtype=float)
        elm = ELM(X_tr_in.shape[1], int(h), Y_tr_soft.shape[1],
                  activation=str(act), reg=float(reg), random_state=RANDOM_STATE_BASE)
        t0 = time.perf_counter();
        elm.fit(X_tr_in, Y_tr_in, sample_weight=sw);
        t1 = time.perf_counter()
        y_prob_tr = elm.predict_proba(X_tr_in);
        y_pred_tr = np.argmax(y_prob_tr, axis=1)
        t2 = time.perf_counter();
        y_prob_va = elm.predict_proba(X_va_in);
        y_pred_va = np.argmax(y_prob_va, axis=1);
        t3 = time.perf_counter()
        inner_tr_accs.append(accuracy_score(y_tr_in, y_pred_tr));
        inner_val_accs.append(accuracy_score(y_va_in, y_pred_va))
        inner_tr_bacc.append(balanced_accuracy_score(y_tr_in, y_pred_tr));
        inner_val_bacc.append(balanced_accuracy_score(y_va_in, y_pred_va))
        for k in topk_list:
            inner_tr_topk[k].append(topk_accuracy_score(y_tr_in, y_prob_tr, k))
            inner_val_topk[k].append(topk_accuracy_score(y_va_in, y_prob_va, k))
        inner_fit.append(t1 - t0);
        inner_pred.append(t3 - t2)
    result = {
        "hidden_size": int(h), "activation": str(act), "reg": float(reg),
        "train_acc_mean": float(np.mean(inner_tr_accs)),
        "val_acc_mean": float(np.mean(inner_val_accs)),
        "train_bacc_mean": float(np.mean(inner_tr_bacc)),
        "val_bacc_mean": float(np.mean(inner_val_bacc)),
        "fit_time_mean_s": float(np.mean(inner_fit)),
        "val_pred_time_mean_s": float(np.mean(inner_pred)),
        **{f"train_top{k}_mean": float(np.mean(inner_tr_topk[k])) for k in topk_list},
        **{f"val_top{k}_mean": float(np.mean(inner_val_topk[k])) for k in topk_list}
    }
    stats = {"acc_tr": np.mean(inner_tr_accs), "acc_val": np.mean(inner_val_accs),
             "bacc_tr": np.mean(inner_tr_bacc), "bacc_val": np.mean(inner_val_bacc),
             "fit_time": np.mean(inner_fit), "pred_time": np.mean(inner_pred),
             "topk_tr": {k: np.mean(inner_tr_topk[k]) for k in topk_list},
             "topk_val": {k: np.mean(inner_val_topk[k]) for k in topk_list}}
    return result, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ENTRADAS
    parser.add_argument("--csv_original", default="dados/matriz_caa_ajustada.csv",
                        help="Caminho para a base ORIGINAL (sem embeddings).")
    parser.add_argument("--csv_sbert", default="dados/base_embeddings_sbert.csv",
                        help="Caminho-alvo da base PRÉ-COMPUTADA (será criada se não existir).")

    # SBERT
    parser.add_argument("--sbert_model", default="paraphrase-multilingual-mpnet-base-v2",
                        help="Modelo SBERT para gerar embeddings.")
    parser.add_argument("--sbert_batch", type=int, default=96, help="Batch size para a geração de embeddings.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Dispositivo para SBERT: auto (tenta GPU), cuda, cpu.")
    parser.add_argument("--copy_dir", type=str,
                        default=r"C:\Users\FamilyLDD\OneDrive\Doutorado\Qualificaçao\artigo reproducao\ProjetoBase\v3\examples",
                        help="Diretório para onde uma cópia da matriz_predita será enviada após criada.")

    parser.add_argument("--knn_k", type=int, default=11,
                        help="k do KNN (probabilístico, com standardization).")
    parser.add_argument("--dt_max_depth", type=int, default=0,
                        help="Profundidade máx. da Árvore (0=sem limite).")

    # TF-IDF (agora ON por padrão)
    parser.add_argument("--tfidf_dim", type=int, default=256,
                        help="Dimensão do SVD aplicado ao TF-IDF (0 desliga).")
    parser.add_argument("--tfidf_min_df", type=int, default=2,
                        help="min_df do TF-IDF (inteiro).")
    parser.add_argument("--tfidf_max_df", type=float, default=0.95,
                        help="max_df do TF-IDF (proporção).")
    parser.add_argument("--tfidf_ngram_max", type=int, default=2,
                        help="N-grama máximo (1=unigramas, 2=bigrams, etc.).")

    # PCA
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="Dimensão do PCA (TruncatedSVD). 0 desliga.")

    # TREINO
    parser.add_argument("--emb_prefix", default="SBERT_", help="Prefixo das colunas de embedding (ex.: SBERT_)")
    parser.add_argument("--outer_folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner_folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--ensemble_n", type=int, default=N_MODELOS_ENSEMBLE)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[600, 800, 1200, 2000],
                        help="Opções de neurônios escondidos do ELM.")
    parser.add_argument("--activations", type=str, nargs="+", default=["gelu", "swish", "relu", "elu", "tanh"],
                        help="Ativações candidatas do ELM.")
    parser.add_argument("--regs", type=float, nargs="+",
                        default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                        help="Regularizações candidatas do ELM.")
    parser.add_argument("--freq_weight", type=float, default=1.5,
                        help="Peso multiplicativo aplicado ao bloco *_freq antes do PCA (ex.: 1.5).")
    parser.add_argument("--inner_top_k", type=int, default=2,
                        help="Quantas melhores configs do inner usar no ensemble final (stack leve, ponderado).")

    # CALIBRAÇÃO / FUSÃO
    parser.add_argument("--sel_temp", type=float, default=1.0,
                        help="Temperatura do softmax para pesar as top-K configs.")
    parser.add_argument("--calib_holdout", type=float, default=0.15,
                        help="Proporção do treino usada para calibração (0.10–0.30 recomendado).")
    parser.add_argument("--mix_etas", type=float, nargs="+", default=[0.25, 0.50, 0.75],
                        help="Pesos para mistura prob/logit: prob*eta + logit*(1-eta).")

    # THRESHOLD TUNING (viés por classe)
    parser.add_argument("--class_bias_search", type=float, nargs="+", default=[0.9, 1.0, 1.1, 1.2, 1.3],
                        help="Multiplicadores por classe para tuning no holdout.")
    parser.add_argument("--class_bias_rounds", type=int, default=2,
                        help="Rodadas do coordinate search por classe.")

    # BLEND
    parser.add_argument("--blend_grid_step", type=float, default=0.1,
                        help="Granulação da malha simplex para calibrar pesos do blend.")

    # PARALELISMO / PERFORMANCE
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Processos paralelos p/ grid/ensemble (-1 usa todos os núcleos).")
    parser.add_argument("--blas_threads", type=int, default=0,
                        help="Limite de threads BLAS por processo (0 não limita).")

    # TEMPO de inferência
    parser.add_argument("--infer_reps", type=int, default=3,
                        help="Repetições p/ cronometrar inferência end-to-end no Test.")

    # ESTATÍSTICA
    parser.add_argument("--alpha", type=float, default=0.05, help="Nível de significância para decisões.")

    args = parser.parse_args()
    run_pipeline(args)