from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


st.set_page_config(
    page_title="Credit Risk Report",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

LABEL_COL = "loan_status"


@st.cache_data
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return (values >= 0.5).astype(int)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def trimf(x: np.ndarray, abc: List[float]) -> np.ndarray:
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    y[x == b] = 1.0

    if b != a:
        rising = (a < x) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    else:
        y[x == a] = 1.0

    if c != b:
        falling = (b < x) & (x < c)
        y[falling] = (c - x[falling]) / (c - b)
    else:
        y[x == b] = 1.0

    return np.clip(y, 0.0, 1.0)


def plot_metric_bars(metrics_df: pd.DataFrame) -> plt.Figure:
    plot_df = metrics_df.melt(
        id_vars=["stage"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="metric", y="value", hue="stage", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Stage Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    return fig


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_score_distribution(scores: pd.Series, threshold: float, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores.dropna(), bins=30, color="#0f4c5c", alpha=0.75)
    ax.axvline(threshold, color="#c06c84", linestyle="--", label=f"Threshold = {threshold:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    return fig


def load_stage_data() -> Dict[str, Dict]:
    df_stage1 = load_csv(RESULTS_DIR / "hasil_tahap1_manual_fis.csv")
    df_stage2 = load_csv(RESULTS_DIR / "hasil_tahap2_ga_fis.csv")
    df_stage3 = load_csv(RESULTS_DIR / "hasil_tahap3_ann_fis.csv")

    stages: Dict[str, Dict] = {}

    if df_stage1 is not None and LABEL_COL in df_stage1.columns and "predicted" in df_stage1.columns:
        stages["Tahap 1 - Manual FIS"] = {
            "df": df_stage1,
            "label_col": LABEL_COL,
            "pred_col": "predicted",
            "score_cols": [c for c in ["risk_score"] if c in df_stage1.columns],
        }

    if df_stage2 is not None and LABEL_COL in df_stage2.columns and "predicted_tahap2" in df_stage2.columns:
        stages["Tahap 2 - GA FIS"] = {
            "df": df_stage2,
            "label_col": LABEL_COL,
            "pred_col": "predicted_tahap2",
            "score_cols": [c for c in ["risk_score_tahap2"] if c in df_stage2.columns],
        }

    if df_stage3 is not None and LABEL_COL in df_stage3.columns and "predicted_tahap3" in df_stage3.columns:
        stages["Tahap 3 - ANN FIS"] = {
            "df": df_stage3,
            "label_col": LABEL_COL,
            "pred_col": "predicted_tahap3",
            "score_cols": [c for c in ["risk_score_tahap3", "ann_score"] if c in df_stage3.columns],
        }

    return stages


def build_predictions(
    stage_name: str,
    stage_info: Dict,
    mode: str,
    thresholds: Dict[str, float],
    score_choices: Dict[str, str],
) -> Tuple[pd.Series, pd.Series, Dict]:
    df = stage_info["df"]
    y_true = coerce_binary(df[stage_info["label_col"]])

    if mode == "Use saved predictions" or not stage_info["score_cols"]:
        y_pred = coerce_binary(df[stage_info["pred_col"]])
        return y_true, y_pred, {"mode": "saved"}

    score_col = score_choices.get(stage_name, stage_info["score_cols"][0])
    threshold = thresholds.get(stage_name, 0.5)
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    y_pred = (scores >= threshold).astype(int)
    return y_true, y_pred, {"mode": "threshold", "score_col": score_col, "threshold": threshold, "scores": scores}


def show_data_warning() -> None:
    st.warning("Result files were not found or missing expected columns. Please check the results folder.")


st.title("Credit Risk Intelligence Battle")
st.caption("Interactive report for results and model comparison")
st.caption("Luthfi Hamam Arsyada (140810230007) - Dafa Ghani Abdul Rabbani (140810230022) - Hafizh Fadhl Muhammad (140810230070)")


st.sidebar.header("Controls")
comparison_mode = st.sidebar.selectbox(
    "Prediction mode",
    ["Use saved predictions", "Use score threshold"],
)

stage_info = load_stage_data()

if not stage_info:
    show_data_warning()
    st.stop()

thresholds: Dict[str, float] = {}
score_choices: Dict[str, str] = {}

if comparison_mode == "Use score threshold":
    st.sidebar.markdown("Set thresholds per stage")
    for stage_name, info in stage_info.items():
        with st.sidebar.expander(stage_name, expanded=False):
            if info["score_cols"]:
                score_choice = st.selectbox(
                    "Score column",
                    info["score_cols"],
                    key=f"score_{stage_name}",
                )
                score_choices[stage_name] = score_choice
                threshold = st.slider(
                    "Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    key=f"threshold_{stage_name}",
                )
                thresholds[stage_name] = threshold
            else:
                st.caption("No score columns available for threshold mode.")

summary = load_json(RESULTS_DIR / "summary_tahap2.json")
ablation = load_json(RESULTS_DIR / "ablation_tahap2.json")

mf_stage_files = {
    "Tahap 1 - Manual FIS": RESULTS_DIR / "mf_params_tahap1.json",
    "Tahap 2 - GA FIS": RESULTS_DIR / "mf_params_tahap2.json",
    "Tahap 3 - ANN FIS": RESULTS_DIR / "mf_params_tahap3_ann.json",
}

mf_data = {stage: load_json(path) for stage, path in mf_stage_files.items()}


tab_overview, tab_ablation, tab_membership, tab_explorer, tab_artifacts = st.tabs(
    [
        "Overview",
        "Ablation",
        "Membership Functions",
        "Data Explorer",
        "Artifacts",
    ]
)

with tab_overview:
    st.subheader("Summary")

    if summary:
        baseline = summary.get("baseline_tahap1", {})
        tuned = summary.get("tuned_tahap2", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline accuracy", f"{baseline.get('accuracy', 0):.3f}")
            st.metric("Baseline F1", f"{baseline.get('f1_score', 0):.3f}")
        with col2:
            st.metric("Tuned accuracy", f"{tuned.get('accuracy', 0):.3f}")
            st.metric("Tuned F1", f"{tuned.get('f1_score', 0):.3f}")
        with col3:
            st.metric("GA best fitness", f"{tuned.get('best_fitness_ga', 0):.3f}")
            st.caption("GA config")
            st.json(tuned.get("ga_config", {}))
    else:
        st.info("Summary file not found. Showing metrics from result CSVs.")

    st.divider()
    st.subheader("Metrics from Result CSVs")

    metrics_rows = []
    prediction_meta = {}
    for stage_name, info in stage_info.items():
        y_true, y_pred, meta = build_predictions(stage_name, info, comparison_mode, thresholds, score_choices)
        prediction_meta[stage_name] = meta
        metrics = compute_metrics(y_true, y_pred)
        metrics_rows.append({"stage": stage_name, **metrics})

    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True)

    chart_col, metric_col = st.columns([1.2, 1])
    with chart_col:
        st.pyplot(plot_metric_bars(metrics_df))
    with metric_col:
        baseline_row = metrics_df.iloc[0] if not metrics_df.empty else None
        for _, row in metrics_df.iterrows():
            delta = None
            if baseline_row is not None and row["stage"] != baseline_row["stage"]:
                delta = row["accuracy"] - baseline_row["accuracy"]
            st.metric(
                label=f"Accuracy - {row['stage']}",
                value=f"{row['accuracy']:.3f}",
                delta=f"{delta:.3f}" if delta is not None else None,
            )

    st.divider()
    st.subheader("Confusion Matrix and Score Distribution")

    stage_options = list(stage_info.keys())
    cm_stage = st.selectbox("Select stage", stage_options, key="cm_stage")
    cm_info = stage_info[cm_stage]
    y_true_cm, y_pred_cm, meta = build_predictions(cm_stage, cm_info, comparison_mode, thresholds, score_choices)

    cm_col, dist_col = st.columns(2)
    with cm_col:
        st.pyplot(plot_confusion_matrix(y_true_cm, y_pred_cm, f"Confusion Matrix - {cm_stage}"))

    with dist_col:
        if meta.get("mode") == "threshold":
            st.pyplot(
                plot_score_distribution(
                    meta["scores"],
                    meta["threshold"],
                    f"Score Distribution - {cm_stage}",
                )
            )
        else:
            st.info("Enable threshold mode to view score distributions.")

with tab_ablation:
    st.subheader("Ablation Study")

    ablation_results = pd.DataFrame(ablation.get("results", []))
    if ablation_results.empty:
        st.info("Ablation results not found.")
    else:
        st.dataframe(ablation_results, use_container_width=True)

        setting = st.selectbox(
            "Select setting",
            ablation_results["setting"].tolist(),
            key="ablation_setting",
        )
        selected = ablation_results[ablation_results["setting"] == setting].iloc[0]

        col1, col2 = st.columns([1.2, 1])
        with col1:
            history = selected.get("best_loss_history", [])
            if history:
                fig, ax = plt.subplots(figsize=(6.5, 4))
                ax.plot(range(1, len(history) + 1), history, color="#2f6690")
                ax.set_title(f"Loss History - {setting}")
                ax.set_xlabel("Generation")
                ax.set_ylabel("Best loss")
                st.pyplot(fig)
            else:
                st.info("Loss history is missing for this setting.")

        with col2:
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.scatter(ablation_results["accuracy"], ablation_results["f1_score"], color="#6d597a")
            ax.scatter(selected["accuracy"], selected["f1_score"], color="#ee6c4d", s=80)
            ax.set_title("Accuracy vs F1")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("F1 score")
            st.pyplot(fig)

with tab_membership:
    st.subheader("Membership Function Parameters")

    available_stages = [stage for stage, data in mf_data.items() if data]
    if not available_stages:
        st.info("Membership function files were not found.")
    else:
        stage_choice = st.selectbox("Stage", available_stages, key="mf_stage")
        variables = list(mf_data[stage_choice].keys())
        if not variables:
            st.info("No variables found for this stage.")
        else:
            variable_choice = st.selectbox("Variable", variables, key="mf_var")
            term_map = mf_data[stage_choice].get(variable_choice, {})

            table_rows = [
                {"term": term, "a": values[0], "b": values[1], "c": values[2]}
                for term, values in term_map.items()
                if isinstance(values, list) and len(values) == 3
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            if table_rows:
                min_x = min(row["a"] for row in table_rows)
                max_x = max(row["c"] for row in table_rows)
                x = np.linspace(min_x, max_x, 200)

                fig, ax = plt.subplots(figsize=(7, 4))
                for row in table_rows:
                    y = trimf(x, [row["a"], row["b"], row["c"]])
                    ax.plot(x, y, label=row["term"])

                ax.set_title(f"Membership Functions - {stage_choice}")
                ax.set_xlabel(variable_choice)
                ax.set_ylabel("Membership")
                ax.legend(loc="upper right")
                st.pyplot(fig)
            else:
                st.info("No valid triangular parameters found for plotting.")

with tab_explorer:
    st.subheader("Result Explorer")

    explorer_stage = st.selectbox("Stage", list(stage_info.keys()), key="explorer_stage")
    explorer_info = stage_info[explorer_stage]
    df = explorer_info["df"].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Loan status", ["All", "0", "1"], key="status_filter")
    with col2:
        pred_filter = st.selectbox("Prediction", ["All", "0", "1"], key="pred_filter")
    with col3:
        score_options = explorer_info["score_cols"] or [explorer_info["pred_col"]]
        score_col = st.selectbox("Score column", score_options, key="score_filter")

    if status_filter != "All" and LABEL_COL in df.columns:
        df = df[df[LABEL_COL] == int(status_filter)]

    if pred_filter != "All" and explorer_info["pred_col"] in df.columns:
        df = df[df[explorer_info["pred_col"]] == int(pred_filter)]

    if score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        min_score = float(df[score_col].min())
        max_score = float(df[score_col].max())
        if max_score > min_score:
            step = max((max_score - min_score) / 100, 0.0001)
            score_range = st.slider(
                "Score range",
                min_value=min_score,
                max_value=max_score,
                value=(min_score, max_score),
                step=step,
            )
            df = df[(df[score_col] >= score_range[0]) & (df[score_col] <= score_range[1])]
        else:
            st.caption("Score range filter disabled: min and max are the same.")

    for col in ["loan_intent", "person_home_ownership", "loan_grade", "cb_person_default_on_file"]:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            selected = st.multiselect(f"Filter {col}", options, default=options, key=f"filter_{col}")
            df = df[df[col].isin(selected)]

    st.caption(f"Rows: {len(df)}")
    st.dataframe(df, use_container_width=True, height=520)

with tab_artifacts:
    st.subheader("Saved Plots")

    plots_dir = RESULTS_DIR / "plots"
    if not plots_dir.exists():
        st.info("No plot directory found.")
    else:
        image_paths = [
            path
            for path in sorted(plots_dir.rglob("*"))
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        ]

        if not image_paths:
            st.info("No images found in the plots directory.")
        else:
            image_labels = [str(path.relative_to(RESULTS_DIR)) for path in image_paths]
            choice = st.selectbox("Select image", image_labels)
            selected_path = image_paths[image_labels.index(choice)]
            st.image(str(selected_path), caption=choice, use_column_width=True)
