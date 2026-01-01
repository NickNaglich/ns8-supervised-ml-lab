"""Streamlit UI for NS8 Lab."""

import sys
from pathlib import Path

import streamlit as st
from contextlib import nullcontext

# Handle running as a script (streamlit run ui/app.py) without package context
try:  # pragma: no cover
    from ui.utils import (
        compute_feature_importance,
        compute_run_fingerprint,
        diff_configs,
        list_configs,
        index_runs,
        load_recent_runs,
        load_results_snapshot,
        load_text_file,
        load_ui_settings,
        run_command,
        load_figures,
        health_info,
        load_run_details,
        load_dataset_sample,
        render_grid_from_row,
        stream_command,
        stream_command_with_log,
        compute_dataset_signature,
        load_dataset_signature_from_run,
        clear_artifacts,
        layout_profile,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import (
        compute_feature_importance,
        compute_run_fingerprint,
        diff_configs,
        list_configs,
        index_runs,
        load_recent_runs,
        load_results_snapshot,
        load_text_file,
        load_ui_settings,
        run_command,
        load_figures,
        health_info,
        load_run_details,
        load_dataset_sample,
        render_grid_from_row,
        stream_command,
        stream_command_with_log,
        compute_dataset_signature,
        load_dataset_signature_from_run,
        clear_artifacts,
        layout_profile,
    )


def section_intro():
    st.title("NS8 Lab")
    st.markdown(
        "Browse experiments, metrics, and figures for the NS8 supervised learning lab. "
        "Default mode is read-only; run controls may be gated."
    )


def section_tasks_overview():
    st.header("Tasks")
    st.markdown(
        "- Task A: view classification\n"
        "- Task B: k-bucket classification (group-by-N recommended)\n"
        "- Task C: regression (predict N)"
    )


def section_overview_intent():
    st.subheader("What you're seeing")
    cols = st.columns(3)
    with cols[0]:
        st.info(
            "NS8 lab uses engineered features (entropy, symmetry, autocorr, FFT) from synthetic grids to study transparency over benchmarks."
        )
    with cols[1]:
        st.info(
            "Tasks: A tests invariance limits, B tests leakage via group-aware splits, C checks that structure/scale are captured."
        )
    with cols[2]:
        st.info("UI is read-only by default; browse artifacts and figures to reason about runs, not just scores.")
    st.markdown(
        "Learn more: see README sections for tasks/results/method, or open model cards in Docs.",
        help="Links appear in Docs tab; README covers tasks/results/method in more detail.",
    )


def section_visual_placeholders():
    """Phase UI-FX0 placeholders for upcoming visual widgets."""
    st.header("Planned visuals (UI-FX)")
    st.info("Dataset overview: head preview, feature histograms (entropy, symmetry, FFT bands), class balance charts.")
    st.info("Model visuals: confusion matrix heatmap, ROC/PR curves, metric trends from cv_results.")
    st.info("Experiment browser enhancements: metric-threshold filters, artifact badges with tooltips, figure thumbnails.")
    st.info("Run compare: delta bar chart for selected metric, side-by-side figures, dataset signature badge.")
    st.caption("These placeholders will be replaced as UI-FX phases land.")


def section_results():
    st.header("Results snapshot")
    results = load_results_snapshot()
    if not results:
        st.info("No results snapshot found.")
        return
    tasks = sorted({row["task"] for row in results})
    if "task_filter" not in st.session_state:
        st.session_state["task_filter"] = tasks
    task_filter = st.multiselect("Filter by task", tasks, default=st.session_state["task_filter"])
    st.session_state["task_filter"] = task_filter
    filtered = [r for r in results if r["task"] in task_filter]
    st.dataframe(filtered, use_container_width=True)


def section_results_browser(index: list, device: str, clear_cache=None):
    st.header("Results Browser")
    if not index:
        st.info("No indexed runs found.")
        return
    with st.expander("Quick tips", expanded=device.lower() != "desktop"):
        st.markdown(
            "- Browse artifacts, not just scores; artifacts column shows missing items.\n"
            "- Task B can collapse under strict (N,k) grouping; group-by-N restores baseline."
        )
    profile = layout_profile(device)
    tasks = sorted({row.get("task", "?") for row in index})
    filter_container = st.expander("Filters", expanded=True) if device.lower() == "mobile" else nullcontext()
    with filter_container:
        col_filters = st.columns(2) if profile.get("filters_in_columns") else [st]
        with col_filters[0]:
            default_tasks = st.session_state.get("task_sel", tasks)
            task_sel = st.multiselect("Tasks", options=tasks, default=default_tasks, key="tasks_filter")
        models = sorted({row.get("model", "?") for row in index})
        with col_filters[-1]:
            default_models = st.session_state.get("model_sel", models)
            model_sel = st.multiselect("Models", options=models, default=default_models, key="models_filter")
        if st.button("Apply filters"):
            st.session_state["task_sel"] = task_sel
            st.session_state["model_sel"] = model_sel
            st.experimental_rerun()
    task_sel = st.session_state.get("task_sel", task_sel)
    model_sel = st.session_state.get("model_sel", model_sel)
    filtered = [r for r in index if r.get("task") in task_sel and r.get("model") in model_sel]
    metric_vals = [r.get("primary_metric") for r in filtered if r.get("primary_metric") is not None]
    if metric_vals:
        min_m, max_m = min(metric_vals), max(metric_vals)
        thresh = st.slider("Primary metric threshold", min_value=float(min_m), max_value=float(max_m), value=float(min_m))
        filtered = [r for r in filtered if (r.get("primary_metric") or 0) >= thresh]
    sort_dir = st.radio("Sort by primary metric", options=["desc", "asc"], horizontal=True, index=0)
    filtered = sorted(filtered, key=lambda r: (r.get("primary_metric") or 0), reverse=(sort_dir == "desc"))
    default_topn = profile.get("default_topn", 0)
    topn = st.slider("Top N (0 = all)", min_value=0, max_value=max(1, len(filtered)), value=default_topn)
    if topn:
        filtered = filtered[:topn]
    sigs = {r.get("dataset_signature") for r in filtered if r.get("dataset_signature")}
    missing_sig = [r for r in filtered if not r.get("dataset_signature")]

    st.subheader("Best per task")
    best_rows = []
    for t in tasks:
        task_rows = [r for r in filtered if r.get("task") == t]
        if task_rows:
            best_rows.append(task_rows[0])
    if best_rows:
        st.dataframe(best_rows, use_container_width=True)
    else:
        st.caption("No runs available for leaderboard.")
    st.caption("Tip: Task B can collapse under strict (N,k) grouping; group-by-N restores baseline.")

    st.subheader("Runs")
    if profile.get("card_mode"):
        show_figs = st.checkbox("Show figures (may be heavy)", value=False)
        # Mobile pagination to avoid long lists
        limit = st.session_state.get("card_limit", profile.get("default_topn", len(filtered)) or len(filtered))
        subset = filtered[:limit]
        for r in subset:
            with st.expander(f"{r.get('run_id')} | {r.get('primary_metric_name')}={r.get('primary_metric')}"):
                st.caption(f"{r.get('task')} · {r.get('model')} · {r.get('source')}")
                badges = r.get("artifact_badges", {})
                missing = [k for k, v in badges.items() if not v]
                st.caption("Artifacts: " + ("all present" if not missing else f"missing {', '.join(missing)}"))
                st.caption(f"Dataset sig: {r.get('dataset_signature') or '-'}")
                st.caption("Features are engineered summaries (no raw grids).")
                det = load_run_details(r["run_path"])
                if det.get("metrics"):
                    st.json(det["metrics"])
                if show_figs and det.get("figures"):
                    with st.expander("Show figure"):
                        st.image(det["figures"][0], caption=Path(det["figures"][0]).name, width=160)
        if len(filtered) > limit:
            if st.button("Load more runs"):
                st.session_state["card_limit"] = limit + profile.get("default_topn", 10)
                st.experimental_rerun()
        else:
            st.session_state["card_limit"] = profile.get("default_topn", len(filtered))
    else:
        table_rows = []
        for r in filtered:
            badges = r.get("artifact_badges", {})
            missing = [k for k, v in badges.items() if not v]
            badge_text = "all present" if not missing else f"missing: {', '.join(missing)}"
            table_rows.append(
                {
                    "run_id": r.get("run_id"),
                    "source": r.get("source"),
                    "task": r.get("task"),
                    "model": r.get("model"),
                    "primary_metric": r.get("primary_metric"),
                    "primary_metric_name": r.get("primary_metric_name"),
                    "artifacts": badge_text,
                    "dataset_sig": r.get("dataset_signature") or "-",
                    "figs": len(r.get("figures", [])),
                    "run_path": r.get("run_path"),
                }
            )
        st.dataframe(table_rows, use_container_width=True, height=profile.get("table_height", None))
        st.caption("Artifacts column: hover to see missing items.")
        expand_default = device.lower() == "desktop"
        if st.checkbox("Expand runs inline", value=expand_default):
            for r in filtered:
                with st.expander(f"{r.get('run_id')} ({r.get('primary_metric_name')}={r.get('primary_metric')})"):
                    st.caption(f"Run path: {r.get('run_path')}")
                    badges = r.get("artifact_badges", {})
                    missing = [k for k, v in badges.items() if not v]
                    st.caption("Artifacts: " + ("all present" if not missing else f"missing {', '.join(missing)}"))
                    det = load_run_details(r["run_path"])
                    if det.get("metrics"):
                        st.json(det["metrics"])
                    if det.get("figures"):
                        st.image(det["figures"][0], caption=Path(det["figures"][0]).name, width=200)
        thumbs_default = device.lower() == "desktop"
        if st.checkbox("Show figure thumbnails (first image per run)", value=thumbs_default):
            d = device.lower()
            cols = st.columns(3) if d == "desktop" else (st.columns(2) if d == "tablet" else None)
            idx = 0
            for r in filtered:
                figs = r.get("figures", [])
                if figs:
                    target_col = cols[idx % len(cols)] if cols else st
                    with target_col:
                        st.image(figs[0], caption=f"{r.get('run_id')} ({Path(figs[0]).name})", width=220 if d == "desktop" else 180)
                        if len(figs) > 1:
                            st.caption(f"+{len(figs)-1} more figure(s) in run folder.")
                    idx += 1
    if clear_cache and st.sidebar.button("Recompute index cache"):
        clear_cache()
        st.experimental_rerun()
    if missing_sig:
        st.warning(f"{len(missing_sig)} run(s) missing dataset signatures.")
    if len(sigs) > 1:
        st.warning("Multiple dataset signatures detected across runs; compare carefully.")
    st.caption("Tip: runs live under reports/experiments/* and runs/* on disk. Open the run_path column for folders.")


def section_run_details(index: list):
    st.header("Run Details")
    if not index:
        st.info("No indexed runs found.")
        return
    with st.expander("Quick tips", expanded=False):
        st.markdown(
            "- Features are engineered summaries (no raw grids).\n"
            "- Check dataset signature if comparing runs or diagnosing drift."
        )
    run_ids = [r["run_id"] for r in index]
    run_sel = st.selectbox("Select a run", options=run_ids)
    run_entry = next((r for r in index if r["run_id"] == run_sel), None)
    if not run_entry:
        st.warning("Run not found in index.")
        return
    details = load_run_details(run_entry["run_path"])
    st.caption(f"Run folder: {run_entry['run_path']}")
    sig = load_dataset_signature_from_run(run_entry["run_path"])
    st.caption(f"Dataset signature: {sig or 'not recorded'}")

    st.subheader("Metrics")
    if details.get("metrics"):
        st.json(details["metrics"])
    else:
        st.caption("No metrics found.")
    st.subheader("Config")
    if details.get("config"):
        st.code(details["config"], language="yaml")
    else:
        st.caption("No config found.")
    st.subheader("CV Results")
    if details.get("cv_results_path"):
        st.caption(f"CV results: {details['cv_results_path']}")
        if details.get("cv_results_df") is not None:
            cv_df = details["cv_results_df"]
            st.dataframe(cv_df, use_container_width=True)
            metric_cols = [c for c in cv_df.columns if c.startswith("mean_test")]
            param_cols = [c for c in cv_df.columns if c.startswith("param_")]
            if metric_cols and param_cols:
                mcol = st.selectbox("Metric for trend", metric_cols, key="cv_metric")
                pcol = st.selectbox("Hyperparameter", param_cols, key="cv_param")
                plot_df = cv_df[[pcol, mcol]].copy().sort_values(pcol)
                st.line_chart(plot_df.set_index(pcol))
            else:
                st.caption("No metric/param columns available for trend plot.")
        else:
            st.caption("Could not load cv_results.csv.")
    else:
        st.caption("No CV results found.")
    st.subheader("Figures")
    if details.get("figures"):
        for fig in details["figures"]:
            st.image(fig, caption=Path(fig).name)
    else:
        st.caption("No figures.")
    st.caption("Features are engineered (hist/sym/autocorr/FFT), not raw grids.")
    curve_imgs = list(Path(run_entry["run_path"]).glob("*roc*.png")) + list(Path(run_entry["run_path"]).glob("*pr*.png")) + list(Path(run_entry["run_path"]).glob("*precision_recall*.png"))
    if curve_imgs:
        st.subheader("ROC / PR curves")
        for fig in curve_imgs:
            st.image(str(fig), caption=fig.name, width=320)
    else:
        st.caption("No ROC/PR curve figures found.")


def section_compare_runs(index: list):
    st.header("Compare runs")
    if not index:
        st.info("No indexed runs found.")
        return
    with st.expander("Quick tips", expanded=False):
        st.markdown(
            "- Compare runs from the same task; signatures that differ imply different data.\n"
            "- Use metric delta to spot gains; figures show where errors move."
        )
    run_ids = [r["run_id"] for r in index]
    col1, col2 = st.columns(2)
    with col1:
        run_a = st.selectbox("Run A", options=run_ids, key="compare_a")
    with col2:
        run_b = st.selectbox("Run B", options=run_ids, key="compare_b")
    if run_a == run_b:
        st.info("Select two different runs to compare.")
        return
    entry_a = next((r for r in index if r["run_id"] == run_a), None)
    entry_b = next((r for r in index if r["run_id"] == run_b), None)
    if not entry_a or not entry_b:
        st.warning("Runs not found in index.")
        return
    if entry_a.get("task") != entry_b.get("task"):
        st.warning("Runs are from different tasks; metric meanings differ.")

    details_a = load_run_details(entry_a["run_path"])
    details_b = load_run_details(entry_b["run_path"])
    sig_a = load_dataset_signature_from_run(entry_a["run_path"])
    sig_b = load_dataset_signature_from_run(entry_b["run_path"])
    st.caption(f"Run A path: {entry_a['run_path']}")
    st.caption(f"Run B path: {entry_b['run_path']}")
    st.caption(f"Run A dataset signature: {sig_a or 'missing'}")
    st.caption(f"Run B dataset signature: {sig_b or 'missing'}")
    if sig_a and sig_b and sig_a != sig_b:
        st.warning("Dataset signatures differ between runs.")
    if not sig_a or not sig_b:
        st.info("One or both runs are missing dataset signatures.")

    st.subheader("Metrics delta")
    metrics_a = details_a.get("metrics", {})
    metrics_b = details_b.get("metrics", {})
    metrics_keys = set(metrics_a.keys()) | set(metrics_b.keys())
    metric_choice = st.selectbox("Metric to highlight", sorted(metrics_keys), key="metric_choice")
    delta = None
    rows = []
    for k in metrics_keys:
        va, vb = metrics_a.get(k), metrics_b.get(k)
        if k == metric_choice and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
        rows.append({"metric": k, "run_a": va, "run_b": vb})
    st.dataframe(rows)
    if delta is not None:
        st.metric(label=f"Δ {metric_choice} (B - A)", value=delta)

    st.subheader("Dataset signature check")
    ds_path = st.text_input("Dataset path for signature check", "")
    if ds_path:
        sig = compute_dataset_signature(ds_path)
        if sig:
            st.code(f"Dataset signature: {sig}")
            cfg_a = details_a.get("config_path") or ""
            cfg_b = details_b.get("config_path") or ""
            fp_a = compute_run_fingerprint(cfg_a, sig)
            fp_b = compute_run_fingerprint(cfg_b, sig)
            st.caption(f"Run A fingerprint: {fp_a}")
            st.caption(f"Run B fingerprint: {fp_b}")
            if fp_a and fp_b and fp_a != fp_b:
                st.warning("Fingerprints differ; runs may use different config/data.")
            if (sig_a and sig and sig_a != sig) or (sig_b and sig and sig_b != sig):
                st.warning("Provided dataset signature differs from stored run signatures.")
        else:
            st.caption("Could not compute dataset signature.")
    else:
        st.caption("Dataset signatures help confirm runs use the same data; differing signatures imply different datasets.")

    st.subheader("Config diff (changed keys)")
    config_a = details_a.get("config_yaml") or details_a.get("config_json") or details_a.get("config")
    config_b = details_b.get("config_yaml") or details_b.get("config_json") or details_b.get("config")
    show_unchanged = st.checkbox("Show unchanged keys", value=False)
    diff = diff_configs(config_a, config_b)
    if diff:
        if show_unchanged:
            st.json(diff)
        else:
            st.json(diff)
    else:
        st.caption("No config differences detected.")

    st.subheader("Figures")
    figs = []
    if details_a.get("figures"):
        figs.append(("Run A", details_a["figures"]))
    if details_b.get("figures"):
        figs.append(("Run B", details_b["figures"]))
    if figs:
        for label, flist in figs:
            st.caption(label)
            for fig in flist:
                st.image(fig, caption=Path(fig).name)
    else:
        st.caption("No figures to compare.")


def section_feature_importance():
    st.header("Feature importance (RF only for now)")
    model_path = st.text_input("Model path (joblib)", "runs/latest/model.joblib")
    dataset_path = st.text_input("Dataset path (csv/parquet)", "data/sample.csv")
    feature_cols = st.text_input("Feature columns (comma-separated)", "")
    target_col = st.text_input("Target column", "target")
    save_path = st.text_input("Optional: save CSV to path", "")
    cols = [c.strip() for c in feature_cols.split(",") if c.strip()]
    if st.button("Compute importance"):
        if not cols:
            st.warning("Provide feature columns.")
            return
        df = compute_feature_importance(model_path, dataset_path, cols, target_col, save_path or None)
        if df is None:
            st.info("No importance available (model may not support it or files missing).")
        else:
            st.dataframe(df.sort_values("importance", ascending=False))
            dest = save_path or str(Path(model_path).parent / "feature_importance.csv")
            st.caption(f"Saved to {dest}")
            st.download_button("Download CSV", data=df.to_csv(index=False), file_name="feature_importance.csv")

def section_runs():
    st.header("Recent runs")
    runs = load_recent_runs()
    if not runs:
        st.info("No runs found in reports/experiments.")
        return
    task_options = sorted({run.get("task", "?") for run in runs})
    if "run_task_sel" not in st.session_state:
        st.session_state["run_task_sel"] = task_options[0] if task_options else ""
    task_sel = st.selectbox("Filter runs by task", options=task_options, index=task_options.index(st.session_state["run_task_sel"]))
    st.session_state["run_task_sel"] = task_sel
    filtered = [r for r in runs if r.get("task") == task_sel]
    for run in filtered:
        title = f"{run['timestamp']} - {run.get('task','?')} - {run['model']}"
        with st.expander(title):
            if run["metrics"]:
                st.json(run["metrics"])
            else:
                st.caption("No metrics.json found.")
            if run.get("config_text"):
                st.code(run["config_text"], language="yaml")
            else:
                st.caption("No config.yaml found.")
            if run.get("run_path"):
                st.caption(f"Run folder: {run['run_path']}")
            if run.get("figures"):
                for fig in run["figures"]:
                    st.image(fig, caption=Path(fig).name)
            else:
                st.caption("No figures found.")


def section_run_controls(mode: str):
    st.header("Run controls")
    if mode != "run":
        st.info("Run mode is disabled. Set mode: run in ui/config.yaml to enable.")
        return
    st.warning("Runs may take time/resources. Prefer configs/task_b_groupN.yaml for Task B (group-N).")
    configs = list_configs()
    if not configs:
        st.warning("No configs found in configs/.")
        return
    config_choice = st.selectbox("Select a tuning config", options=configs)
    st.caption("Dry-run command")
    st.code(f"python -m ns8lab.cli tune --config {config_choice}", language="bash")
    if "task_b" in Path(config_choice).name and "groupN" not in Path(config_choice).name:
        st.info("For Task B, recommended config is group-N: configs/task_b_groupN.yaml")
    confirm = st.checkbox("I understand and want to run commands", value=False)
    persist_logs = st.checkbox("Save UI logs to runs/ui_logs", value=False)
    dry_run = st.checkbox("Dry-run only (show command, do not execute)", value=False)
    st.divider()
    if st.button("Quick baseline (Task A)", disabled=not confirm):
        cmd = ["python", "-m", "ns8lab.cli", "train", "--task", "task_a_view", "--samples", "400", "--seed", "0"]
        st.code(" ".join(cmd), language="bash")
        if not dry_run:
            runner = stream_command_with_log if persist_logs else stream_command
            for line in runner(cmd):
                st.write(line)
    if st.button("Run tune", disabled=not confirm):
        st.write("Executing... (this may take time)")
        cmd = ["python", "-m", "ns8lab.cli", "tune", "--config", config_choice]
        st.code(" ".join(cmd), language="bash")
        if not dry_run:
            runner = stream_command_with_log if persist_logs else stream_command
            for line in runner(cmd):
                st.write(line)
    st.divider()
    if st.button("Run tests", disabled=not confirm):
        st.write("Executing tests...")
        cmd = ["python", "-m", "pytest"]
        st.code(" ".join(cmd), language="bash")
        if not dry_run:
            runner = stream_command_with_log if persist_logs else stream_command
            for line in runner(cmd):
                st.write(line)


def section_docs():
    st.header("Docs and cards")
    readme_text = load_text_file("README.md")
    plan_text = load_text_file("PHASED-PLAN.md")
    model_cards = load_text_file("reports/results/model_cards.md")
    tabs = st.tabs(["README", "PHASED-PLAN", "Model Cards"])
    with tabs[0]:
        st.markdown(readme_text or "README.md not found.")
    with tabs[1]:
        st.markdown(plan_text or "PHASED-PLAN.md not found.")
    with tabs[2]:
        st.markdown(model_cards or "Model cards not found.")


def section_figures():
    st.header("Figures at a glance")
    sources = st.multiselect(
        "Figure sources",
        options=["reports/figures", "reports/experiments", "runs"],
        default=["reports/figures", "reports/experiments"],
    )
    recursive = st.checkbox("Include subfolders (runs/experiments)", value=True)
    filter_text = st.text_input("Filter by name (optional)", "")
    figs = load_figures(fig_dirs=sources, recursive=recursive)
    if filter_text:
        figs = [f for f in figs if filter_text.lower() in Path(f).name.lower()]
    if not figs:
        st.info("No figures found for the selected sources/filter.")
        return
    limit = st.slider("Max figures to show", min_value=1, max_value=max(1, len(figs)), value=min(30, len(figs)))
    for fig in figs[:limit]:
        st.image(fig, caption=Path(fig).name)


def section_repro():
    st.header("How to reproduce")
    cmds = {
        "Install deps": "python -m pip install -e .[dev]",
        "Dataset": "python -m ns8lab.cli dataset --task task_a_view --n-samples 200 --output data/task_a.csv",
        "Train (Task A)": "python -m ns8lab.cli train --task task_a_view --samples 400 --seed 0",
        "Train (Task C)": "python -m ns8lab.cli train --task task_c_regress_n --samples 400 --seed 0",
        "Tune (Task B, recommended)": "python -m ns8lab.cli tune --config configs/task_b_groupN.yaml",
    }
    for label, cmd in cmds.items():
        st.caption(label)
        st.code(cmd, language="bash")


def section_health(settings: dict):
    st.header("Health / status")
    info = health_info(settings)
    st.json({"mode": info.get("mode"), "versions": {k: v for k, v in info.items() if k.endswith("_version")}})
    st.subheader("Paths")
    for name, data in info.get("paths", {}).items():
        st.caption(f"{name}: {data['path']} (exists: {data['exists']})")
    st.subheader("Fingerprint")
    fp = info.get("fingerprint", {})
    if fp.get("computed"):
        st.json(fp)
    else:
        st.caption("Fingerprint not computed yet. Provide a dataset path to compute signature.")
        ds_path = st.text_input("Dataset path for signature", "")
        if ds_path:
            sig = compute_dataset_signature(ds_path)
            if sig:
                st.code(f"Dataset signature: {sig}")
                cfg_path = st.text_input("Config path for fingerprint", "configs/task_a.yaml")
                if cfg_path:
                    fp_hash = compute_run_fingerprint(cfg_path, sig)
                    if fp_hash:
                        st.code(f"Run fingerprint: {fp_hash}")
                    else:
                        st.caption("Could not compute run fingerprint.")
            else:
                st.caption("Could not compute signature.")
    st.subheader("Session state")
    st.json({k: v for k, v in st.session_state.items() if k not in {"_session_state" }})
    st.subheader("Danger zone: Clear artifacts")
    st.caption("Deletes runs/ and reports/experiments (recreates empty folders). Use sparingly.")
    confirm = st.checkbox("I understand this deletes run artifacts", value=False)
    really = st.text_input("Type CLEAR to confirm", value="")
    if st.button("Clear runs and experiments", disabled=not confirm or really != "CLEAR"):
        cleared = clear_artifacts()
        st.success(f"Cleared: {cleared}")


def section_dataset_explorer():
    st.header("Dataset Explorer")
    with st.expander("Quick tips", expanded=False):
        st.markdown(
            "- Data uses engineered features to force structural learning; raw grids are not trained.\n"
            "- Render grids to sanity-check (N, k, view), but models see the summaries above."
        )
    default_path = st.session_state.get("dataset_path", "data/sample.csv")
    sample_path = st.text_input("Dataset sample path (csv/parquet)", default_path, key="dataset_path")
    st.caption("Tip: generate a sample with `python -m ns8lab.cli dataset --task task_a_view --n-samples 400 --output data/sample.csv`.")

    @st.cache_data
    def cached_dataset(path: str):
        return load_dataset_sample(path)

    df = cached_dataset(sample_path)
    if df is None:
        st.info("No dataset loaded. Provide a valid csv/parquet path, then Refresh.")
        return
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Targets")
    if "target" in df.columns:
        counts = df["target"].value_counts()
        st.bar_chart(counts)
    else:
        st.caption("No target column found.")
    st.caption("Grids are reduced to engineered features for training; raw grids are not used.")

    st.subheader("Feature histograms (optional)")
    try:
        import pandas as pd

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        preferred = [c for c in num_cols if any(key in c.lower() for key in ["entropy", "sym", "fft", "band", "energy"])]
        default_cols = preferred[:4] if preferred else num_cols[:3]
        hist_cols = st.multiselect("Numeric columns", options=num_cols, default=default_cols, help="Pick key engineered features (entropy, symmetry, FFT bands).")
        for col in hist_cols:
            st.bar_chart(df[col].value_counts().sort_index())
    except Exception:
        st.caption("Could not render histograms.")

    st.subheader("Render NS8 grid from selected row")
    if {"N", "k", "view"}.issubset(df.columns):
        idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        row = df.iloc[int(idx)].to_dict()
        grid = render_grid_from_row(row)
        if grid is not None:
            st.write(f"Grid shape: {grid.shape}, view: {row.get('view')}")
            st.dataframe(grid)
        else:
            st.caption("Unable to render grid.")
    else:
        st.caption("Columns N, k, view required to render grids.")

    st.subheader("Pair plot (sampled, optional)")
    try:
        import pandas as pd

        sample_df = df.sample(min(len(df), 200), random_state=0)
        num_cols = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]
        if len(num_cols) >= 2:
            pair_cols = st.multiselect("Select 2 numeric features", num_cols, default=num_cols[:2])
            if len(pair_cols) == 2:
                st.scatter_chart(sample_df, x=pair_cols[0], y=pair_cols[1])
            else:
                st.caption("Pick exactly 2 columns for scatter.")
        else:
            st.caption("Not enough numeric columns for scatter.")
    except Exception:
        st.caption("Could not render pair plot.")


def main():
    settings = load_ui_settings()
    st.sidebar.markdown(f"Mode: **{settings.get('mode', 'read-only')}**")
    device = st.sidebar.selectbox("Layout (device)", options=["Desktop", "Tablet", "Mobile"], index=0, key="layout_device")
    density_choice = st.sidebar.selectbox("Density", options=["Compact", "Comfortable"], index=0, key="layout_density")
    profile = layout_profile(device, density=density_choice)
    if device.lower() == "desktop":
        st.set_page_config(layout="wide")
    # Apply spacing/font density
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-left: {profile['padding_px']}px; padding-right: {profile['padding_px']}px; }}
        .stDataFrame, .stMetric, .stSelectbox, .stSlider, .stRadio, .stMultiselect, .stTextInput {{
            font-size: {profile['font_px']}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<a name='top'></a>", unsafe_allow_html=True)
    # Persist page selection
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview"
    pages = {
        "Overview": "overview",
        "Results Browser": "browser",
        "Run Details": "details",
        "Compare": "compare",
        "Runs": "runs",
        "Dataset Explorer": "dataset",
        "Feature Importance": "importance",
        "Run Controls": "run_controls",
        "Docs": "docs",
        "Figures": "figures",
        "Reproduce": "repro",
        "Health": "health",
    }
    page = st.sidebar.radio("Navigate", list(pages.keys()), index=list(pages.keys()).index(st.session_state["nav_page"]))
    st.session_state["nav_page"] = page

    # Cache index
    @st.cache_data
    def cached_index():
        return index_runs()

    if st.sidebar.button("Refresh index"):
        cached_index.clear()
    index = cached_index()

    if pages[page] == "overview":
        section_intro()
        section_tasks_overview()
        section_overview_intent()
        section_results()
    elif pages[page] == "browser":
        section_results_browser(index, device, clear_cache=cached_index.clear)
    elif pages[page] == "details":
        section_run_details(index)
    elif pages[page] == "compare":
        section_compare_runs(index)
    elif pages[page] == "dataset":
        section_dataset_explorer()
    elif pages[page] == "importance":
        section_feature_importance()
    elif pages[page] == "runs":
        section_runs()
    elif pages[page] == "run_controls":
        section_run_controls(settings.get("mode", "read-only"))
    elif pages[page] == "docs":
        section_docs()
    elif pages[page] == "figures":
        section_figures()
    elif pages[page] == "repro":
        section_repro()
    elif pages[page] == "health":
        section_health(settings)
    st.markdown("[Back to top](#top)", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
