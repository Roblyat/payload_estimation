from __future__ import annotations

def render_evaluation(st, cfg, paths, run, pad_button, log_view):

    st.header("4) Evaluate + Combine (DeLaN + LSTM)")

    # pipeline artifacts from earlier stages
    res_out = st.session_state.get("residual_npz", "")
    lstm_model_path = st.session_state.get("lstm_model_path", "")
    lstm_scalers_path = st.session_state.get("lstm_scalers_path", "")
    H = int(st.session_state.get("H", 50))
    feature_mode = st.session_state.get("feature_mode", "full")

    lstm_dir_name = st.session_state.get("lstm_dir_name", "")
    eval_out = f"{paths.evaluation}/{lstm_dir_name}" if lstm_dir_name else f"{paths.evaluation}/_eval_tmp"

    metrics_out = f"{paths.evaluation}/_metrics_plots"   # NEW: aggregated plots live here

    e_col1, e_col2, e_col3 = st.columns([1.2, 1.2, 1.2])

    with e_col1:
        split = st.selectbox(
            "Eval split",
            ["test", "train"],
            index=0,
            help="Which split to evaluate on (reads from residual trajectory NPZ).",
        )
        
    with e_col2:
        pad_button()
        if st.button("Evaluate + combine", use_container_width=True):
            run(
                f"{cfg.COMPOSE} exec -T evaluation bash -lc "
                f"\"python3 scripts/combined_evaluation.py "
                f"--residual_npz {res_out} "
                f"--model {lstm_model_path} "
                f"--scalers {lstm_scalers_path} "
                f"--out_dir {eval_out} "
                f"--H {H} "
                f"--split {split} "
                f"--features {feature_mode} "
                f"--save_pred_npz"
                f"\""
            )

    with e_col3:
        pad_button()
        if st.button("Eval metrics plots", use_container_width=True, key="btn_eval_metrics_plots"):
            run(
                f"{cfg.COMPOSE} exec -T evaluation bash -lc "
                f"\"python3 scripts/eval_metrics_boxplots.py "
                f"--eval_root {paths.evaluation} "
                f"--out_dir {metrics_out} "
                f"\""
            )

        pad_button()
        if st.button("DeLaN metrics plots", use_container_width=True, key="btn_delan_metrics_plots"):
            run(
                f"{cfg.COMPOSE} exec -T evaluation bash -lc "
                f"\"python3 scripts/delan_metrics_boxplots.py "
                f"--delan_root {paths.models_delan} "
                f"--out_dir {paths.models_delan}/_plots "
                f"\""
            )

        pad_button()
        if st.button("LSTM metrics plots", use_container_width=True, key="btn_lstm_metrics_plots"):
            run(
                f"{cfg.COMPOSE} exec -T evaluation bash -lc "
                f"\"python3 scripts/lstm_metrics_boxplots.py "
                f"--lstm_root {paths.models_lstm} "
                f"--out_dir {paths.models_lstm}/_plots "
                f"\""
            )

    # Optional: pull extra resolved paths if you want them shown
    npz_in   = st.session_state.get("npz_in", "")
    ckpt     = st.session_state.get("delan_ckpt", "")
    win_out  = st.session_state.get("windows_npz", "")
    lstm_out = st.session_state.get("lstm_out_dir", "")

    with st.expander("Resolved paths (debug)"):
        st.write(
            {
                "npz_in": npz_in,
                "delan_ckpt": ckpt,
                "residual_npz": res_out,
                "windows_npz": win_out,
                "lstm_out_dir": lstm_out,
                "lstm_model_path": lstm_model_path,
                "lstm_scalers_path": lstm_scalers_path,
                "eval_out": eval_out,
                "feature_mode": feature_mode,
                "metrics_out": metrics_out,
            }
        )