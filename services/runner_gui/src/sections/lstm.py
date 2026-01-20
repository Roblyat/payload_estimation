from __future__ import annotations
from pathlib import Path
import sys
if "/workspace/shared/src" not in sys.path:
    sys.path.insert(0, "/workspace/shared/src")
from path_helpers import artifact_file, resolve_npz_path

def render_lstm(st, cfg, paths, run, pad_button, log_view):

    st.header("3) LSTM")

    # ---- inputs coming from earlier sections (preprocess + delan) ----
    dataset_name  = st.session_state.get("dataset_name", "ur5")
    run_tag       = st.session_state.get("run_tag", "A")
    base_id       = st.session_state.get("base_id", f"{dataset_name}__{run_tag}")

    H             = int(st.session_state.get("H", 50))
    feature_mode  = st.session_state.get("feature_mode", "full")
    delan_backend = st.session_state.get("delan_backend", "jax")

    # From DeLaN section (should be set when you trained/exported)
    delan_tag     = st.session_state.get("delan_tag", "")
    res_out = resolve_npz_path(st.session_state.get("residual_npz", ""))

    # Default window filename if preprocess section didn't already set it
    default_windows_npz_name = f"{base_id}__lstm_windows_H{H}__feat_{feature_mode}__delan_{delan_backend}.npz"

    # Use preprocess values if present
    windows_name = st.session_state.get("windows_npz_name", default_windows_npz_name)
    win_out = st.session_state.get("win_out", artifact_file(paths.processed, Path(windows_name).stem, "npz"))

    # Show it (no second text_input by default)
    st.text_input(
        f"LSTM windows ({paths.processed}/)",
        windows_name,
        disabled=True,
        key="lstm_windows_name_display",
    )
    st.caption(f"Resolved windows path: {win_out}")

    # Default seed for LSTM (do NOT depend on a local `seed` variable)
    default_seed = int(st.session_state.get("delan_seed", 4))

    # ---- ensure advanced hyperparams exist for naming on every rerun ----
    st.session_state.setdefault("lstm_units", 128)
    st.session_state.setdefault("lstm_dropout", 0.20)
    st.session_state.setdefault("lstm_val_split", 0.10)
    st.session_state.setdefault("lstm_eps", 1e-8)
    st.session_state.setdefault("lstm_no_plots", False)

    l_row1_c1, l_row1_c2, l_row1_c3, l_row1_c4 = st.columns([2.2, 1.0, 1.0, 1.2])

    with l_row1_c1:
        # Use session_state (exists on every rerun) so naming never crashes
        units_name = int(st.session_state["lstm_units"])
        dropout_name = float(st.session_state["lstm_dropout"])
        do_tag = str(dropout_name).replace(".", "p")

        # epochs/batch are defined in other columns, but their widgets also persist in session_state
        # So we still read from the local vars AFTER they are created below.
        # For now, use safe fallbacks (they will update as soon as those widgets exist).
        epochs_name = int(st.session_state.get("lstm_epochs", 60))
        batch_name  = int(st.session_state.get("lstm_batch", 64))
        seed_name   = int(st.session_state.get("lstm_seed", default_seed))

        default_lstm_dir_name = (
            f"{base_id}__{delan_tag}__feat_{feature_mode}__lstm_s{seed_name}_H{H}_ep{epochs_name}_b{batch_name}_u{units_name}_do{do_tag}"
        )

        lstm_dir_name = st.text_input(
            f"LSTM output directory ({paths.models_lstm}/)",
            default_lstm_dir_name,
            help="All LSTM outputs go here (model, scalers, plots, predictions, eval outputs).",
        )
        lstm_out = f"{paths.models_lstm}/{lstm_dir_name}"

    with l_row1_c2:
        epochs = st.number_input(
            "LSTM epochs",
            min_value=1,
            max_value=1000,
            value=60,
            step=1,
            key="lstm_epochs",
            help="Training epochs for residual LSTM.",
        )

    with l_row1_c3:
        batch = st.number_input(
            "LSTM batch size",
            min_value=1,
            max_value=4096,
            value=64,
            step=1,
            key="lstm_batch",
            help="Batch size for residual LSTM training.",
        )

    # --- NEW: LSTM advanced hyperparameters ---
    with st.expander("Advanced LSTM hyperparameters"):
        val_split = st.number_input(
            "Validation split",
            min_value=0.0,
            max_value=0.9,
            # value=0.10,
            step=0.01,
            format="%.2f",
            key="lstm_val_split",
            help="Fraction of training windows used for validation (Keras validation_split).",
        )

        units = st.number_input(
            "LSTM units",
            min_value=1,
            max_value=4096,
            # value=128,
            step=1,
            key="lstm_units",
            help="Hidden size per LSTM layer (applied to both stacked layers).",
        )

        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            # value=0.20,
            step=0.05,
            format="%.2f",
            key="lstm_dropout",
            help="Dropout applied after each LSTM layer.",
        )

        eps = st.number_input(
            "Scaler eps",
            min_value=0.0,
            # value=1e-8,
            format="%.1e",
            key="lstm_eps",
            help="Epsilon for std clamp in scalers (avoid divide-by-zero).",
        )

        no_plots = st.checkbox(
            "Disable plots (--no_plots)",
            value=False,
            key="lstm_no_plots",
            help="Skip generating loss_curve.png and residual_gt_vs_pred.png.",
        )

    with l_row1_c4:
        pad_button()
        if st.button("Build LSTM windows", use_container_width=True):
            run(
                f"{cfg.COMPOSE} exec -T preprocess bash -lc "
                f"\"python3 scripts/build_lstm_windows.py "
                f"--in_npz {res_out} "
                f"--out_npz {win_out} "
                f"--H {H} "
                f"--features {feature_mode}"
                f"\""
            )

    # NEW: separate seed for LSTM (do not reuse DeLaN seed implicitly)
    lstm_seed = st.number_input(
        "LSTM seed",
        min_value=0,
        max_value=999,
        value=default_seed,
        step=1,
        key="lstm_seed",
        help="Seed used only for naming / reproducibility on the LSTM stage.",
    )

    # Model filename = "<lstm_out_dir>.keras"
    default_model_name = f"{lstm_dir_name}.keras"
    lstm_model_name = st.text_input(
        "LSTM model filename (inside LSTM out dir)",
        default_model_name,
        help="Filename passed to train_residual_lstm.py via --model_name.",
    )

    # Train button row
    l_btn1, _lsp = st.columns([1.0, 7.0])
    with l_btn1:
        if st.button("Train LSTM", use_container_width=True):
            # optional flag
            no_plots_flag = " --no_plots" if no_plots else ""

            run(
                f"{cfg.COMPOSE} exec -T lstm bash -lc "
                f"\"python3 scripts/train_residual_lstm.py "
                f"--npz {win_out} "
                f"--out_dir {lstm_out} "
                f"--model_name {lstm_model_name} "
                f"--epochs {epochs} "
                f"--batch {batch} "
                f"--val_split {val_split} "
                f"--seed {lstm_seed} "
                f"--units {units} "
                f"--dropout {dropout} "
                f"--eps {eps}"
                f"{no_plots_flag}"
                f"\""
            )

    # Derived paths for training/eval
    lstm_model_path = f"{lstm_out}/{lstm_model_name}"
    lstm_scalers_path = f"{lstm_out}/scalers_H{H}.npz"

    st.session_state["lstm_dir_name"] = lstm_dir_name
    st.session_state["lstm_out_dir"] = lstm_out
    st.session_state["lstm_model_path"] = lstm_model_path
    st.session_state["lstm_scalers_path"] = lstm_scalers_path
    st.session_state["windows_npz"] = win_out