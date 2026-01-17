import subprocess
import streamlit as st
import sys
import time
import os

st.set_page_config(page_title="APE Pipeline Runner", layout="wide")
st.title("APE Pipeline Runner")

# persistent log buffer
if "runner_log" not in st.session_state:
    st.session_state.runner_log = ""

# Tabs: controls vs logs
tab_controls, tab_logs = st.tabs(["Controls", "Logs"])

# A global-ish placeholder for live log updates (set inside Logs tab below)
LOG_VIEW = None

# Throttle UI refresh (seconds). Logs are still fully captured in runner_log.
LOG_UI_THROTTLE_S = 0.25

# ----------------------------
# Constants (base directories)
# ----------------------------
BASE_PREPROCESSED = "/workspace/shared/data/preprocessed"
BASE_PROCESSED = "/workspace/shared/data/processed"
BASE_MODELS_DELAN = "/workspace/shared/models/delan"
BASE_MODELS_LSTM = "/workspace/shared/models/lstm"
BASE_EVALUATION = "/workspace/shared/evaluation"
FEATURE_MODES = ["full", "tau_hat", "state", "state_tauhat"]
FEATURE_HELP = {
    "full": "x_k = [q, qd, qdd, tau_hat] (dim=24)",
    "tau_hat": "x_k = [tau_hat] (dim=6)",
    "state": "x_k = [q, qd, qdd] (dim=18)",
    "state_tauhat": "x_k = [qd, qdd, tau_hat] (dim=18)",
}

COMPOSE = "docker compose -p payload_estimation --project-directory /workspace --env-file /workspace/.env -f /workspace/docker-compose.yml"

# ----------------------------
# Runner helper
# ----------------------------
def _append_log(s: str, max_chars: int = 250_000):
    st.session_state.runner_log += s
    # keep memory bounded
    if len(st.session_state.runner_log) > max_chars:
        st.session_state.runner_log = st.session_state.runner_log[-max_chars:]


def run(cmd: str):
    # keep command visible in Controls (optional)
    st.code(cmd)

    # stream output into session log + update Logs tab view
    with st.spinner("Running... (see Logs tab)"):
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_ui = time.monotonic()

        for line in p.stdout:
            _append_log(line)

            # Throttle UI updates (but keep capturing all logs)
            now = time.monotonic()
            if (LOG_VIEW is not None) and ((now - last_ui) >= LOG_UI_THROTTLE_S):
                try:
                    LOG_VIEW.code(st.session_state.runner_log)
                except Exception:
                    pass
                last_ui = now

        # Final forced update so the last lines appear immediately
        if LOG_VIEW is not None:
            try:
                LOG_VIEW.code(st.session_state.runner_log)
            except Exception:
                pass

        rc = p.wait()

    if rc != 0:
        st.error(f"Command failed (code {rc}). Check Logs tab.")
        return False

    st.success("Done.")
    return True


def pad_button():
    # Streamlit columns often mis-align buttons relative to text_inputs/number_inputs.
    # This adds a bit of vertical padding so buttons sit on the same baseline.
    st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)

with tab_logs:
    st.subheader("Runner logs")

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("Clear logs"):
            st.session_state.runner_log = ""
    with c2:
        st.download_button(
            "Download logs",
            data=st.session_state.runner_log,
            file_name="runner_gui_logs.txt",
        )

    # live view placeholder used by run()
    LOG_VIEW = st.empty()
    LOG_VIEW.code(st.session_state.runner_log)

with tab_controls:
    # ----------------------------
    # 1) Preprocess section
    # ----------------------------
    st.header("1) Preprocess")

    p_col1, p_col2, p_col3, p_col4 = st.columns([2.0, 0.8, 1.2, 1.2])

    with p_col1:
        dataset_name = st.text_input(
            "Dataset name",
            "ur5",
            help="Used to auto-construct filenames below. You can still override any filename textbox.",
        )
        run_tag = st.text_input(
            "Run tag (experiment id)",
            "A",
            help="Short identifier to distinguish multiple runs on the same dataset (e.g. A, lr3e-4, sweep1).",
        )


    with p_col2:
        H = st.number_input(
            "History length H",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            help="Sequence length for the LSTM windows (Stage 2).",
        )

    with p_col3:
        feature_mode = st.selectbox(
            "LSTM features",
            FEATURE_MODES,
            index=0,
            help="Which per-timestep features to use for LSTM windowing + combined evaluation.\n"
                + "\n".join([f"{k}: {v}" for k, v in FEATURE_HELP.items()]),
        )

    with p_col4:
        pad_button()
        build_delan_placeholder = st.button("Build DeLaN dataset (pseudo)", use_container_width=True)


    # Filenames (textbox style: base path in label, filename in input)
    base_id = f"{dataset_name}__{run_tag}"

    # File name textboxes (base path shown in label)
    default_delan_npz_name = f"delan_{dataset_name}_dataset.npz"
    default_windows_npz_name = f"{dataset_name}__{run_tag}__lstm_windows_H{H}__feat_{feature_mode}.npz"

    delan_npz_name = st.text_input(
        f"DeLaN dataset ({BASE_PREPROCESSED}/)",
        default_delan_npz_name,
        help="NPZ produced by preprocess (trajectory NPZ for DeLaN training).",
    )
    npz_in = f"{BASE_PREPROCESSED}/{delan_npz_name}"

    windows_npz_name = st.text_input(
        f"LSTM windows ({BASE_PROCESSED}/)",
        default_windows_npz_name,
        help="NPZ created by build_lstm_windows.py (X/Y windows for LSTM).",
    )
    win_out = f"{BASE_PROCESSED}/{windows_npz_name}"

    if build_delan_placeholder:
        st.info("Pseudo button: wire this later to call preprocess/scripts/build_delan_dataset.py (or your final dataset builder).")

    st.divider()

    # ----------------------------
    # 2) DeLaN section
    # ----------------------------
    st.header("2) DeLaN")

    d_row1_c1, d_row1_c2, d_row1_c3, d_row1_c4 = st.columns([1.3, 1.0, 1.2, 1.6])

    with d_row1_c1:
        structured = st.checkbox(
            "Structured (uncheck = black_box)",
            value=True,
            help="Structured = DeLaN (learns Lagrangian structure). Unchecked = black-box dynamics model.",
        )
        model_type = "structured" if structured else "black_box"

        render = st.checkbox(
            "Render plots",
            value=False,
            help="If enabled, DeLaN will try to show figures (requires X11). Plots are saved either way in the model folder.",
        )
        render_flag = 1 if render else 0

    with d_row1_c2:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999,
            value=4,
            step=1,
            help="Random seed for weight init / shuffling.",
        )

    with d_row1_c3:
        
        # ----------------------------
        # NEW: per-parameter overrides (dropdown-style)
        # ----------------------------
        # Defaults shown in UI (match your printed 'Final hyper' for default preset)
        preset_defaults = {
            "default": {
                "n_width": 64,
                "n_depth": 2,
                "n_minibatch": 512,
                "diagonal_epsilon": 0.1,
                "diagonal_shift": 2.0,
                "activation": "tanh",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "max_epoch": 2000,
            },
            "fast_debug": {
                "n_width": 64,
                "n_depth": 2,
                "n_minibatch": 512,
                "diagonal_epsilon": 0.1,
                "diagonal_shift": 2.0,
                "activation": "tanh",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "max_epoch": 200,
            },
            "long_train": {
                "n_width": 64,
                "n_depth": 2,
                "n_minibatch": 512,
                "diagonal_epsilon": 0.1,
                "diagonal_shift": 2.0,
                "activation": "tanh",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "max_epoch": 5000,
            },
        }

        # preset change in dropdown hyper parameter, override checked stay
        def apply_preset_to_widgets(preset_name: str, preset_defaults: dict):
            hp = preset_defaults[preset_name]

            # Map "preset default keys" -> (override checkbox key, value widget key)
            mapping = {
                "n_width": ("ovr_n_width", "val_n_width"),
                "n_depth": ("ovr_n_depth", "val_n_depth"),
                "n_minibatch": ("ovr_n_minibatch", "val_n_minibatch"),
                "diagonal_epsilon": ("ovr_diagonal_epsilon", "val_diagonal_epsilon"),
                "diagonal_shift": ("ovr_diagonal_shift", "val_diagonal_shift"),
                "activation": ("ovr_activation", "val_activation"),
                "learning_rate": ("ovr_learning_rate", "val_learning_rate"),
                "weight_decay": ("ovr_weight_decay", "val_weight_decay"),
                "max_epoch": ("ovr_max_epoch", "val_max_epoch"),
            }

            for k, (ovr_key, val_key) in mapping.items():
                # Only overwrite the widget value if the user is NOT overriding it
                if not st.session_state.get(ovr_key, False):
                    st.session_state[val_key] = hp[k]


        delan_preset = st.selectbox(
            "DeLaN hyperparameter preset",
            ["default", "fast_debug", "long_train"],
            index=0,
            key="delan_preset",
            on_change=lambda: apply_preset_to_widgets(st.session_state["delan_preset"], preset_defaults),
            help="Controls training hyperparameters inside your train script via --hp_preset.",
        )

        if "preset_initialized" not in st.session_state:
            apply_preset_to_widgets(st.session_state["delan_preset"], preset_defaults)
            st.session_state["preset_initialized"] = True

        hp0 = preset_defaults.get(delan_preset, preset_defaults["default"])

        # ---- NEW: init value widgets once to avoid "default + session_state" warning
        init_map = {
            "val_n_width": int(hp0["n_width"]),
            "val_n_depth": int(hp0["n_depth"]),
            "val_n_minibatch": int(hp0["n_minibatch"]),
            "val_diagonal_epsilon": float(hp0["diagonal_epsilon"]),
            "val_diagonal_shift": float(hp0["diagonal_shift"]),
            "val_activation": hp0["activation"],
            "val_learning_rate": float(hp0["learning_rate"]),
            "val_weight_decay": float(hp0["weight_decay"]),
            "val_max_epoch": int(hp0["max_epoch"]),
        }
        for k, v in init_map.items():
            st.session_state.setdefault(k, v)

        hp_override_ui = {}  # always defined

        with st.expander("Advanced hyperparameters (override preset)"):
            st.caption("Select parameters you want to override; unselected ones stay controlled by the preset.")

            hp_override_ui = {}

            def override_row_bool(key: str, label: str):
                return st.checkbox(label, value=False, key=f"ovr_{key}")

            colA, colB = st.columns([1.2, 3.0])

            with colA:
                o_n_width = override_row_bool("n_width", "Override n_width")
                o_n_depth = override_row_bool("n_depth", "Override n_depth")
                o_batch = override_row_bool("n_minibatch", "Override n_minibatch")
                o_diag_eps = override_row_bool("diagonal_epsilon", "Override diagonal_epsilon")
                o_diag_shift = override_row_bool("diagonal_shift", "Override diagonal_shift")
                o_act = override_row_bool("activation", "Override activation")
                o_lr = override_row_bool("learning_rate", "Override learning_rate")
                o_wd = override_row_bool("weight_decay", "Override weight_decay")
                o_ep = override_row_bool("max_epoch", "Override max_epoch")

            with colB:
                v_n_width = st.number_input("n_width", 1, 4096, key="val_n_width")
                v_n_depth = st.number_input("n_depth", 1, 16, key="val_n_depth")
                v_batch = st.number_input("n_minibatch", 1, 65536, key="val_n_minibatch")

                v_diag_eps = st.number_input("diagonal_epsilon", format="%.6f", key="val_diagonal_epsilon")
                v_diag_shift = st.number_input("diagonal_shift", format="%.6f", key="val_diagonal_shift")

                v_act = st.selectbox("activation", ["tanh","relu","softplus","gelu","swish"], key="val_activation")

                v_lr = st.number_input("learning_rate", format="%.8f", key="val_learning_rate")
                v_wd = st.number_input("weight_decay", format="%.8f", key="val_weight_decay")
                v_ep = st.number_input("max_epoch", 1, 200000, key="val_max_epoch")

            if o_n_width: hp_override_ui["n_width"] = v_n_width
            if o_n_depth: hp_override_ui["n_depth"] = v_n_depth
            if o_batch: hp_override_ui["n_minibatch"] = v_batch
            if o_diag_eps: hp_override_ui["diagonal_epsilon"] = v_diag_eps
            if o_diag_shift: hp_override_ui["diagonal_shift"] = v_diag_shift
            if o_act: hp_override_ui["activation"] = v_act
            if o_lr: hp_override_ui["learning_rate"] = v_lr
            if o_wd: hp_override_ui["weight_decay"] = v_wd
            if o_ep: hp_override_ui["max_epoch"] = v_ep

    with d_row1_c4:
        st.caption("DeLaN outputs are auto-derived from Dataset name / Seed / Type. Override anytime below.")

    # effective DeLaN epochs for naming (override wins, else preset)
    delan_epochs_eff = int(hp_override_ui.get("max_epoch", st.session_state.get("val_max_epoch", hp0["max_epoch"])))

    model_short = "struct" if structured else "black"

    # NEW: delan_tag has ONLY config (no dataset/run tag), so it won't double in residual/lstm names
    delan_tag = f"delan_{model_short}_s{seed}_ep{delan_epochs_eff}"

    # delan_id is ONLY for the delan run folder / checkpoint naming
    delan_id = f"delan_{dataset_name}_{run_tag}__{delan_tag}"

    delan_run_dir = f"{BASE_MODELS_DELAN}/{delan_id}"

    default_ckpt_name = f"{delan_id}.jax"
    default_residual_name = f"{base_id}__residual_{delan_tag}.npz"

    delan_ckpt_name = st.text_input(
        f"DeLaN checkpoint ({delan_run_dir}/)",
        default_ckpt_name,
        help="Checkpoint file written by DeLaN training (stored inside the run folder).",
    )
    ckpt = f"{delan_run_dir}/{delan_ckpt_name}"

    residual_name = st.text_input(
        f"Residual trajectories ({BASE_PROCESSED}/)",
        default_residual_name,
        help="Trajectory NPZ produced by export_delan_residuals_jax.py (tau_hat + residual r_tau per trajectory).",
    )
    res_out = f"{BASE_PROCESSED}/{residual_name}"

    # Buttons on same row
    d_btn1, d_btn2, _spacer = st.columns([1.0, 1.0, 6.0])
    with d_btn1:
        if st.button("Train DeLaN", use_container_width=True):

            # Build extra CLI args from UI overrides (map UI-names -> CLI-names)
            ui = hp_override_ui  # defined in the expander above
            ui_to_cli = {
                "n_width": "n_width",
                "n_depth": "n_depth",
                "n_minibatch": "batch",
                "learning_rate": "lr",
                "weight_decay": "wd",
                "max_epoch": "epochs",
                "diagonal_epsilon": "diag_eps",
                "diagonal_shift": "diag_shift",
                "activation": "activation",
            }

            hp_flags = ""
            for ui_k, v in ui.items():
                cli_k = ui_to_cli[ui_k]
                hp_flags += f" --{cli_k} '{v}'" if isinstance(v, str) else f" --{cli_k} {v}"

            run(
                f"{COMPOSE} exec -T delan_jax bash -lc "
                f"\"python3 /workspace/delan_jax/scripts/rbyt_train_delan_jax.py "
                f"--npz {npz_in} "
                f"-t {model_type} "
                f"-s {seed} "
                f"-r {render_flag} "
                f"--hp_preset {delan_preset} "
                f"{hp_flags} "
                f"--save_path {ckpt}"
                f"\""
            )

    with d_btn2:
        if st.button("Export residuals", use_container_width=True):
            run(
                f"{COMPOSE} exec -T delan_jax bash -lc "
                f"\"python3 /workspace/delan_jax/scripts/export_delan_residuals_jax.py "
                f"--npz_in {npz_in} "
                f"--ckpt {ckpt} "
                f"--out {res_out}"
                f"\""
            )

    st.divider()

    # ----------------------------
    # 3) LSTM section
    # ----------------------------
    st.header("3) LSTM")

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
        seed_name   = int(st.session_state.get("lstm_seed", int(seed)))

        default_lstm_dir_name = (
            f"{base_id}__{delan_tag}__feat_{feature_mode}__lstm_s{seed_name}_H{H}_ep{epochs_name}_b{batch_name}_u{units_name}_do{do_tag}"
        )

        lstm_dir_name = st.text_input(
            f"LSTM output directory ({BASE_MODELS_LSTM}/)",
            default_lstm_dir_name,
            help="All LSTM outputs go here (model, scalers, plots, predictions, eval outputs).",
        )
        lstm_out = f"{BASE_MODELS_LSTM}/{lstm_dir_name}"

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
            value=0.10,
            step=0.01,
            format="%.2f",
            key="lstm_val_split",
            help="Fraction of training windows used for validation (Keras validation_split).",
        )

        units = st.number_input(
            "LSTM units",
            min_value=1,
            max_value=4096,
            value=128,
            step=1,
            key="lstm_units",
            help="Hidden size per LSTM layer (applied to both stacked layers).",
        )

        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            value=0.20,
            step=0.05,
            format="%.2f",
            key="lstm_dropout",
            help="Dropout applied after each LSTM layer.",
        )

        eps = st.number_input(
            "Scaler eps",
            min_value=0.0,
            value=1e-8,
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
                f"{COMPOSE} exec -T preprocess bash -lc "
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
        value=int(seed),
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
                f"{COMPOSE} exec -T lstm bash -lc "
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

    st.divider()

    # ----------------------------
    # 4) Evaluate + combine
    # ----------------------------
    st.header("4) Evaluate + Combine (DeLaN + LSTM)")

    eval_out = f"{BASE_EVALUATION}/{lstm_dir_name}"

    e_col1, e_col2, e_col3 = st.columns([1.2, 1.2, 6.0])

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
                f"{COMPOSE} exec -T evaluation bash -lc "
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
        pass

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
            }
        )