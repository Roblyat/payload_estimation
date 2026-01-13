import subprocess
import streamlit as st
import sys
import time

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

    p_col1, p_col2, p_col3 = st.columns([2.2, 1.0, 1.2])

    with p_col1:
        dataset_name = st.text_input(
            "Dataset name",
            "ur5",
            help="Used to auto-construct filenames below. You can still override any filename textbox.",
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
        pad_button()
        build_delan_placeholder = st.button("Build DeLaN dataset (pseudo)", use_container_width=True)

    # File name textboxes (base path shown in label)
    default_delan_npz_name = f"delan_{dataset_name}_dataset.npz"
    default_windows_npz_name = f"{dataset_name}_lstm_windows_H{H}.npz"

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
        delan_preset = st.selectbox(
            "DeLaN hyperparameter preset",
            ["default", "fast_debug", "long_train"],
            index=0,
            help="Controls training hyperparameters inside your train script via --hp_preset.",
        )

    with d_row1_c4:
        st.caption("DeLaN outputs are auto-derived from Dataset name / Seed / Type. Override anytime below.")

    # Filenames (textbox style: base path in label, filename in input)
    default_ckpt_name = f"delan_{dataset_name}_{'struct' if structured else 'blackbox'}_seed{seed}.jax"
    default_residual_name = f"{dataset_name}_residual_traj.npz"

    delan_ckpt_name = st.text_input(
        f"DeLaN checkpoint ({BASE_MODELS_DELAN}/)",
        default_ckpt_name,
        help="Checkpoint file written by DeLaN training.",
    )
    ckpt = f"{BASE_MODELS_DELAN}/{delan_ckpt_name}"

    residual_name = st.text_input(
        f"Residual trajectories ({BASE_PROCESSED}/)",
        default_residual_name,
        help="Trajectory NPZ produced by export_ur5_residuals_jax.py (tau_hat + residual r_tau per trajectory).",
    )
    res_out = f"{BASE_PROCESSED}/{residual_name}"

    # Buttons on same row
    d_btn1, d_btn2, _spacer = st.columns([1.0, 1.0, 6.0])
    with d_btn1:
        if st.button("Train DeLaN", use_container_width=True):
            run(
                f"{COMPOSE} exec -T delan_jax bash -lc "
                f"\"python3 -m deep_lagrangian_networks.rbyt_train_delan_jax "
                f"--npz {npz_in} "
                f"-t {model_type} "
                f"-s {seed} "
                f"-r {render_flag} "
                f"--hp_preset {delan_preset} "
                f"--save_path {ckpt}"
                f"\""
            )

    with d_btn2:
        if st.button("Export residuals", use_container_width=True):
            run(
                f"{COMPOSE} exec -T delan_jax bash -lc "
                f"\"python3 -m deep_lagrangian_networks.export_ur5_residuals_jax "
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

    l_row1_c1, l_row1_c2, l_row1_c3, l_row1_c4 = st.columns([2.2, 1.0, 1.0, 1.2])

    with l_row1_c1:
        default_lstm_dir_name = f"residual_lstm_H{H}_scaled"
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
            help="Training epochs for residual LSTM.",
        )

    with l_row1_c3:
        batch = st.number_input(
            "LSTM batch size",
            min_value=1,
            max_value=4096,
            value=64,
            step=1,
            help="Batch size for residual LSTM training.",
        )

    with l_row1_c4:
        pad_button()
        if st.button("Build LSTM windows", use_container_width=True):
            run(
                f"{COMPOSE} exec -T preprocess bash -lc "
                f"\"python3 scripts/build_lstm_windows.py "
                f"--in_npz {res_out} "
                f"--out_npz {win_out} "
                f"--H {H}"
                f"\""
            )

    # Model filename
    default_model_name = f"best_seed{seed}_H{H}.keras"
    lstm_model_name = st.text_input(
        "LSTM model filename (inside LSTM out dir)",
        default_model_name,
        help="Filename passed to train_residual_lstm.py via --model_name.",
    )

    # Train button row
    l_btn1, _lsp = st.columns([1.0, 7.0])
    with l_btn1:
        if st.button("Train LSTM", use_container_width=True):
            run(
                f"{COMPOSE} exec -T lstm bash -lc "
                f"\"python3 scripts/train_residual_lstm.py "
                f"--npz {win_out} "
                f"--out_dir {lstm_out} "
                f"--model_name {lstm_model_name} "
                f"--epochs {epochs} "
                f"--batch {batch}"
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
                f"{COMPOSE} exec -T lstm bash -lc "
                f"\"python3 scripts/evaluate_and_combine.py "
                f"--residual_npz {res_out} "
                f"--model {lstm_model_path} "
                f"--scalers {lstm_scalers_path} "
                f"--out_dir {lstm_out}/eval_combined "
                f"--H {H} "
                f"--split {split} "
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
            }
        )
