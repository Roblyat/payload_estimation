from __future__ import annotations

def render_preprocess(st, cfg, paths, run, pad_button, log_view):

    st.header("1) Preprocess")

    def _safe_tag(x: float) -> str:
        return str(x).replace(".", "p")

    p_col1, p_col2, p_col3, p_col4 = st.columns([2.0, 0.8, 1.2, 1.2])

    raw_data_path = "/workspace/shared/data/raw"
    in_format = "long"
    derive_qdd = True

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
        test_fraction = st.number_input(
            "Test fraction",
            min_value=0.05,
            max_value=0.9,
            value=0.2,
            step=0.05,
            format="%.2f",
            help="Fraction of trajectories used for test split.",
        )
        val_fraction = st.number_input(
            "Val fraction",
            min_value=0.0,
            max_value=0.9,
            value=0.1,
            step=0.05,
            format="%.2f",
            help="Fraction of trajectories used for validation split.",
        )


    with p_col2:
        derive_qdd = st.checkbox(
            "Derive Acceleration",
            value=True,
            help="If Acceleration column is not provided in the raw CSV, derive it from Velocity via numerical differentiation.",
        )
        filter_accel = st.checkbox(
            "Filter q/qd/tau",
            value=False,
            help="Apply a Butterworth low-pass filter to q, qd, and tau (and optionally qdd).",
        )
        filter_cutoff_hz = st.number_input(
            "Filter cutoff (Hz)",
            min_value=0.1,
            max_value=200.0,
            value=20.0,
            step=1.0,
            help="Low-pass cutoff frequency for filtering.",
        )
        filter_order = st.number_input(
            "Filter order",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Butterworth filter order.",
        )
        filter_qdd = st.checkbox(
            "Filter qdd",
            value=True,
            help="Apply the same low-pass filter to qdd after derivation.",
        )
        H = st.number_input(
            "History length H",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            help="Sequence length for the LSTM windows (Stage 2).",
        )
        in_format = st.selectbox(
            "Input format",
            cfg.INPUT_FORMATS,
            index=0,
            help="Format of the raw data file.",
        )

    with p_col3:
        col_format = st.selectbox(
            "CSV column format",
            cfg.COLUMN_FORMATS,
            index=0,
            help="Format of the raw CSV file:\n"
                "- long: rows are (Time, Joint Name, Position, Velocity, Acceleration, Effort)\n"
                "- wide: rows are frames with q1..q6, dq1..dq6, Iq1..Iq6, etc. (converted to long internally)",
        )
        feature_mode = st.selectbox(
            "LSTM features",
            cfg.FEATURE_MODES,
            index=0,
            help="Which per-timestep features to use for LSTM windowing + combined evaluation.\n"
            + "\n".join([f"{k}: {v}" for k, v in cfg.FEATURE_HELP.items()]),
        )

    with p_col4:
        pad_button()
        if st.button("Preprocess DeLaN", use_container_width=True):
            out_npz_name = f"delan_{dataset_name}_tf{_safe_tag(test_fraction)}_vf{_safe_tag(val_fraction)}_dataset.npz"
            run(
                f"{cfg.COMPOSE} exec -T preprocess bash -lc "
                f"\"python3 scripts/build_delan_dataset.py "
                f"--qdd {derive_qdd} "
                f"--col_format {col_format} "
                f"--test_fraction {test_fraction} "
                f"--val_fraction {val_fraction} "
                f"--filter_accel {filter_accel} "
                f"--filter_cutoff_hz {filter_cutoff_hz} "
                f"--filter_order {filter_order} "
                f"--filter_qdd {filter_qdd} "
                f"--raw_csv {raw_data_path}/{dataset_name}.{in_format} "
                f"--out_npz {paths.preprocessed}/{out_npz_name} "
                f"\""
            )

    # Filenames (textbox style: base path in label, filename in input)
    base_id = f"{dataset_name}__{run_tag}"

    # File name textboxes (base path shown in label)
    delan_backend = st.session_state.get("delan_backend", "jax")

    default_delan_npz_name = (
        f"delan_{dataset_name}_tf{_safe_tag(test_fraction)}_vf{_safe_tag(val_fraction)}_dataset.npz"
    )
    default_windows_npz_name = f"{base_id}__lstm_windows_H{H}__feat_{feature_mode}__delan_{delan_backend}.npz"

    delan_npz_name = st.text_input(
        f"DeLaN dataset ({paths.preprocessed}/)",
        default_delan_npz_name,
        help="NPZ produced by preprocess (trajectory NPZ for DeLaN training).",
    )
    npz_in = f"{paths.preprocessed}/{delan_npz_name}"

    windows_npz_name = st.text_input(
        f"LSTM windows ({paths.processed}/)",
        default_windows_npz_name,
        help="NPZ created by build_lstm_windows.py (X/Y windows for LSTM).",
    )
    win_out = f"{paths.processed}/{windows_npz_name}"

    st.session_state["dataset_name"] = dataset_name
    st.session_state["run_tag"] = run_tag
    st.session_state["H"] = int(H)
    st.session_state["feature_mode"] = feature_mode
    st.session_state["test_fraction"] = float(test_fraction)
    st.session_state["val_fraction"] = float(val_fraction)
    st.session_state["filter_accel"] = bool(filter_accel)
    st.session_state["filter_cutoff_hz"] = float(filter_cutoff_hz)
    st.session_state["filter_order"] = int(filter_order)
    st.session_state["filter_qdd"] = bool(filter_qdd)
    st.session_state["base_id"] = f"{dataset_name}__{run_tag}"
    st.session_state["npz_in"] = npz_in
    st.session_state["win_out"] = win_out
    st.session_state["windows_npz_name"] = windows_npz_name
    st.session_state["delan_npz_name"] = delan_npz_name
