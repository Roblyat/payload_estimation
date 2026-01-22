from __future__ import annotations

def render_preprocess(st, cfg, paths, run, pad_button, log_view):

    st.header("1) Preprocess")

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


    with p_col2:
        derive_qdd = st.checkbox(
            "Derive Acceleration",
            value=True,
            help="If Acceleration column is not provided in the raw CSV, derive it from Velocity via numerical differentiation.",
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
            run(
                f"{cfg.COMPOSE} exec -T preprocess bash -lc "
                f"\"python3 scripts/build_delan_dataset.py "
                f"--qdd {derive_qdd} "
                f"--col_format {col_format} "
                f"--raw_csv {raw_data_path}/{dataset_name}.{in_format} "
                f"--out_npz {paths.preprocessed}/delan_{dataset_name}_dataset.npz "
                f"\""
            )

    # Filenames (textbox style: base path in label, filename in input)
    base_id = f"{dataset_name}__{run_tag}"

    # File name textboxes (base path shown in label)
    delan_backend = st.session_state.get("delan_backend", "jax")

    default_delan_npz_name = f"delan_{dataset_name}_dataset.npz"
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
    st.session_state["base_id"] = f"{dataset_name}__{run_tag}"
    st.session_state["npz_in"] = npz_in
    st.session_state["win_out"] = win_out
    st.session_state["windows_npz_name"] = windows_npz_name
    st.session_state["delan_npz_name"] = delan_npz_name