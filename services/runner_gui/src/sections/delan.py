from __future__ import annotations

def render_delan(st, cfg, paths, run, pad_button, log_view):

    st.header("2) DeLaN")

    dataset_name = st.session_state.get("dataset_name", "ur5")
    run_tag      = st.session_state.get("run_tag", "A")
    H            = int(st.session_state.get("H", 50))
    feature_mode = st.session_state.get("feature_mode", "full")
    base_id      = st.session_state.get("base_id", f"{dataset_name}__{run_tag}")

    # resolved inputs from preprocess (with safe fallbacks)
    npz_in = st.session_state.get(
        "npz_in",
        f"{paths.preprocessed}/delan_{dataset_name}_dataset.npz",
    )

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

    st.session_state["delan_seed"] = int(seed)
    st.session_state["delan_tag"] = delan_tag

    # delan_id is ONLY for the delan run folder / checkpoint naming
    delan_id = f"delan_{dataset_name}_{run_tag}__{delan_tag}"

    delan_run_dir = f"{paths.models_delan}/{delan_id}"

    default_ckpt_name = f"{delan_id}.jax"
    default_residual_name = f"{base_id}__residual_{delan_tag}.npz"

    delan_ckpt_name = st.text_input(
        f"DeLaN checkpoint ({delan_run_dir}/)",
        default_ckpt_name,
        help="Checkpoint file written by DeLaN training (stored inside the run folder).",
    )
    ckpt = f"{delan_run_dir}/{delan_ckpt_name}"

    st.session_state["delan_id"] = delan_id
    st.session_state["delan_run_dir"] = delan_run_dir
    st.session_state["delan_ckpt"] = ckpt

    residual_name = st.text_input(
        f"Residual trajectories ({paths.processed}/)",
        default_residual_name,
        help="Trajectory NPZ produced by export_delan_residuals_jax.py (tau_hat + residual r_tau per trajectory).",
    )
    res_out = f"{paths.processed}/{residual_name}"

    st.session_state["residual_npz"] = res_out

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
                f"{cfg.COMPOSE} exec -T delan_jax bash -lc "
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
                f"{cfg.COMPOSE} exec -T delan_jax bash -lc "
                f"\"python3 /workspace/delan_jax/scripts/export_delan_residuals_jax.py "
                f"--npz_in {npz_in} "
                f"--ckpt {ckpt} "
                f"--out {res_out}"
                f"\""
            )