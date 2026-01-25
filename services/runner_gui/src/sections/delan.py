from __future__ import annotations
from pathlib import Path
import sys

if "/workspace/shared/src" not in sys.path:
    sys.path.insert(0, "/workspace/shared/src")

from path_helpers import artifact_file

def render_delan(st, cfg, paths, run, pad_button, log_view):

    st.header("2) DeLaN")

    dataset_name = st.session_state.get("dataset_name", "ur5")
    run_tag      = st.session_state.get("run_tag", "A")
    H            = int(st.session_state.get("H", 50))
    feature_mode = st.session_state.get("feature_mode", "full")
    base_id      = st.session_state.get("base_id", f"{dataset_name}__{run_tag}")

    # --- DeLaN backend (jax vs torch) ---
    delan_backend = st.selectbox(
        "DeLaN backend",
        cfg.DELAN_BACKENDS,
        index=0,  # default "jax"
        help="Choose which DeLaN implementation to train/export with.",
        key="delan_backend",
    )

    delan_service = cfg.DELAN_SERVICE[delan_backend]
    train_script = cfg.DELAN_TRAIN_SCRIPT[delan_backend]
    export_script = cfg.DELAN_EXPORT_SCRIPT[delan_backend]
    ckpt_ext = cfg.DELAN_CKPT_EXT[delan_backend]

    # resolved inputs from preprocess (with safe fallbacks)
    default_dataset_stem = f"delan_{dataset_name}_dataset"
    npz_in = st.session_state.get(
        "npz_in",
        artifact_file(paths.preprocessed, Path(default_dataset_stem).stem, "npz"),
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

        d_log_every = st.number_input(
            "Log every (epochs)",
            min_value=0,
            max_value=100000,
            value=5,
            step=1,
            help="Print/record training curves every N epochs. 0 disables periodic logging (epoch 1 still logs).",
            key="d_log_every",
        )
        d_eval_every = st.number_input(
            "Eval every (epochs)",
            min_value=0,
            max_value=100000,
            value=5,
            step=1,
            help="Evaluate on val/test every N epochs (for elbow plot). 0 disables periodic eval.",
            key="d_eval_every",
        )

        st.divider()

        d_early_stop = st.checkbox(
            "Early stopping (val_mse)",
            value=False,
            help="Stop training when val_mse hasn't improved for N evaluation events. Uses the same evaluation cadence as 'Eval every'.",
            key="d_early_stop",
        )

        d_early_stop_patience = st.number_input(
            "Early stop patience (eval events)",
            min_value=0,
            max_value=100000,
            value=10,
            step=1,
            help="How many evaluation events without improvement before stopping. With eval_every=1, this equals epochs.",
            key="d_early_stop_patience",
        )
        d_early_stop_min_delta = st.number_input(
            "Early stop min_delta",
            min_value=0.0,
            max_value=1e9,
            value=0.0,
            step=1e-6,
            format="%.6f",
            help="Minimum improvement in val_mse to reset patience. 0.0 counts any improvement.",
            key="d_early_stop_min_delta",
        )
        d_early_stop_warmup_evals = st.number_input(
            "Early stop warmup evals",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
            help="Ignore non-improving evals for the first N evaluation events (useful to skip early noise).",
            key="d_early_stop_warmup_evals",
        )

        if d_early_stop and int(d_eval_every) == 0:
            st.warning("Early stopping is enabled, but Eval every is 0 (no evals) so early stopping won't trigger.")

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
            "lutter_like": {
                "n_width": 128,
                "n_depth": 2,
                "n_minibatch": 1024,
                "diagonal_epsilon": 0.1,
                "diagonal_shift": 2.0,
                "activation": "softplus",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "max_epoch": 2000,
            },
            "lutter_like_256": {
                "n_width": 256,
                "n_depth": 2,
                "n_minibatch": 1024,
                "diagonal_epsilon": 0.1,
                "diagonal_shift": 2.0,
                "activation": "softplus",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "max_epoch": 2000,
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
            ["default", "fast_debug", "lutter_like", "lutter_like_256", "long_train"],
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

    def _fmt_hp(x: float) -> str:
        if x == 0:
            return "0"
        ax = abs(x)
        if ax < 1e-2 or ax >= 1e2:
            s = f"{x:.0e}"
        else:
            s = f"{x:g}"
        s = s.replace("+", "")
        s = s.replace("e-0", "e-").replace("e+0", "e")
        return s.replace(".", "p")

    eff_hp = dict(hp0)
    eff_hp.update(hp_override_ui)
    hp_suffix = (
        f"act{eff_hp['activation']}_b{int(eff_hp['n_minibatch'])}"
        f"_lr{_fmt_hp(float(eff_hp['learning_rate']))}"
        f"_wd{_fmt_hp(float(eff_hp['weight_decay']))}"
        f"_w{int(eff_hp['n_width'])}_d{int(eff_hp['n_depth'])}"
    )

    delan_tag = f"delan_{delan_backend}_{model_short}_s{seed}_ep{delan_epochs_eff}_{hp_suffix}"

    st.session_state["delan_seed"] = int(seed)
    st.session_state["delan_tag"] = delan_tag

    # delan_id is ONLY for the delan run folder / checkpoint naming
    # NEW: model folder includes backend so stage-1 runs never collide
    delan_id = f"{dataset_name}__{run_tag}__{delan_tag}"

    delan_run_dir = f"{paths.models_delan}/{delan_id}"

    default_ckpt_name = f"{delan_id}.{ckpt_ext}"
    
    default_residual_name = f"{base_id}__residual__{delan_tag}.npz"

    delan_ckpt_name = st.text_input(
        f"DeLaN checkpoint ({delan_run_dir}/)",
        default_ckpt_name,
        help="Checkpoint file written by DeLaN training (stored inside the run folder).",
    )
    ckpt = f"{delan_run_dir}/{delan_ckpt_name}"

    st.session_state["delan_run_dir"] = delan_run_dir

    # backend-specific ckpt key
    ckpt_state_key = f"delan_ckpt_{delan_backend}"
    st.session_state["delan_id"] = delan_id
    st.session_state["delan_tag"] = delan_tag
    st.session_state[ckpt_state_key] = ckpt

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
                f"{cfg.COMPOSE} exec -T {delan_service} bash -lc "
                f"\"python3 {train_script} "
                f"--npz {npz_in} "
                f"-t {model_type} "
                f"-s {seed} "
                f"-r {render_flag} "
                f"--hp_preset {delan_preset} "
                f"--eval_every {int(d_eval_every)} "
                f"--log_every {int(d_log_every)} "
                f"{(' --early_stop' if d_early_stop else '')} "
                f"--early_stop_patience {int(d_early_stop_patience)} "
                f"--early_stop_min_delta {float(d_early_stop_min_delta)} "
                f"--early_stop_warmup_evals {int(d_early_stop_warmup_evals)} "
                f"{hp_flags} "
                f"--save_path {ckpt}"
                f"\""
            )

    with d_btn2:
        if st.button("Export residuals", use_container_width=True):
            run(
                f"{cfg.COMPOSE} exec -T {delan_service} bash -lc "
                f"\"python3 {export_script} "
                f"--npz_in {npz_in} "
                f"--ckpt {ckpt} "
                f"--out {res_out}"
                f"\""
            )
