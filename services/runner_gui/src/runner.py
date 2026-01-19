from __future__ import annotations
import subprocess
import time
import streamlit as st

from config import AppConfig
from paths import make_paths

from sections.preprocess import render_preprocess
from sections.delan import render_delan
from sections.lstm import render_lstm
from sections.evaluation import render_evaluation

LOG_UI_THROTTLE_S = 0.25

def app_main():
    st.set_page_config(page_title="Algorithmic Payload Estimation", layout="wide")
    st.title("Algorithmic Payload Estimation (Pipeline Runner)")

    if "runner_log" not in st.session_state:
        st.session_state.runner_log = ""

    tab_controls, tab_logs = st.tabs(["Controls", "Logs"])
    log_view = None

    def _append_log(s: str, max_chars: int = 250_000):
        st.session_state.runner_log += s
        if len(st.session_state.runner_log) > max_chars:
            st.session_state.runner_log = st.session_state.runner_log[-max_chars:]

    def run(cmd: str):
        st.code(cmd)
        with st.spinner("Running... (see Logs tab)"):
            p = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            last_ui = time.monotonic()
            for line in p.stdout:
                _append_log(line)
                now = time.monotonic()
                if log_view is not None and (now - last_ui) >= LOG_UI_THROTTLE_S:
                    log_view.code(st.session_state.runner_log)
                    last_ui = now
            rc = p.wait()
            if log_view is not None:
                log_view.code(st.session_state.runner_log)
        return rc == 0

    def pad_button():
        st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)

    with tab_logs:
        st.subheader("Runner logs")
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("Clear logs"):
                st.session_state.runner_log = ""
        with c2:
            st.download_button("Download logs", st.session_state.runner_log, file_name="runner_gui_logs.txt")

        log_view = st.empty()
        log_view.code(st.session_state.runner_log)

    cfg = AppConfig()
    paths = make_paths(cfg)

    with tab_controls:
        render_preprocess(st, cfg, paths, run, pad_button, log_view)
        st.divider()
        render_delan(st, cfg, paths, run, pad_button, log_view)
        st.divider()
        render_lstm(st, cfg, paths, run, pad_button, log_view)
        st.divider()
        render_evaluation(st, cfg, paths, run, pad_button, log_view)