"""Run:  streamlit run app/streamlit_app.py      (bound to 127.0.0.1 via .streamlit/config.toml)"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kycrag.generate import LLMGenerator  # noqa: E402
from kycrag.pipeline import KYCAssistant  # noqa: E402
from kycrag.retrieve import ABLATIONS, Retriever  # noqa: E402

st.set_page_config(page_title="KYC rules assistant", layout="centered")

STATUS_LABEL = {
    "found": ("Answered from the corpus", "green"),
    "partial": ("Partly answered — some of this isn't in the corpus", "orange"),
    "not_found": ("Not in the corpus", "gray"),
    "refused": ("Outside this assistant's scope", "gray"),
    "unverified": ("Citations failed verification", "red"),
}


@st.cache_resource
def get_assistant(index: str) -> KYCAssistant:
    return KYCAssistant(Retriever(index), LLMGenerator())


with st.sidebar:
    st.subheader("Settings")
    cfg_name = st.selectbox("Retrieval configuration", [c.name for c in ABLATIONS], index=len(ABLATIONS) - 1,
                            help="Switch configs to demo the ablation study live.")
    cfg = next(c for c in ABLATIONS if c.name == cfg_name)
    st.caption("Study aid over public RBI text. Not compliance or legal advice. "
               "Don't enter customer details; they are redacted before anything is sent.")

st.title("KYC rules assistant")
st.write("Ask about Indian bank KYC rules. Every answer cites the RBI paragraph it relies on.")

question = st.text_input("Your question", placeholder="What are the limits on a small account?")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        assistant = get_assistant(cfg.index)
    except FileNotFoundError:
        st.error("No index found. Run `python -m kycrag.cli build` first.")
        st.stop()
    with st.spinner("Searching the directions…"):
        res = assistant.ask(question, cfg)

    if res["pii_found"]:
        st.warning(f"Removed personal details before searching: {', '.join(res['pii_found'])}. "
                   "Please ask about the rule, not a specific customer.")
    label, colour = STATUS_LABEL.get(res["status"], (res["status"], "gray"))
    st.markdown(f":{colour}[**{label}**]")
    st.markdown(res["answer"])
    if res.get("conflicts"):
        st.info(f"Source conflict: {res['conflicts']}")

    hits = {h["chunk_id"]: h for h in res["hits"]}
    if res["citations"]:
        st.subheader("Sources cited")
        for cid in res["citations"]:
            h = hits.get(cid)
            if not h:
                continue
            tag = " · superseded" if h["status"] == "superseded" else ""
            tag += " · bank policy" if h["authority"] == "internal" else ""
            with st.expander(f"{h['doc_id']}, para {h['para_ref']}{tag}"):
                st.caption(h["heading_path"])
                st.text(h["text"])
    with st.expander("All retrieved passages (debug)"):
        for h in res["hits"]:
            st.write(f"`{h['chunk_id']}` · {h['via']} · score {h['score']}")
    st.caption(f"Config {res['config']} · corpus {res['corpus_version']}")
