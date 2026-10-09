# app.py  可信心理健康知识问答（Web）
import os, sys
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import engine, generation

st.set_page_config(page_title="可信心理健康知识问答", page_icon="🧠", layout="centered")
st.title("🧠 可信心理健康知识问答")
st.caption("每句有出处 · 没依据就拒答 · 仅科普，不替代就医")

if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("例如：吃抗抑郁药会上瘾吗？")
if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("检索资料并生成…"):
            engine.init()
            r = generation.answer(q)
            st.markdown(r["answer"])
            if r.get("contexts"):
                with st.expander("引用资料（可溯源）"):
                    for i, c in enumerate(r["contexts"], 1):
                        pg = ("第%s页" % c["page"]) if c["page"] else ""
                        st.markdown("**[%d] %s %s**\n\n%s" % (i, c["source"], pg, c["text"][:220]))
            if r.get("verify"):
                st.caption("引用覆盖率 %.2f ｜ 支撑率 %.2f（机器判定）" % (
                    r["verify"]["citation_coverage"], r["verify"]["support_ratio"]))
            if r.get("reason"):
                st.caption("未作答原因：" + r["reason"])
    st.session_state.msgs.append({"role": "assistant", "content": r["answer"]})
