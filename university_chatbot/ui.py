"""
Enhanced Streamlit Frontend for the University Chatbot.
Beautiful visualization of the RAG + Judge evaluation pipeline.

Run with:
    streamlit run ui.py
"""

import os

import httpx
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")

st.set_page_config(
    page_title="University Assistant - LLM Judge",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .hero-title {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin-top: 0.7rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
            animation: fadeIn 0.45s ease-in;
        }

        .evaluation-shell {
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
            border: 1px solid rgba(251, 146, 60, 0.2);
            box-shadow: 0 8px 18px rgba(251, 146, 60, 0.12);
        }

        .badge {
            display: inline-block;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .badge-pass {
            background: rgba(16, 185, 129, 0.12);
            color: #047857;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .badge-fail {
            background: rgba(239, 68, 68, 0.12);
            color: #b91c1c;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .score-bar {
            width: 100%;
            height: 10px;
            background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.6rem 0 1rem 0;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #1d4ed8, #7c3aed);
            border-radius: 999px;
            transition: width 0.5s ease;
        }

        .stage-card {
            background: rgba(37, 99, 235, 0.04);
            border-radius: 12px;
            border-left: 4px solid #2563eb;
            padding: 0.8rem 0.9rem;
            margin: 0.3rem 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            animation: fadeIn 0.35s ease-in;
        }

        .stage-card.done {
            border-left-color: #10b981;
            background: rgba(16, 185, 129, 0.06);
        }

        .stage-icon {
            font-size: 1.2rem;
            animation: pulse 1.8s infinite;
        }

        .stage-card.done .stage-icon {
            animation: none;
        }

        .metric-card {
            background: linear-gradient(135deg, #dbeafe 0%, #e9d5ff 100%);
            border-radius: 14px;
            padding: 1rem 0.8rem;
            text-align: center;
            height: 100%;
            box-shadow: 0 8px 18px rgba(124, 58, 237, 0.08);
        }

        .hallucination-box {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 5px solid #dc2626;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
            color: #7f1d1d;
        }

        .safe-box {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border-left: 5px solid #16a34a;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
            color: #14532d;
        }

        .reason-box {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-left: 5px solid #0284c7;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
            color: #0f172a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_ok" not in st.session_state:
    st.session_state.api_ok = None

# sidebar
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    try:
        health = httpx.get(HEALTH_URL, timeout=3)
        if health.status_code == 200:
            st.success("✅ Backend online")
            st.session_state.api_ok = True
        else:
            st.error("❌ Backend unavailable")
            st.session_state.api_ok = False
    except Exception:
        st.error("❌ Backend unavailable")
        st.session_state.api_ok = False

    st.markdown("---")
    st.markdown("### 📌 What this app does")
    st.markdown(
        "- Retrieves context from the university knowledge base\n"
        "- Generates an answer\n"
        "- Runs LLM-as-a-Judge quality evaluation\n"
        "- Flags hallucinations and low-quality responses"
    )

    st.markdown("---")
    st.markdown("### 📊 Session stats")
    total_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
    total_passed = len([m for m in st.session_state.messages if m.get("passed") is True])
    pass_rate = (total_passed / total_questions * 100) if total_questions else 0
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", total_questions)
    with col2:
        st.metric("Pass rate", f"{pass_rate:.0f}%")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# header
st.markdown('<h1 class="hero-title">🎓 University Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RAG-powered assistant with LLM-as-a-Judge quality control</div>', unsafe_allow_html=True)
st.markdown("---")

# render old conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            col1, col2 = st.columns([0.08, 0.92])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**You:** {message['content']}")

    if message["role"] == "assistant":
        evaluation = message.get("evaluation") or {}
        score = message.get("score")
        passed = message.get("passed", False)
        attempts = message.get("attempts", 1)

        with st.container():
            col1, col2 = st.columns([0.08, 0.92])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown("**Assistant:**")
                st.markdown(f'<div class="answer-box">{message["content"]}</div>', unsafe_allow_html=True)

                if evaluation or score is not None:
                    with st.expander(f"📊 Judge evaluation: {score}/10" if score is not None else "📊 Judge evaluation"):
                        badge_class = "badge-pass" if passed else "badge-fail"
                        badge_text = "✅ PASSED" if passed else "❌ FAILED"
                        st.markdown(f'<div class="badge {badge_class}">{badge_text} • {attempts} attempt(s)</div>', unsafe_allow_html=True)

                        if score is not None:
                            score_pct = min(score * 10, 100)
                            st.markdown(f"**Overall score: {score}/10**")
                            st.markdown(f'<div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>', unsafe_allow_html=True)

                        if evaluation:
                            cols = st.columns(5)
                            metric_names = ["Faithfulness", "Correctness", "Relevance", "Completeness", "Clarity"]
                            for idx, key in enumerate(metric_names):
                                with cols[idx]:
                                    value = evaluation.get(key.lower(), 0)
                                    st.markdown(f'<div class="metric-card"><div>{key}</div><h3>{value}/10</h3></div>', unsafe_allow_html=True)

                            if evaluation.get("hallucination"):
                                claims = evaluation.get("hallucinated_claims", [])
                                st.markdown('<div class="hallucination-box"><strong>🚨 Hallucination detected</strong><br>The answer contains unsupported claims.</div>', unsafe_allow_html=True)
                                if claims:
                                    st.markdown("**Unsupported claims:**")
                                    for claim in claims:
                                        st.markdown(f"- {claim}")
                            else:
                                st.markdown('<div class="safe-box"><strong>✅ No hallucination detected</strong><br>All claims are grounded in the retrieved context.</div>', unsafe_allow_html=True)

                            reason = evaluation.get("reason")
                            if reason:
                                st.markdown(f'<div class="reason-box"><strong>📝 Judge reasoning:</strong><br>{reason}</div>', unsafe_allow_html=True)

        st.markdown("---")

# input
st.markdown("### ❓ Ask the university assistant")

if prompt := st.chat_input("Ask about admissions, fees, academic policies, or student support..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.container():
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            st.markdown("👤")
        with col2:
            st.markdown(f"**You:** {prompt}")

    st.markdown("---")
    st.markdown("### 🔄 Processing pipeline")

    stage_cols = st.columns(4)
    stages = [
        (stage_cols[0], "🔍", "Retrieving context"),
        (stage_cols[1], "🤖", "Generating answer"),
        (stage_cols[2], "⚖️", "Judging quality"),
        (stage_cols[3], "✅", "Finalizing response"),
    ]

    for container, icon, label in stages:
        with container:
            st.markdown(f'<div class="stage-card"><span class="stage-icon">{icon}</span><span>{label}</span></div>', unsafe_allow_html=True)

    try:
        if not st.session_state.api_ok:
            st.error("⚠️ Your backend is not running. Start it with `python app.py` first.")
        else:
            with st.spinner("Running the retrieval → generation → evaluation pipeline..."):
                response = httpx.post(API_URL, json={"question": prompt}, timeout=120.0)
                response.raise_for_status()
                data = response.json()

            answer = data.get("answer", "No answer returned.")
            score = data.get("score")
            passed = data.get("passed_quality_check", False)
            attempts = data.get("attempts", 1)
            evaluation = data.get("evaluation") or {}

            for idx, (container, icon, label) in enumerate(stages):
                with container:
                    status = "done" if idx <= 3 else ""
                    st.markdown(f'<div class="stage-card {status}"><span class="stage-icon">{icon}</span><span>{label}</span></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🤖 Final answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("### ⚖️ Judge evaluation report")
            badge_class = "badge-pass" if passed else "badge-fail"
            badge_text = "✅ PASSED QUALITY CHECK" if passed else "❌ FAILED QUALITY CHECK"
            st.markdown(f'<div class="badge {badge_class}">{badge_text} • {attempts} attempt(s)</div>', unsafe_allow_html=True)

            if score is not None:
                score_pct = min(score * 10, 100)
                st.markdown(f"**Overall score: {score}/10**")
                st.markdown(f'<div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>', unsafe_allow_html=True)

            if evaluation:
                cols = st.columns(5)
                metric_names = ["Faithfulness", "Correctness", "Relevance", "Completeness", "Clarity"]
                for idx, key in enumerate(metric_names):
                    with cols[idx]:
                        value = evaluation.get(key.lower(), 0)
                        st.markdown(f'<div class="metric-card"><div>{key}</div><h3>{value}/10</h3></div>', unsafe_allow_html=True)

                if evaluation.get("hallucination"):
                    claims = evaluation.get("hallucinated_claims", [])
                    st.markdown('<div class="hallucination-box"><strong>🚨 Hallucination detected</strong><br>The answer contains claims not grounded in the retrieved knowledge.</div>', unsafe_allow_html=True)
                    if claims:
                        st.markdown("**Unsupported claims:**")
                        for claim in claims:
                            st.markdown(f"- {claim}")
                else:
                    st.markdown('<div class="safe-box"><strong>✅ No hallucination detected</strong><br>The answer is grounded in the retrieved context.</div>', unsafe_allow_html=True)

                reason = evaluation.get("reason")
                if reason:
                    st.markdown(f'<div class="reason-box"><strong>📝 Judge reasoning:</strong><br>{reason}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "evaluation": evaluation,
                "score": score,
                "passed": passed,
                "attempts": attempts,
            })

    except httpx.ConnectError:
        st.error("⚠️ Could not connect to the backend API. Ensure `python app.py` is running.")
    except httpx.TimeoutException:
        st.error("⚠️ The request timed out. Please try again.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
