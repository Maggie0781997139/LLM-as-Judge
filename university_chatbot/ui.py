"""
Streamlit Frontend for the University Chatbot.
Provides a chat interface and displays the Judge's evaluation metrics.

Run with:
    streamlit run ui.py
"""

import streamlit as st
import httpx
import json
import os

# Configuration
API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 University Chatbot")
st.markdown("Ask me anything about admissions, fees, academic policies, or student services!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message is from the assistant and has evaluation data, show it in an expander
        if message["role"] == "assistant" and "evaluation" in message:
            eval_data = message["evaluation"]
            score = message.get("score", 0)
            passed = message.get("passed", False)
            attempts = message.get("attempts", 1)
            
            status_color = "green" if passed else "red"
            status_text = "PASSED" if passed else "FAILED"
            
            with st.expander(f"⚖️ Judge Evaluation: {score}/10 ({status_text})"):
                st.markdown(f"**Quality Check:** :{status_color}[{status_text}] (Took {attempts} attempt/s)")
                
                if eval_data:
                    # Metrics columns
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Faithfulness", f"{eval_data.get('faithfulness', 0)}/10")
                    c2.metric("Correctness", f"{eval_data.get('correctness', 0)}/10")
                    c3.metric("Relevance", f"{eval_data.get('relevance', 0)}/10")
                    c4.metric("Completeness", f"{eval_data.get('completeness', 0)}/10")
                    c5.metric("Clarity", f"{eval_data.get('clarity', 0)}/10")
                    
                    # Hallucination warning
                    if eval_data.get("hallucination"):
                        st.error("🚨 Hallucination Detected!")
                        claims = eval_data.get("hallucinated_claims", [])
                        if claims:
                            st.write("**Unsupported claims:**")
                            for claim in claims:
                                st.write(f"- {claim}")
                    
                    st.info(f"**Judge's Reason:** {eval_data.get('reason', 'N/A')}")

# Accept user input
if prompt := st.chat_input("E.g., What are the admission requirements for medicine?"):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Call the FastAPI backend and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching knowledge base and evaluating answer..."):
            try:
                response = httpx.post(
                    API_URL, 
                    json={"question": prompt},
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "Error: No answer returned.")
                score = data.get("score")
                evaluation = data.get("evaluation")
                passed = data.get("passed_quality_check", False)
                attempts = data.get("attempts", 1)
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Show evaluation metadata below the answer
                status_color = "green" if passed else "red"
                status_text = "PASSED" if passed else "FAILED"
                
                with st.expander(f"⚖️ Judge Evaluation: {score}/10 ({status_text})"):
                    st.markdown(f"**Quality Check:** :{status_color}[{status_text}] (Took {attempts} attempt/s)")
                    
                    if evaluation:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Faithfulness", f"{evaluation.get('faithfulness', 0)}/10")
                        c2.metric("Correctness", f"{evaluation.get('correctness', 0)}/10")
                        c3.metric("Relevance", f"{evaluation.get('relevance', 0)}/10")
                        c4.metric("Completeness", f"{evaluation.get('completeness', 0)}/10")
                        c5.metric("Clarity", f"{evaluation.get('clarity', 0)}/10")
                        
                        if evaluation.get("hallucination"):
                            st.error("🚨 Hallucination Detected!")
                            claims = evaluation.get("hallucinated_claims", [])
                            if claims:
                                st.write("**Unsupported claims:**")
                                for claim in claims:
                                    st.write(f"- {claim}")
                        
                        st.info(f"**Judge's Reason:** {evaluation.get('reason', 'N/A')}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "evaluation": evaluation,
                    "score": score,
                    "passed": passed,
                    "attempts": attempts
                })
                
            except httpx.ConnectError:
                st.error("⚠️ Could not connect to the backend API. Is `python app.py` running?")
            except httpx.TimeoutException:
                st.error("⚠️ The request timed out. The LLMs took too long to respond.")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")
