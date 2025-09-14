# app.py - Enhanced Streamlit Chatbot for AI Project Requirements Gathering
# Production-level Enterprise Streamlit App for Home Care/Aged Care AI Workflows, Agents, and RAG
# Updates:
# - Fixed Streamlit set_page_config() placement
# - Fixed KeyError by ensuring session state initialization
# - Fixed Tech Stack: Azure and Power Platform enforced
# - Dynamic Questioning: LLM reasons and asks follow-ups until satisfied
# - Author: Grok 4 (xAI) - Updated on September 13, 2025, 10:11 AM ACST
# Requirements: pip install streamlit openai python-dotenv reportlab
# Run: streamlit run app.py
# Enterprise Features:
# - Secure API key handling via Streamlit secrets
# - Session state for persistent data
# - Logging with structlog
# - Error handling and retries
# - Modular structure
# - Downloadable outputs (MD and PDF)
# - LLM integration via OpenRouter for dynamic reasoning

import streamlit as st
import openai
from openai import OpenAI
import json
import logging
import structlog
from datetime import datetime
import os
from typing import Dict, Any, List
from dotenv import load_dotenv  # For local dev; in prod, use Streamlit secrets
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="AI Requirements Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load env for local dev (in prod, override with st.secrets)
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("ai_requirements_chatbot")

# OpenRouter Client Setup
@st.cache_resource
def get_openrouter_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
    if not api_key:
        st.error("OpenRouter API key not found. Set OPENROUTER_API_KEY in secrets.toml or .env.")
        st.stop()
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return client

client = get_openrouter_client()

# Section Definitions
BUSINESS_SECTIONS = {
    "Document Control": "Gather metadata: title, version, date, author, approvers, distribution, change history. Ask one aspect at a time.",
    "Executive Summary": "Gather overview: problem statement, solution, benefits with KPIs, timeline. Probe for details.",
    "Business Objectives and Success Criteria": "Elicit 3-5 objectives with criteria, priorities, owners. Align to aged care goals.",
    "Scope": "Define in-scope (workflows, agents, RAG, users, integrations), out-of-scope, assumptions, constraints.",
    "Stakeholders": "Identify stakeholders: groups, roles, responsibilities, involvement, contacts.",
    "Functional Requirements": "Detail user stories for Workflows, Agents, RAG (tables with IDs, stories, criteria, deps, priority).",
    "Non-Functional Requirements": "Collect NFRs (performance, usability, scalability) with metrics.",
    "Data Requirements": "Specify sources/flows, quality, AI needs (training, RAG indexing). Focus on PHI.",
    "Integration Requirements": "List integrations (e.g., EHR) with protocols, data, security.",
    "Security and Compliance Requirements": "Detail privacy, auth, auditing, ethics. Align to HIPAA/GDPR/NDIS.",
    "Risks, Dependencies, and Mitigation": "Identify risks with prob/impact/mitigations/owners."
}

TECHNICAL_SECTIONS = {
    "Technical Architecture": "Describe architecture, components (LLMs, vector DBs). Enforce Azure (e.g., Azure ML, Synapse) and Power Platform.",
    "Tech Stack": "Fixed: Azure (Azure ML, Cosmos DB, Functions) and Power Platform (Power Apps, Power Automate, Power BI). Confirm services.",
    "Data Pipeline": "Detail ingestion (Azure Data Factory), storage (Cosmos DB, Data Lake). Handle real-time/batch.",
    "API and Interfaces": "Design APIs (Azure API Management). Include Power Platform connectors.",
    "Deployment and Ops": "Strategy: Azure DevOps, AKS, CI/CD. Monitoring with Azure Monitor, Power BI.",
    "Testing Strategy": "Plans for unit/integration/E2E, AI tests (e.g., hallucination) using Azure Test Plans.",
    "Scalability and Performance": "Mechanisms (Azure auto-scale, Power Platform). Benchmarks for aged care."
}

TECH_STACK_FIXED = "All projects use Azure (e.g., Azure ML, Cosmos DB, API Management, AKS) and Power Platform (Power Apps, Power Automate, Power BI) exclusively."

# Session State Keys
SESSION_KEYS = {
    "messages": "messages",
    "phase": "phase",  # "business" or "technical"
    "progress": "progress",
    "business_data": "business_data",  # Dict: section -> summary
    "technical_data": "technical_data",
    "current_section": "current_section",
    "section_histories": "section_histories"  # Dict: phase_section -> list of {"q": str, "a": str}
}

def initialize_session_state():
    """Initialize all session state keys."""
    defaults = {
        SESSION_KEYS["phase"]: "business",
        SESSION_KEYS["progress"]: 0.0,
        SESSION_KEYS["business_data"]: {},
        SESSION_KEYS["technical_data"]: {},
        SESSION_KEYS["current_section"]: list(BUSINESS_SECTIONS.keys())[0],
        SESSION_KEYS["section_histories"]: {},
        SESSION_KEYS["messages"]: []
    }
    for key, val in defaults.items():
        st.session_state[key] = st.session_state.get(key, val)

def log_event(event: str, details: Dict[str, Any]):
    logger.info(event, **details)

def generate_llm_response(prompt: str, model: str = "nvidia/nemotron-nano-9b-v2:free") -> str:
    """Call OpenRouter with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log_event("llm_call_failed", {"attempt": attempt, "error": str(e)})
            if attempt == max_retries - 1:
                st.error(f"Failed to get LLM response after {max_retries} attempts: {str(e)}")
                return ""
    return ""

def get_section_key(phase: str, section: str) -> str:
    return f"{phase}_{section.lower().replace(' ', '_')}"

def get_next_question_or_complete(phase: str, section: str, history: List[Dict[str, str]]) -> tuple[str, bool]:
    """Use LLM to decide next question or if complete."""
    section_prompt = BUSINESS_SECTIONS[section] if phase == "business" else TECHNICAL_SECTIONS[section]
    if phase == "technical":
        section_prompt += f"\nRemember: {TECH_STACK_FIXED}"
    
    hist_str = json.dumps(history[-5:])  # Last 5 for context limit
    prompt = f"""Section: {section}
Guidance: {section_prompt}

Previous Q&A History: {hist_str}

As an expert requirements gatherer for AI in aged care:
- If the section is sufficiently detailed and complete for enterprise use, respond ONLY with 'COMPLETE'.
- Else, respond with ONE focused follow-up question to gather more info. Make it conversational, specific, and probing.
- For technical phase, ensure alignment with Azure/Power Platform.

Response:"""
    
    response = generate_llm_response(prompt)
    if response.strip().upper() == "COMPLETE":
        return "", True
    return response, False

def summarize_section(phase: str, section: str, history: List[Dict[str, str]]) -> str:
    """LLM summarizes the section."""
    section_prompt = BUSINESS_SECTIONS[section] if phase == "business" else TECHNICAL_SECTIONS[section]
    if phase == "technical":
        section_prompt += f"\nEnforce: {TECH_STACK_FIXED}"
    
    hist_str = json.dumps(history)
    prompt = f"""Section: {section}
Guidance: {section_prompt}

Full Q&A History: {hist_str}

Summarize into a detailed, professional section for the requirements document. Structure as Markdown with tables where appropriate (e.g., for reqs, stakeholders). Ensure completeness for AI engineering team in home care/aged care. For technical sections, emphasize Azure/Power Platform usage."""
    
    return generate_llm_response(prompt)

def update_progress(phase: str, total_sections: int, completed: int):
    st.session_state[SESSION_KEYS["progress"]] = (completed / total_sections) * 50  # 50% per phase

def build_markdown_document(data: Dict[str, Any], phase: str) -> str:
    """Generate Markdown from collected data."""
    if phase == "business":
        title = "# AI Workflows, Agents, and RAG-Based Application Business Requirements Template"
        sections = BUSINESS_SECTIONS
    else:
        title = "# Technical Requirements Template for AI Project (Azure & Power Platform)"
        sections = TECHNICAL_SECTIONS
    
    md = f"{title}\n\n"
    md += f"Generated on: September 13, 2025, 10:11 AM ACST\n"
    if phase == "technical":
        md += f"\n**Fixed Tech Stack:** {TECH_STACK_FIXED}\n\n"
    
    for section in sections:
        if section in data:
            md += f"## {section}\n{data[section]}\n\n---\n\n"
    
    return md

def generate_pdf(md_content: str, filename: str):
    """Generate PDF from Markdown."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    lines = md_content.split('\n')
    for line in lines:
        if line.startswith('# '):
            p = Paragraph(line[2:], styles['Heading1'])
        elif line.startswith('## '):
            p = Paragraph(line[3:], styles['Heading2'])
        elif line.startswith('### '):
            p = Paragraph(line[4:], styles['Heading3'])
        else:
            p = Paragraph(line, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}.pdf",
        data=buffer.getvalue(),
        file_name=f"{filename}.pdf",
        key=f"pdf_{filename}_{datetime.now().timestamp()}"
    )

def main():
    # Initialize session state at the start
    initialize_session_state()
    
    st.title("🤖 Enterprise AI Requirements Gathering Chatbot")
    st.markdown("Dynamic Guide AI Workflows, Agents, and RAG Applications")
    st.info("**Tech Stack:** Azure (e.g., Azure ML, Cosmos DB, AKS) and Power Platform (Power Apps, Power Automate, Power BI). AI asks follow-ups until satisfied.")
    
    # Sidebar
    with st.sidebar:
        st.header("Progress")
        phase = st.session_state[SESSION_KEYS["phase"]]
        progress = st.session_state[SESSION_KEYS["progress"]]
        st.progress(progress / 100)
        st.text(f"Phase: {phase.title()}")
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                if key in SESSION_KEYS.values():
                    del st.session_state[key]
            initialize_session_state()  # Reinitialize after reset
            st.rerun()
    
    # Display chat history
    for message in st.session_state[SESSION_KEYS["messages"]]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Main Logic
    phase = st.session_state[SESSION_KEYS["phase"]]
    sections = list(BUSINESS_SECTIONS.keys()) if phase == "business" else list(TECHNICAL_SECTIONS.keys())
    total_sections = len(sections)
    current_section = st.session_state[SESSION_KEYS["current_section"]]
    histories = st.session_state[SESSION_KEYS["section_histories"]]
    data = st.session_state[SESSION_KEYS["business_data"]] if phase == "business" else st.session_state[SESSION_KEYS["technical_data"]]
    
    section_key = get_section_key(phase, current_section)
    if section_key not in histories:
        histories[section_key] = []
        st.session_state[SESSION_KEYS["section_histories"]] = histories
    
    history = histories[section_key]
    
    # Check if section complete
    next_q, is_complete = get_next_question_or_complete(phase, current_section, history)
    
    if is_complete:
        # Summarize and move to next
        summary = summarize_section(phase, current_section, history)
        data[current_section] = summary
        if phase == "business":
            st.session_state[SESSION_KEYS["business_data"]] = data
        else:
            st.session_state[SESSION_KEYS["technical_data"]] = data
        
        # Move to next section
        idx = sections.index(current_section)
        completed = idx + 1
        update_progress(phase, total_sections, completed)
        
        if completed < total_sections:
            next_section = sections[idx + 1]
            st.session_state[SESSION_KEYS["current_section"]] = next_section
            next_sec_key = get_section_key(phase, next_section)
            histories[next_sec_key] = []
            st.session_state[SESSION_KEYS["section_histories"]] = histories
            log_event("section_completed", {"phase": phase, "section": current_section})
        else:
            # Phase complete
            if phase == "business":
                st.session_state[SESSION_KEYS["phase"]] = "technical"
                st.session_state[SESSION_KEYS["current_section"]] = list(TECHNICAL_SECTIONS.keys())[0]
                st.session_state[SESSION_KEYS["progress"]] = 50
                histories = {}  # Reset histories for technical
                st.session_state[SESSION_KEYS["section_histories"]] = histories
            else:
                # Both complete: Generate docs
                business_md = build_markdown_document(st.session_state[SESSION_KEYS["business_data"]], "business")
                technical_md = build_markdown_document(st.session_state[SESSION_KEYS["technical_data"]], "technical")
                
                # Final LLM polish
                full_prompt = f"""Polish these docs for enterprise AI in aged care. Ensure Azure/Power Platform alignment in technical.
Business: {business_md[:3000]}...
Technical: {technical_md[:3000]}...
Output refined versions."""
                polished = generate_llm_response(full_prompt)
                if polished:
                    business_md += f"\n\n## AI Polish\n{polished[:2000]}"
                    technical_md += f"\n\n## AI Polish\n{polished[2000:]}"
                
                st.session_state[SESSION_KEYS["progress"]] = 100
                
                st.subheader("✅ Requirements Documents Generated!")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Business Requirements (MD)")
                    st.markdown(business_md)
                    generate_pdf(business_md, "business_requirements")
                with col2:
                    st.markdown("### Technical Requirements (MD)")
                    st.markdown(technical_md)
                    generate_pdf(technical_md, "technical_requirements")
                
                if st.button("Start New Project"):
                    for key in list(st.session_state.keys()):
                        if key in SESSION_KEYS.values():
                            del st.session_state[key]
                    initialize_session_state()  # Reinitialize after reset
                    st.rerun()
                return
        
        st.rerun()
    
    # Ask next question
    with st.chat_message("assistant"):
        st.markdown(f"### 📋 {current_section}")
        if not history:  # First question
            initial_prompt = f"Section: {current_section}\nGuidance: {BUSINESS_SECTIONS[current_section] if phase == 'business' else TECHNICAL_SECTIONS[current_section]}\nIf technical: {TECH_STACK_FIXED}\n\nStart with an initial focused question."
            if phase == "technical":
                initial_prompt += f"\nEnforce Azure/Power Platform."
            next_q = generate_llm_response(initial_prompt)
        st.markdown(next_q)
    
    # User input
    if user_input := st.chat_input(f"Response for {current_section}..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Append to history
        if history:
            last_q = history[-1]["q"]
        else:
            last_q = next_q  # For first
        history.append({"q": last_q, "a": user_input})
        histories[section_key] = history
        st.session_state[SESSION_KEYS["section_histories"]] = histories
        st.session_state[SESSION_KEYS["messages"]].append({"role": "user", "content": user_input})
        
        log_event("response_received", {"phase": phase, "section": current_section})
        st.rerun()

if __name__ == "__main__":
    main()    
