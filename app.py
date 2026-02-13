import csv
import io
import json
import os
import re
from datetime import date
from typing import Dict, List, Tuple

import requests
import streamlit as st


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL_BASE = "anthropic/claude-opus-4.1"
DEFAULT_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "Strategic Forecast Question Builder",
}


# -------------------------
# Small utilities
# -------------------------
def today_iso() -> str:
    return date.today().isoformat()


def model_name(base: str, online: bool) -> str:
    base = (base or "").strip().replace(":online", "")
    return f"{base}:online" if online else base


def strip_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


def normalize(text: str) -> str:
    text = strip_urls(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def parse_questions(raw: str) -> List[Dict[str, str]]:
    blocks = re.split(r"\n(?=Q\d+\b)", raw.strip(), flags=re.MULTILINE)
    out: List[Dict[str, str]] = []
    for b in blocks:
        if not b.strip().startswith("Q"):
            continue
        item = {
            "id": "",
            "axis": "",
            "title": "",
            "horizon": "",
            "question": "",
            "why_it_matters": "",
            "decision_link": "",
            "signals": "",
        }
        for ln in b.splitlines():
            if re.match(r"^Q\d+", ln):
                item["id"] = ln.strip().split()[0]
            else:
                m = re.match(
                    r"^(Axis|Title|Horizon|Question|Why it matters|Decision link|Signal hints):\s*(.*)$",
                    ln.strip(),
                )
                if m:
                    key, val = m.group(1), m.group(2)
                    map_key = {
                        "Axis": "axis",
                        "Title": "title",
                        "Horizon": "horizon",
                        "Question": "question",
                        "Why it matters": "why_it_matters",
                        "Decision link": "decision_link",
                        "Signal hints": "signals",
                    }[key]
                    item[map_key] = val.strip()
        if item["id"] and item["question"]:
            out.append(item)
    return out


def questions_to_csv(raw: str) -> str:
    rows = parse_questions(raw)
    headers = ["id", "axis", "title", "horizon", "question", "why_it_matters", "decision_link", "signals"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


# -------------------------
# OpenRouter
# -------------------------
def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int = 180,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **DEFAULT_HEADERS,
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter error {res.status_code}: {res.text[:800]}")
    return res.json()["choices"][0]["message"]["content"]


# -------------------------
# Prompt pack
# -------------------------
def p_research(company_brief: str) -> str:
    return f"""You are a strategic intelligence analyst.

Rules:
- Use current information if web access is available.
- Do not output URLs.
- Keep language concise and business-oriented.
- Focus on medium-term (6-24 months) and long-term (24-60 months).

Company brief:
{company_brief}

Output EXACTLY:

STRATEGIC AXES:
1) <Axis name> — <why it matters for this company>
2) ...
(5 to 9 axes total)

KEY UNCERTAINTIES BY AXIS:
- <Axis name>: <3 uncertainties separated by semicolons>
- ...

FORECASTER NOTES:
- 6 short bullets describing what expert forecasters can uniquely add.
"""


def p_generate(company_brief: str, research_notes: str, n_questions: int) -> str:
    return f"""You are designing forecasting questions to be asked to professional forecasters (Metaculus-style).

Objective:
- Create high-value questions that reduce decision uncertainty for the company.
- Questions MUST be relevant to this specific company and sector.
- Medium-term focus first (6-24m), then include some longer-term directional questions (24-60m).
- Resolution criteria can be light; clarity and decision value are top priority.

Hard rules:
- No URLs.
- No vague wording like "significantly" without threshold.
- Keep each question answerable by an external forecaster using public + domain signals.
- Keep outputs clean and compact.

Company brief:
{company_brief}

Research notes:
{research_notes}

Produce exactly {n_questions} questions.

Output format:

AXES SUMMARY:
- <Axis> | <business KPI/decision impacted> | <main horizon>
...

QUESTIONS:
Q1
Axis: <one axis from summary>
Title: <short>
Horizon: <12m|24m|36m|60m>
Question: <clear forecasting question>
Why it matters: <1 sentence for business impact>
Decision link: <decision influenced by this answer>
Signal hints: <2-4 signal types forecasters should track>

Q2
...
"""


def p_refresh(raw_questions: str, company_brief: str) -> str:
    return f"""You are a quality editor for strategic forecasting questions.

Tasks:
- Tighten wording.
- Remove duplicates.
- Make horizons explicit and balanced across 12m/24m/36m+.
- Ensure each question is tied to a decision.
- Use up-to-date framing if relevant.

No URLs. Keep exact output structure with AXES SUMMARY then QUESTIONS.

Company brief:
{company_brief}

Draft:
{raw_questions}
"""


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Strategic Forecast Question Builder", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --bg: #f7fbff;
        --card: #ffffff;
        --ink: #12324a;
        --muted: #4d6a7f;
        --blue: #1f8ef1;
        --green: #20b486;
      }
      .stApp { background: #ffffff; color: #000000 !important; }
      .block-container { padding-top: 2rem; }
      h1, h2, h3, p, label, span, div { color: #000000; }
      div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4edf5; }
      div[data-testid="stSidebar"] * { color: #000000 !important; }
      .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cfd8e3 !important;
      }
      .stSelectbox div[data-baseweb="select"] > div,
      .stMultiSelect div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #000000 !important;
      }
      .stSlider [data-baseweb="slider"] { color: #1f8ef1 !important; }
      .stButton button {
        background: #f2f8ff !important;
        color: #000000 !important;
        border: 1px solid #c8d8ea !important;
      }
      .tag {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.35rem;
      }
      .tag-blue { background: #e6f3ff; color: #1668aa; }
      .tag-green { background: #e8f9f2; color: #16795d; }
      .card {
        background: var(--card);
        border: 1px solid #e4edf5;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Strategic Forecast Question Builder")
st.caption("Input: short company/sector brief. Output: strategic axes + forecast questions for pro forecasters.")

with st.sidebar:
    st.header("OpenRouter")
    api_key = st.text_input("OPENROUTER_API_KEY", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")

    st.header("Model")
    base_model = st.text_input("Base model", value=DEFAULT_MODEL_BASE)
    use_online_research = st.checkbox("Research uses :online", value=True)
    use_online_generate = st.checkbox("Generate uses :online", value=True)
    use_online_refresh = st.checkbox("Refresh/editor pass uses :online", value=True)

    st.header("Output")
    n_questions = st.slider("Number of questions", 8, 60, 24, 2)
    run_refresh = st.checkbox("Run final quality pass", value=True)

    st.header("Temperatures")
    temp_research = st.slider("Research", 0.0, 1.0, 0.2, 0.05)
    temp_generate = st.slider("Generate", 0.0, 1.0, 0.35, 0.05)
    temp_refresh = st.slider("Refresh", 0.0, 1.0, 0.15, 0.05)

    st.header("Max tokens")
    tok_research = st.number_input("Research", 500, 5000, 1400, 100)
    tok_generate = st.number_input("Generate", 1000, 12000, 4200, 200)
    tok_refresh = st.number_input("Refresh", 800, 12000, 3200, 200)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Company brief")
    brief = st.text_area(
        "Describe the company and sector in a few lines",
        height=250,
        placeholder="Business model, key geographies, major cost/revenue drivers, and 2-3 strategic decisions in the next 12-24 months.",
    )
    constraints = st.text_area(
        "Optional constraints",
        height=100,
        placeholder="e.g., emphasize EU regulation and procurement risk, keep at least 30% questions at 36m+ horizon.",
    )
    run = st.button("Generate strategic axes + questions", type="primary", use_container_width=True, disabled=not api_key)

with right:
    st.subheader("2) Output")
    info_box = st.empty()
    research_box = st.empty()
    draft_box = st.empty()
    final_box = st.empty()

if run:
    if not brief.strip():
        st.error("Please add a company brief.")
        st.stop()

    full_brief = brief.strip()
    if constraints.strip():
        full_brief += f"\n\nExtra constraints:\n{constraints.strip()}"

    info_box.markdown(
        "<div class='card'><span class='tag tag-blue'>Medium term first</span><span class='tag tag-green'>Metaculus-ready framing</span></div>",
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Researching strategic axes with up-to-date context…"):
            research = openrouter_chat(
                api_key=api_key,
                model=model_name(base_model, use_online_research),
                messages=[{"role": "user", "content": p_research(full_brief)}],
                temperature=float(temp_research),
                max_tokens=int(tok_research),
                timeout=240,
            )
        research = normalize(research)
        research_box.text_area("Research notes", research, height=250)

        with st.spinner("Generating forecast questions…"):
            draft = openrouter_chat(
                api_key=api_key,
                model=model_name(base_model, use_online_generate),
                messages=[{"role": "user", "content": p_generate(full_brief, research, int(n_questions))}],
                temperature=float(temp_generate),
                max_tokens=int(tok_generate),
                timeout=300,
            )
        draft = normalize(draft)
        draft_box.text_area("Draft output", draft, height=320)

        final_text = draft
        if run_refresh:
            with st.spinner("Running quality/editor pass…"):
                final_text = openrouter_chat(
                    api_key=api_key,
                    model=model_name(base_model, use_online_refresh),
                    messages=[{"role": "user", "content": p_refresh(draft, full_brief)}],
                    temperature=float(temp_refresh),
                    max_tokens=int(tok_refresh),
                    timeout=240,
                )
            final_text = normalize(final_text)

        final_box.text_area("Final strategic questions", final_text, height=520)

        st.markdown("### Downloads")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download .txt",
                data=final_text.encode("utf-8"),
                file_name=f"strategic_questions_{today_iso()}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download .csv",
                data=questions_to_csv(final_text).encode("utf-8"),
                file_name=f"strategic_questions_{today_iso()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Run failed: {exc}")
