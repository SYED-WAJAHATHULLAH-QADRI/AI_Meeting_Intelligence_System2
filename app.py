# ============================================================
# AI MEETING INTELLIGENCE SYSTEM (AIMIS)
# MSc Computing Project Prototype
# ============================================================

import os
import time
import shutil
import tempfile
from typing import List, Optional

import pandas as pd
import streamlit as st
import whisper

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIMIS | AI Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #a9b4c2;
        margin-bottom: 25px;
    }

    .hero-box {
        padding: 25px;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 70, 140, 0.35),
                rgba(25, 25, 35, 0.85)
            );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #161b22;
        border: 1px solid #30363d;
        text-align: center;
    }

    .section-title {
        margin-top: 20px;
        margin-bottom: 15px;
        font-size: 27px;
        font-weight: 700;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(35, 134, 54, 0.15);
        border: 1px solid rgba(35, 134, 54, 0.5);
    }

    .footer {
        text-align: center;
        color: #8b949e;
        padding-top: 25px;
        font-size: 13px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CONSTANTS
# ============================================================

WHISPER_MODEL_NAME = "small.en"

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# 4. STRUCTURED OUTPUT SCHEMA
# ============================================================

class Decision(BaseModel):

    decision: str = Field(
        description="Confirmed decision made during the meeting."
    )

    evidence: str = Field(
        description="Transcript evidence supporting the decision."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class ActionItem(BaseModel):

    action: str = Field(
        description="Task or action identified in the meeting."
    )

    owner: Optional[str] = Field(
        default=None,
        description="Responsible person if explicitly identified."
    )

    deadline: Optional[str] = Field(
        default=None,
        description="Deadline if explicitly stated."
    )

    status: str = Field(
        description="assigned, proposed, or unclear"
    )

    evidence: str = Field(
        description="Transcript evidence supporting the action."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class MeetingIntelligence(BaseModel):

    meeting_title: str

    summary: str

    key_topics: List[str]

    decisions: List[Decision]

    action_items: List[ActionItem]

    unresolved_issues: List[str]

    ambiguities: List[str]


# ============================================================
# 5. LOAD WHISPER
# ============================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model(
        WHISPER_MODEL_NAME
    )


# ============================================================
# 6. GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:

        return None

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


# ============================================================
# 7. GEMINI PROMPT
# ============================================================

def create_meeting_prompt(
    transcript
):

    return f"""
You are an AI Meeting Intelligence System.

Analyse the complete meeting transcript and extract accurate,
structured meeting intelligence.

Follow these rules carefully.

1. MEETING SUMMARY
Provide a concise factual summary of the meeting.

2. KEY TOPICS
Identify the principal subjects discussed.

3. DECISIONS
Record only confirmed decisions.

A confirmed decision may include language such as:

- "we agreed"
- "we decided"
- "it is confirmed"
- "we will proceed"
- "the final decision is"

Do not classify suggestions, possibilities, rejected proposals,
or unresolved discussions as decisions.

4. ACTION ITEMS
Identify tasks that someone has:

- accepted,
- been assigned,
- committed to complete.

5. OWNERS
Only assign an owner when responsibility is clearly stated
or directly identifiable from the transcript.

If no owner is identified, return null.

6. DEADLINES
Only include a deadline when a date, day, time,
or explicit timeframe is stated.

If no deadline is stated, return null.

7. EVIDENCE
Every decision and action item must contain supporting
evidence from the transcript.

8. HALLUCINATION CONTROL
Do not invent:

- decisions,
- action items,
- owners,
- deadlines,
- dates,
- facts.

9. UNRESOLVED ISSUES
Identify matters discussed but not finally resolved.

10. AMBIGUITIES
Identify statements that may require human clarification.

Before returning the result, scan the entire transcript
for completeness.

MEETING TRANSCRIPT
------------------

{transcript}
"""


# ============================================================
# 8. GEMINI ANALYSIS FUNCTION
# ============================================================

def analyse_meeting(
    transcript
):

    client = load_gemini_client()

    if client is None:

        raise RuntimeError(
            "GEMINI_API_KEY has not been configured "
            "in Streamlit secrets."
        )


    prompt = create_meeting_prompt(
        transcript
    )


    interaction = client.interactions.create(

        model=GEMINI_MODEL,

        input=prompt,

        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema":
                MeetingIntelligence
                .model_json_schema()
        }
    )


    intelligence = (
        MeetingIntelligence
        .model_validate_json(
            interaction.output_text
        )
    )


    return intelligence


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎙️ AIMIS")

    st.caption(
        "AI Meeting Intelligence System"
    )

    st.divider()

    st.markdown(
        "### System Pipeline"
    )

    st.markdown(
        """
        **1. Upload Audio**

        ↓

        **2. Whisper ASR**

        ↓

        **3. Meeting Transcript**

        ↓

        **4. Gemini Analysis**

        ↓

        **5. Meeting Intelligence**
        """
    )

    st.divider()

    st.markdown(
        "### AI Models"
    )

    st.write(
        f"**Speech-to-Text:** {WHISPER_MODEL_NAME}"
    )

    st.write(
        f"**Meeting Intelligence:** {GEMINI_MODEL}"
    )

    st.divider()

    st.caption(
        "MSc Computing Project Prototype"
    )


# ============================================================
# 10. HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="main-title">
        🎙️ AI Meeting Intelligence System
        </div>

        <div class="subtitle">
        Automatically transform meeting audio into
        transcripts, summaries, decisions, action items,
        owners, deadlines and follow-up intelligence.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. FEATURE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🎧 Input",
        "Audio Meeting"
    )

with col2:

    st.metric(
        "📝 ASR",
        "Whisper"
    )

with col3:

    st.metric(
        "🧠 Intelligence",
        "Gemini"
    )

with col4:

    st.metric(
        "📋 Output",
        "Structured Report"
    )


st.divider()


# ============================================================
# 12. AUDIO UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Meeting Audio</div>',
    unsafe_allow_html=True
)


uploaded_audio = st.file_uploader(

    "Upload an MP3, WAV or M4A meeting recording",

    type=[
        "mp3",
        "wav",
        "m4a"
    ],

    help=(
        "The recording will be transcribed using "
        "Whisper and analysed using Gemini."
    )
)


if uploaded_audio is None:

    st.info(
        "Upload a meeting recording to begin."
    )


# ============================================================
# 13. AUDIO PREVIEW
# ============================================================

if uploaded_audio is not None:

    st.success(
        f"Audio loaded: {uploaded_audio.name}"
    )

    st.audio(
        uploaded_audio
    )


    file_size_mb = (
        uploaded_audio.size /
        (1024 * 1024)
    )


    info1, info2 = st.columns(2)

    with info1:

        st.metric(
            "File",
            uploaded_audio.name
        )

    with info2:

        st.metric(
            "Size",
            f"{file_size_mb:.2f} MB"
        )


    st.divider()


    # ========================================================
    # 14. PROCESS BUTTON
    # ========================================================

    process_button = st.button(

        "🚀 Analyse Meeting",

        type="primary",

        use_container_width=True
    )


    if process_button:


        # ====================================================
        # CHECK FFMPEG
        # ====================================================

        if shutil.which(
            "ffmpeg"
        ) is None:

            st.error(
                "FFmpeg is not available. "
                "Ensure packages.txt contains: ffmpeg"
            )

            st.stop()


        # ====================================================
        # TEMPORARY AUDIO FILE
        # ====================================================

        file_extension = (
            os.path.splitext(
                uploaded_audio.name
            )[1]
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_audio.getbuffer()
            )

            audio_path = (
                temp_file.name
            )


        try:


            # =================================================
            # WHISPER TRANSCRIPTION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '2. Speech-to-Text Transcription'
                '</div>',
                unsafe_allow_html=True
            )


            with st.spinner(
                "Whisper is transcribing the meeting..."
            ):


                transcription_start = (
                    time.perf_counter()
                )


                whisper_model = (
                    load_whisper_model()
                )


                whisper_result = (
                    whisper_model.transcribe(
                        audio_path,
                        language="en",
                        temperature=0.0
                    )
                )


                transcript = (
                    whisper_result[
                        "text"
                    ]
                    .strip()
                )


                transcription_time = (
                    time.perf_counter()
                    -
                    transcription_start
                )


            st.success(
                "✅ Meeting transcription completed"
            )


            # =================================================
            # TRANSCRIPT
            # =================================================

            st.text_area(

                "Meeting Transcript",

                value=transcript,

                height=300
            )


            transcript_words = len(
                transcript.split()
            )


            t1, t2 = st.columns(2)

            with t1:

                st.metric(
                    "Transcript Words",
                    transcript_words
                )

            with t2:

                st.metric(
                    "Whisper Processing Time",
                    f"{transcription_time:.2f} sec"
                )


            st.download_button(

                label="⬇️ Download Transcript",

                data=transcript,

                file_name="meeting_transcript.txt",

                mime="text/plain",

                use_container_width=True
            )


            st.divider()


            # =================================================
            # GEMINI MEETING INTELLIGENCE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '3. AI Meeting Intelligence'
                '</div>',
                unsafe_allow_html=True
            )


            with st.spinner(
                "Gemini is analysing decisions, "
                "actions and meeting outcomes..."
            ):


                intelligence_start = (
                    time.perf_counter()
                )


                intelligence = (
                    analyse_meeting(
                        transcript
                    )
                )


                intelligence_time = (
                    time.perf_counter()
                    -
                    intelligence_start
                )


            st.success(
                "✅ Meeting intelligence generated"
            )


            # =================================================
            # MEETING TITLE
            # =================================================

            st.header(
                f"📌 {intelligence.meeting_title}"
            )


            # =================================================
            # RESULT METRICS
            # =================================================

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )


            with metric1:

                st.metric(
                    "Key Topics",
                    len(
                        intelligence.key_topics
                    )
                )


            with metric2:

                st.metric(
                    "Decisions",
                    len(
                        intelligence.decisions
                    )
                )


            with metric3:

                st.metric(
                    "Action Items",
                    len(
                        intelligence.action_items
                    )
                )


            with metric4:

                st.metric(
                    "AI Analysis Time",
                    f"{intelligence_time:.2f}s"
                )


            # =================================================
            # TABS
            # =================================================

            (
                summary_tab,
                decisions_tab,
                actions_tab,
                followup_tab,
                technical_tab
            ) = st.tabs(
                [
                    "📋 Summary",
                    "✅ Decisions",
                    "📌 Action Items",
                    "🔎 Follow-up",
                    "⚙️ Technical"
                ]
            )


            # =================================================
            # SUMMARY TAB
            # =================================================

            with summary_tab:

                st.subheader(
                    "Meeting Summary"
                )

                st.write(
                    intelligence.summary
                )


                st.subheader(
                    "Key Topics"
                )


                if intelligence.key_topics:

                    for topic in (
                        intelligence.key_topics
                    ):

                        st.markdown(
                            f"- **{topic}**"
                        )

                else:

                    st.info(
                        "No key topics identified."
                    )


            # =================================================
            # DECISIONS TAB
            # =================================================

            with decisions_tab:

                st.subheader(
                    "Confirmed Decisions"
                )


                if intelligence.decisions:


                    for number, decision in enumerate(
                        intelligence.decisions,
                        start=1
                    ):


                        st.markdown(
                            f"### Decision {number}"
                        )


                        st.write(
                            decision.decision
                        )


                        st.caption(
                            f"Confidence: "
                            f"{decision.confidence:.0%}"
                        )


                        with st.expander(
                            "View supporting evidence"
                        ):

                            st.write(
                                decision.evidence
                            )


                else:

                    st.info(
                        "No confirmed decisions "
                        "were identified."
                    )


            # =================================================
            # ACTION ITEMS TAB
            # =================================================

            with actions_tab:

                st.subheader(
                    "Action Items"
                )


                if intelligence.action_items:


                    action_rows = []


                    for action in (
                        intelligence.action_items
                    ):


                        action_rows.append({

                            "Action":
                                action.action,

                            "Owner":
                                action.owner
                                or "Not specified",

                            "Deadline":
                                action.deadline
                                or "Not specified",

                            "Status":
                                action.status,

                            "Confidence":
                                f"{action.confidence:.0%}"
                        })


                    actions_df = (
                        pd.DataFrame(
                            action_rows
                        )
                    )


                    st.dataframe(

                        actions_df,

                        use_container_width=True,

                        hide_index=True
                    )


                    st.subheader(
                        "Supporting Evidence"
                    )


                    for number, action in enumerate(
                        intelligence.action_items,
                        start=1
                    ):


                        with st.expander(
                            f"Action {number}: "
                            f"{action.action}"
                        ):

                            st.write(
                                action.evidence
                            )


                else:

                    st.info(
                        "No action items identified."
                    )


            # =================================================
            # FOLLOW-UP TAB
            # =================================================

            with followup_tab:

                st.subheader(
                    "Unresolved Issues"
                )


                if intelligence.unresolved_issues:

                    for issue in (
                        intelligence.unresolved_issues
                    ):

                        st.markdown(
                            f"- {issue}"
                        )

                else:

                    st.success(
                        "No unresolved issues identified."
                    )


                st.subheader(
                    "Ambiguities"
                )


                if intelligence.ambiguities:

                    for ambiguity in (
                        intelligence.ambiguities
                    ):

                        st.markdown(
                            f"- {ambiguity}"
                        )

                else:

                    st.success(
                        "No significant ambiguities identified."
                    )


            # =================================================
            # TECHNICAL TAB
            # =================================================

            with technical_tab:

                st.subheader(
                    "Processing Information"
                )


                technical_df = pd.DataFrame(
                    {
                        "Component": [
                            "Speech Recognition",
                            "Language Model",
                            "Transcript Words",
                            "Whisper Time",
                            "Gemini Time",
                            "Total AI Processing Time"
                        ],

                        "Value": [
                            WHISPER_MODEL_NAME,
                            GEMINI_MODEL,
                            transcript_words,
                            f"{transcription_time:.2f} sec",
                            f"{intelligence_time:.2f} sec",
                            (
                                f"{transcription_time + intelligence_time:.2f} sec"
                            )
                        ]
                    }
                )


                st.dataframe(

                    technical_df,

                    use_container_width=True,

                    hide_index=True
                )


            # =================================================
            # DOWNLOAD COMPLETE REPORT
            # =================================================

            st.divider()


            report_text = f"""
AI MEETING INTELLIGENCE REPORT
==============================

Meeting Title:
{intelligence.meeting_title}


SUMMARY
-------

{intelligence.summary}


KEY TOPICS
----------

{chr(10).join(
    "- " + topic
    for topic in intelligence.key_topics
)}


DECISIONS
---------

{chr(10).join(
    f"{i+1}. {d.decision}"
    for i, d in enumerate(
        intelligence.decisions
    )
) or "No confirmed decisions identified."}


ACTION ITEMS
------------

{chr(10).join(
    f"{i+1}. {a.action} | "
    f"Owner: {a.owner or 'Not specified'} | "
    f"Deadline: {a.deadline or 'Not specified'}"
    for i, a in enumerate(
        intelligence.action_items
    )
) or "No action items identified."}


UNRESOLVED ISSUES
-----------------

{chr(10).join(
    "- " + item
    for item in intelligence.unresolved_issues
) or "None identified."}


AMBIGUITIES
-----------

{chr(10).join(
    "- " + item
    for item in intelligence.ambiguities
) or "None identified."}


SYSTEM INFORMATION
------------------

Whisper Model: {WHISPER_MODEL_NAME}
Gemini Model: {GEMINI_MODEL}
Whisper Processing Time: {transcription_time:.2f} seconds
Gemini Processing Time: {intelligence_time:.2f} seconds


Generated by:
AI Meeting Intelligence System (AIMIS)
MSc Computing Project Prototype
"""


            st.download_button(

                label="📥 Download Complete Meeting Report",

                data=report_text,

                file_name="AIMIS_meeting_report.txt",

                mime="text/plain",

                type="primary",

                use_container_width=True
            )


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as error:

            st.error(
                "The meeting could not be processed."
            )

            st.exception(
                error
            )


        # ====================================================
        # REMOVE TEMPORARY AUDIO
        # ====================================================

        finally:

            try:

                if os.path.exists(
                    audio_path
                ):

                    os.remove(
                        audio_path
                    )

            except Exception:

                pass


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    <strong>
    AI Meeting Intelligence System (AIMIS)
    </strong>

    <br>

    MSc Computing Project Prototype

    <br><br>

    Whisper Speech Recognition
    •
    Gemini Meeting Intelligence
    •
    Structured Decision & Action Extraction

    </div>
    """,
    unsafe_allow_html=True
)
