# ============================================================
# AI MEETING INTELLIGENCE SYSTEM (AIMIS)
# MSc Computing Project Prototype
# ============================================================

import os
import time
import shutil
import tempfile
from typing import List, Optional, Literal

import pandas as pd
import streamlit as st
import whisper

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIMIS | AI Meeting Intelligence System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.main-title {
    font-size: 45px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 10px;
}

.main-subtitle {
    font-size: 18px;
    opacity: 0.80;
    line-height: 1.6;
    margin-bottom: 5px;
}

.hero-box {
    padding: 28px;
    border-radius: 18px;
    border: 1px solid rgba(120,120,120,0.25);
    background: linear-gradient(
        135deg,
        rgba(30, 70, 140, 0.25),
        rgba(20, 25, 35, 0.15)
    );
    margin-bottom: 25px;
}

.section-label {
    font-size: 28px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

.footer-box {
    text-align: center;
    opacity: 0.65;
    font-size: 13px;
    padding-top: 15px;
    padding-bottom: 15px;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 3. SYSTEM CONFIGURATION
# ============================================================

WHISPER_MODEL_NAME = "small.en"

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# 4. STRUCTURED MEETING INTELLIGENCE SCHEMA
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
        description="Task or action identified during the meeting."
    )

    owner: Optional[str] = Field(
        default=None,
        description=(
            "Responsible person if explicitly identifiable "
            "from the transcript."
        )
    )

    deadline: Optional[str] = Field(
        default=None,
        description=(
            "Deadline if explicitly stated in the transcript."
        )
    )

    status: Literal[
        "assigned",
        "proposed",
        "unclear"
    ]

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
# 5. LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():

    model = whisper.load_model(
        WHISPER_MODEL_NAME
    )

    return model


# ============================================================
# 6. LOAD GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        return None


    return genai.Client(
        api_key=api_key
    )


# ============================================================
# 7. CREATE STRUCTURED GEMINI PROMPT
# ============================================================

def create_meeting_prompt(
    transcript: str
):

    return f"""
You are an AI Meeting Intelligence System.

Analyse the entire meeting transcript carefully.

Your task is to extract accurate and structured
meeting intelligence.

Follow all rules below.


1. MEETING SUMMARY

Provide a concise and factual summary of the meeting.


2. KEY TOPICS

Identify the main subjects discussed.


3. CONFIRMED DECISIONS

Record only decisions that were clearly confirmed.

Examples of decision signals include:

- "we agreed"
- "we decided"
- "it is confirmed"
- "the final decision is"
- "we will proceed with"

Do NOT treat the following as confirmed decisions:

- suggestions
- possibilities
- rejected proposals
- opinions
- unresolved alternatives


4. ACTION ITEMS

Record tasks where someone:

- accepted responsibility,
- was directly assigned responsibility,
- explicitly committed to completing something.


5. OWNERS

Only record an owner when the responsible person
is identifiable from the transcript.

If no owner is identifiable:

owner = null


6. DEADLINES

Only record a deadline if a date, day, time,
or explicit timeframe is stated.

If no deadline is given:

deadline = null


7. ACTION STATUS

Use one of:

assigned
proposed
unclear


8. EVIDENCE

Every decision and every action item must contain
supporting evidence from the transcript.


9. HALLUCINATION CONTROL

Never invent:

- decisions
- action items
- owners
- deadlines
- dates
- people
- facts


10. UNRESOLVED ISSUES

Identify matters discussed but not finally resolved.


11. AMBIGUITIES

Identify statements requiring clarification
or human interpretation.


12. FINAL COMPLETENESS CHECK

Before returning the answer:

- scan the entire transcript,
- check every confirmed decision,
- check every assigned action,
- check owners,
- check deadlines,
- check unresolved matters,
- avoid unsupported information.


MEETING TRANSCRIPT
------------------

{transcript}
"""


# ============================================================
# 8. GEMINI ANALYSIS FUNCTION
# ============================================================

def analyse_meeting(
    transcript: str
):

    client = load_gemini_client()


    if client is None:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured "
            "in Streamlit Secrets."
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


    result = (
        MeetingIntelligence
        .model_validate_json(
            interaction.output_text
        )
    )


    return result


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🎙️ AIMIS"
    )

    st.caption(
        "AI Meeting Intelligence System"
    )

    st.divider()


    st.subheader(
        "System Pipeline"
    )


    st.markdown(
        """
**1. Upload Meeting Audio**

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


    st.subheader(
        "System Components"
    )


    st.write(
        f"🎧 **Speech Recognition**  \n"
        f"{WHISPER_MODEL_NAME}"
    )


    st.write(
        f"🧠 **Meeting Intelligence**  \n"
        f"{GEMINI_MODEL}"
    )


    st.divider()


    st.caption(
        "MSc Computing Project Prototype"
    )


# ============================================================
# 10. HERO SECTION
#
# IMPORTANT:
# HTML begins from the left edge.
# This prevents Streamlit displaying it as code.
# ============================================================

st.markdown(
    """
<div class="hero-box">
<div class="main-title">🎙️ AI Meeting Intelligence System</div>
<div class="main-subtitle">
Automatically transform meeting audio into transcripts, summaries,
decisions, action items, responsible owners, deadlines,
unresolved issues and follow-up intelligence.
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 11. SYSTEM WORKFLOW CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        label="🎧 Input",
        value="Audio Meeting"
    )


with col2:

    st.metric(
        label="📝 Speech-to-Text",
        value="Whisper"
    )


with col3:

    st.metric(
        label="🧠 Intelligence",
        value="Gemini"
    )


with col4:

    st.metric(
        label="📋 Output",
        value="Structured Report"
    )


st.divider()


# ============================================================
# 12. UPLOAD MEETING AUDIO
# ============================================================

st.markdown(
    '<div class="section-label">'
    '1. Upload Meeting Audio'
    '</div>',
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
        "The uploaded recording will be transcribed "
        "using Whisper and analysed using Gemini."
    )
)


# ============================================================
# 13. WAITING STATE
# ============================================================

if uploaded_audio is None:

    st.info(
        "👆 Upload a meeting recording to begin the analysis."
    )


# ============================================================
# 14. AUDIO SELECTED
# ============================================================

if uploaded_audio is not None:


    st.success(
        f"✅ Audio loaded successfully: "
        f"{uploaded_audio.name}"
    )


    # --------------------------------------------------------
    # AUDIO PLAYER
    # --------------------------------------------------------

    st.audio(
        uploaded_audio
    )


    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    file_size_mb = (
        uploaded_audio.size /
        (1024 * 1024)
    )


    info_col1, info_col2 = (
        st.columns(2)
    )


    with info_col1:

        st.metric(
            "Audio File",
            uploaded_audio.name
        )


    with info_col2:

        st.metric(
            "File Size",
            f"{file_size_mb:.2f} MB"
        )


    st.divider()


    # ========================================================
    # 15. PROCESS BUTTON
    # ========================================================

    process_button = st.button(

        "🚀 Process Meeting",

        type="primary",

        use_container_width=True
    )


    if process_button:


        # ====================================================
        # 16. CHECK FFMPEG
        # ====================================================

        if shutil.which(
            "ffmpeg"
        ) is None:

            st.error(
                "FFmpeg is not available on the server."
            )

            st.info(
                "Make sure packages.txt contains: ffmpeg"
            )

            st.stop()


        # ====================================================
        # 17. SAVE AUDIO TEMPORARILY
        # ====================================================

        file_extension = os.path.splitext(
            uploaded_audio.name
        )[1]


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
            # 18. WHISPER TRANSCRIPTION
            # =================================================

            st.markdown(
                '<div class="section-label">'
                '2. Meeting Transcription'
                '</div>',
                unsafe_allow_html=True
            )


            with st.spinner(
                "🎧 Whisper is transcribing the meeting..."
            ):


                whisper_start = (
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


                whisper_time = (
                    time.perf_counter()
                    -
                    whisper_start
                )


            st.success(
                "✅ Meeting transcription completed."
            )


            # =================================================
            # 19. TRANSCRIPT DISPLAY
            # =================================================

            st.subheader(
                "📝 Meeting Transcript"
            )


            st.text_area(

                "Transcribed meeting content",

                value=transcript,

                height=300,

                key="meeting_transcript"
            )


            transcript_word_count = len(
                transcript.split()
            )


            trans_col1, trans_col2 = (
                st.columns(2)
            )


            with trans_col1:

                st.metric(
                    "Transcript Words",
                    transcript_word_count
                )


            with trans_col2:

                st.metric(
                    "Whisper Processing Time",
                    f"{whisper_time:.2f} sec"
                )


            st.download_button(

                label="⬇️ Download Transcript",

                data=transcript,

                file_name=(
                    "AIMIS_meeting_transcript.txt"
                ),

                mime="text/plain",

                use_container_width=True
            )


            st.divider()


            # =================================================
            # 20. GEMINI ANALYSIS
            # =================================================

            st.markdown(
                '<div class="section-label">'
                '3. AI Meeting Intelligence'
                '</div>',
                unsafe_allow_html=True
            )


            with st.spinner(
                "🧠 Gemini is analysing meeting "
                "decisions, actions and outcomes..."
            ):


                gemini_start = (
                    time.perf_counter()
                )


                intelligence = (
                    analyse_meeting(
                        transcript
                    )
                )


                gemini_time = (
                    time.perf_counter()
                    -
                    gemini_start
                )


            st.success(
                "✅ Meeting intelligence generated successfully."
            )


            # =================================================
            # 21. MEETING TITLE
            # =================================================

            st.header(
                f"📌 {intelligence.meeting_title}"
            )


            # =================================================
            # 22. ANALYSIS METRICS
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
                    "Gemini Time",
                    f"{gemini_time:.2f} sec"
                )


            st.divider()


            # =================================================
            # 23. RESULT TABS
            # =================================================

            (
                summary_tab,
                decision_tab,
                action_tab,
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
            # 24. SUMMARY TAB
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


                if (
                    intelligence.key_topics
                ):


                    for topic in (
                        intelligence.key_topics
                    ):

                        st.markdown(
                            f"- {topic}"
                        )


                else:

                    st.info(
                        "No key topics identified."
                    )


            # =================================================
            # 25. DECISIONS TAB
            # =================================================

            with decision_tab:


                st.subheader(
                    "Confirmed Decisions"
                )


                if (
                    intelligence.decisions
                ):


                    for index, decision in enumerate(
                        intelligence.decisions,
                        start=1
                    ):


                        st.markdown(
                            f"### Decision {index}"
                        )


                        st.write(
                            decision.decision
                        )


                        st.caption(
                            "Confidence: "
                            f"{decision.confidence:.0%}"
                        )


                        with st.expander(
                            "View supporting evidence"
                        ):


                            st.write(
                                decision.evidence
                            )


                        st.divider()


                else:

                    st.info(
                        "No confirmed decisions "
                        "were identified."
                    )


            # =================================================
            # 26. ACTION ITEMS TAB
            # =================================================

            with action_tab:


                st.subheader(
                    "Action Items"
                )


                if (
                    intelligence.action_items
                ):


                    action_rows = []


                    for item in (
                        intelligence.action_items
                    ):


                        action_rows.append(
                            {
                                "Action":
                                    item.action,

                                "Owner":
                                    item.owner
                                    or "Not specified",

                                "Deadline":
                                    item.deadline
                                    or "Not specified",

                                "Status":
                                    item.status,

                                "Confidence":
                                    (
                                        f"{item.confidence:.0%}"
                                    )
                            }
                        )


                    action_dataframe = (
                        pd.DataFrame(
                            action_rows
                        )
                    )


                    st.dataframe(

                        action_dataframe,

                        use_container_width=True,

                        hide_index=True
                    )


                    st.subheader(
                        "Action Evidence"
                    )


                    for index, item in enumerate(
                        intelligence.action_items,
                        start=1
                    ):


                        with st.expander(
                            f"Action {index}: "
                            f"{item.action}"
                        ):


                            st.write(
                                item.evidence
                            )


                else:

                    st.info(
                        "No action items were identified."
                    )


            # =================================================
            # 27. FOLLOW-UP TAB
            # =================================================

            with followup_tab:


                st.subheader(
                    "Unresolved Issues"
                )


                if (
                    intelligence.unresolved_issues
                ):


                    for issue in (
                        intelligence.unresolved_issues
                    ):

                        st.markdown(
                            f"- {issue}"
                        )


                else:

                    st.success(
                        "✅ No unresolved issues identified."
                    )


                st.divider()


                st.subheader(
                    "Ambiguities"
                )


                if (
                    intelligence.ambiguities
                ):


                    for ambiguity in (
                        intelligence.ambiguities
                    ):

                        st.markdown(
                            f"- {ambiguity}"
                        )


                else:

                    st.success(
                        "✅ No significant ambiguities identified."
                    )


            # =================================================
            # 28. TECHNICAL TAB
            # =================================================

            with technical_tab:


                st.subheader(
                    "System Processing Information"
                )


                total_processing_time = (
                    whisper_time +
                    gemini_time
                )


                technical_df = pd.DataFrame(
                    {
                        "Component": [
                            "Speech Recognition",
                            "Language Model",
                            "Transcript Words",
                            "Whisper Processing Time",
                            "Gemini Processing Time",
                            "Total AI Processing Time"
                        ],

                        "Value": [
                            WHISPER_MODEL_NAME,
                            GEMINI_MODEL,
                            transcript_word_count,
                            f"{whisper_time:.2f} sec",
                            f"{gemini_time:.2f} sec",
                            f"{total_processing_time:.2f} sec"
                        ]
                    }
                )


                st.dataframe(

                    technical_df,

                    use_container_width=True,

                    hide_index=True
                )


            # =================================================
            # 29. CREATE DOWNLOADABLE MEETING REPORT
            # =================================================

            st.divider()


            decision_text = "\n".join(
                [
                    (
                        f"{index}. {decision.decision}\n"
                        f"   Evidence: {decision.evidence}\n"
                        f"   Confidence: "
                        f"{decision.confidence:.0%}"
                    )

                    for index, decision in enumerate(
                        intelligence.decisions,
                        start=1
                    )
                ]
            )


            if not decision_text:

                decision_text = (
                    "No confirmed decisions identified."
                )


            action_text = "\n".join(
                [
                    (
                        f"{index}. {item.action}\n"
                        f"   Owner: "
                        f"{item.owner or 'Not specified'}\n"
                        f"   Deadline: "
                        f"{item.deadline or 'Not specified'}\n"
                        f"   Status: {item.status}\n"
                        f"   Evidence: {item.evidence}\n"
                        f"   Confidence: "
                        f"{item.confidence:.0%}"
                    )

                    for index, item in enumerate(
                        intelligence.action_items,
                        start=1
                    )
                ]
            )


            if not action_text:

                action_text = (
                    "No action items identified."
                )


            topics_text = "\n".join(
                [
                    f"- {topic}"
                    for topic in (
                        intelligence.key_topics
                    )
                ]
            )


            unresolved_text = "\n".join(
                [
                    f"- {issue}"
                    for issue in (
                        intelligence.unresolved_issues
                    )
                ]
            )


            if not unresolved_text:

                unresolved_text = (
                    "No unresolved issues identified."
                )


            ambiguity_text = "\n".join(
                [
                    f"- {item}"
                    for item in (
                        intelligence.ambiguities
                    )
                ]
            )


            if not ambiguity_text:

                ambiguity_text = (
                    "No significant ambiguities identified."
                )


            report_text = f"""
AI MEETING INTELLIGENCE SYSTEM
==============================

MEETING TITLE
-------------

{intelligence.meeting_title}


MEETING SUMMARY
---------------

{intelligence.summary}


KEY TOPICS
----------

{topics_text}


CONFIRMED DECISIONS
-------------------

{decision_text}


ACTION ITEMS
------------

{action_text}


UNRESOLVED ISSUES
-----------------

{unresolved_text}


AMBIGUITIES
-----------

{ambiguity_text}


SYSTEM PROCESSING INFORMATION
-----------------------------

Whisper Model:
{WHISPER_MODEL_NAME}

Gemini Model:
{GEMINI_MODEL}

Transcript Word Count:
{transcript_word_count}

Whisper Processing Time:
{whisper_time:.2f} seconds

Gemini Processing Time:
{gemini_time:.2f} seconds

Total AI Processing Time:
{whisper_time + gemini_time:.2f} seconds


Generated by:
AI Meeting Intelligence System (AIMIS)
MSc Computing Project Prototype
"""


            # =================================================
            # 30. DOWNLOAD COMPLETE REPORT
            # =================================================

            st.subheader(
                "📥 Export Meeting Intelligence"
            )


            st.download_button(

                label=(
                    "Download Complete Meeting Report"
                ),

                data=report_text,

                file_name=(
                    "AIMIS_meeting_intelligence_report.txt"
                ),

                mime="text/plain",

                type="primary",

                use_container_width=True
            )


        # ====================================================
        # 31. ERROR HANDLING
        # ====================================================

        except Exception as error:


            st.error(
                "❌ The meeting could not be processed."
            )


            error_text = str(
                error
            )


            if (
                "429" in error_text
                or
                "too_many_requests" in error_text.lower()
            ):


                st.warning(
                    "Gemini is temporarily rate limited. "
                    "Please wait and try again."
                )


            elif (
                "GEMINI_API_KEY"
                in error_text
            ):


                st.warning(
                    "The Gemini API key has not been "
                    "configured correctly."
                )


            else:


                st.exception(
                    error
                )


        # ====================================================
        # 32. CLEAN TEMPORARY AUDIO
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
# 33. FOOTER
# ============================================================

st.divider()


st.markdown(
    """
<div class="footer-box">
<strong>AI Meeting Intelligence System (AIMIS)</strong>
<br>
MSc Computing Project Prototype
<br><br>
Whisper Speech Recognition • Gemini Meeting Intelligence •
Structured Decision and Action Extraction
</div>
""",
    unsafe_allow_html=True
)
