# ============================================================
# AI MEETING INTELLIGENCE SYSTEM (AIMIS)
# MSc Computing Project Prototype
# ============================================================

import os
import re
import time
import shutil
import tempfile
import subprocess
from typing import List, Optional, Literal

import pandas as pd
import streamlit as st
import torch
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
# 2. SYSTEM SETTINGS
# ============================================================

WHISPER_MODEL_NAME = "small.en"
GEMINI_MODEL = "gemini-3.6-flash"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. SMALL CSS IMPROVEMENTS
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.22);
    padding: 14px;
    border-radius: 12px;
}

.aimis-subtitle {
    font-size: 1.05rem;
    opacity: 0.75;
    margin-bottom: 1.5rem;
}

.aimis-footer {
    text-align: center;
    opacity: 0.60;
    font-size: 0.85rem;
    padding: 15px;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 4. STRUCTURED OUTPUT MODELS
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
        description=(
            "Responsible person when explicitly identifiable."
        )
    )

    deadline: Optional[str] = Field(
        default=None,
        description=(
            "Deadline when explicitly stated."
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
# 5. SESSION STATE
# ============================================================

SESSION_DEFAULTS = {

    "upload_signature": None,

    "transcript": "",

    "editable_transcript": "",

    "intelligence": None,

    "audio_duration": None,

    "whisper_time": None,

    "gemini_time": None,
}


for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 6. RESET FUNCTION
# ============================================================

def reset_analysis():

    st.session_state.transcript = ""
    st.session_state.editable_transcript = ""
    st.session_state.intelligence = None
    st.session_state.audio_duration = None
    st.session_state.whisper_time = None
    st.session_state.gemini_time = None


# ============================================================
# 7. LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model(
        WHISPER_MODEL_NAME,
        device=DEVICE
    )


# ============================================================
# 8. GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        return None


    if not api_key:

        return None


    return genai.Client(
        api_key=api_key
    )


# ============================================================
# 9. NORMALISE AUDIO
#
# Every uploaded recording is converted to:
# - WAV
# - PCM 16-bit
# - Mono
# - 16 kHz
#
# before Whisper receives it.
# ============================================================

def normalise_audio_for_whisper(
    input_path: str,
    output_path: str
):

    command = [

        "ffmpeg",

        "-y",

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        input_path,

        "-vn",

        "-ac",
        "1",

        "-ar",
        "16000",

        "-c:a",
        "pcm_s16le",

        output_path
    ]


    process = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True
    )


    if process.returncode != 0:

        raise RuntimeError(
            "FFmpeg could not decode the uploaded recording.\n\n"
            + process.stderr[-1500:]
        )


    if not os.path.exists(
        output_path
    ):

        raise RuntimeError(
            "FFmpeg did not create the normalised audio file."
        )


    if os.path.getsize(
        output_path
    ) <= 1000:

        raise RuntimeError(
            "The converted audio file appears to be empty."
        )


    # --------------------------------------------------------
    # VERIFY WHISPER CAN READ THE AUDIO
    # --------------------------------------------------------

    audio_array = whisper.load_audio(
        output_path
    )


    if len(audio_array) == 0:

        raise RuntimeError(
            "No readable audio samples were detected."
        )


    duration_seconds = (
        len(audio_array) /
        16000
    )


    if duration_seconds < 1.0:

        raise RuntimeError(
            "The meeting recording is too short. "
            "Please upload audio longer than one second."
        )


    return duration_seconds


# ============================================================
# 10. WHISPER TRANSCRIPTION
# ============================================================

def transcribe_meeting(
    audio_path: str
):

    model = load_whisper_model()


    start_time = (
        time.perf_counter()
    )


    result = model.transcribe(

        audio_path,

        language="en",

        task="transcribe",

        temperature=0.0,

        fp16=(
            DEVICE == "cuda"
        ),

        condition_on_previous_text=False,

        verbose=False
    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    transcript = (
        result.get(
            "text",
            ""
        )
        .strip()
    )


    if not transcript:

        raise RuntimeError(
            "Whisper completed processing but produced "
            "an empty transcript. Please check the recording."
        )


    return transcript, elapsed


# ============================================================
# 11. GEMINI PROMPT
# ============================================================

def create_meeting_prompt(
    transcript: str
):

    return f"""
You are an AI Meeting Intelligence System.

Analyse the complete meeting transcript carefully.

Extract accurate structured meeting intelligence.

Follow these rules.


1. MEETING TITLE

Create a short factual title based on the meeting.


2. SUMMARY

Write a concise factual summary.


3. KEY TOPICS

Identify the major subjects discussed.


4. CONFIRMED DECISIONS

Include only confirmed decisions.

Examples of confirmation language include:

- we agreed
- we decided
- it is confirmed
- the final decision is
- we will proceed with

Do not convert suggestions, possibilities, rejected options,
opinions or unresolved discussions into decisions.


5. ACTION ITEMS

Identify tasks where somebody:

- accepted responsibility,
- was assigned responsibility,
- committed to completing something.


6. OWNERS

Only identify an owner when responsibility is explicit
or clearly attributable from the transcript.

Otherwise:

owner = null


7. DEADLINES

Only include a deadline when a date, day, time
or explicit timeframe appears in the transcript.

Otherwise:

deadline = null


8. ACTION STATUS

Use exactly one of:

assigned
proposed
unclear


9. EVIDENCE

Every decision and action item must include
supporting transcript evidence.


10. HALLUCINATION CONTROL

Never invent:

- decisions
- action items
- people
- owners
- deadlines
- dates
- facts


11. UNRESOLVED ISSUES

Identify matters that were discussed but not resolved.


12. AMBIGUITIES

Identify information requiring clarification.


13. FINAL CHECK

Before completing the response:

- scan the entire transcript,
- check all confirmed decisions,
- check all actions,
- check owners,
- check deadlines,
- check unresolved matters,
- remove unsupported information.


MEETING TRANSCRIPT
------------------

{transcript}
"""


# ============================================================
# 12. GEMINI ANALYSIS
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


    start_time = (
        time.perf_counter()
    )


    interaction = client.interactions.create(

        model=GEMINI_MODEL,

        input=prompt,

        response_format={

            "type": "text",

            "mime_type":
                "application/json",

            "schema":
                MeetingIntelligence
                .model_json_schema()
        }
    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    intelligence = (

        MeetingIntelligence

        .model_validate_json(
            interaction.output_text
        )
    )


    return intelligence, elapsed


# ============================================================
# 13. GEMINI ERROR MESSAGE
# ============================================================

def show_gemini_error(
    error
):

    error_text = str(
        error
    )


    lower_error = (
        error_text.lower()
    )


    if (
        "429" in lower_error
        or
        "too_many_requests"
        in lower_error
    ):

        retry_match = re.search(
            r"retry in\s+([0-9.]+)s",
            error_text,
            flags=re.IGNORECASE
        )


        if retry_match:

            wait_time = float(
                retry_match.group(1)
            )


            st.warning(
                f"Gemini is temporarily rate limited. "
                f"Please wait approximately "
                f"{wait_time:.0f} seconds and try again."
            )

        else:

            st.warning(
                "Gemini is temporarily rate limited. "
                "Please wait and try again."
            )


    elif (
        "quota_exceeded"
        in lower_error
    ):

        st.warning(
            "The available Gemini API quota "
            "has been exhausted."
        )


    elif (
        "gemini_api_key"
        in lower_error
        or
        "api key"
        in lower_error
    ):

        st.error(
            "The Gemini API key is missing or invalid."
        )


    else:

        st.error(
            "Gemini could not analyse the transcript."
        )


        with st.expander(
            "Technical details"
        ):

            st.code(
                error_text
            )


# ============================================================
# 14. REPORT CREATOR
# ============================================================

def create_report(
    intelligence,
    transcript,
    whisper_time,
    gemini_time,
    audio_duration
):

    topics_text = "\n".join(
        [
            f"- {topic}"
            for topic
            in intelligence.key_topics
        ]
    )


    if not topics_text:

        topics_text = (
            "No key topics identified."
        )


    decisions_text = "\n\n".join(
        [
            (
                f"{i}. {decision.decision}\n"
                f"Evidence: {decision.evidence}\n"
                f"Confidence: "
                f"{decision.confidence:.0%}"
            )

            for i, decision
            in enumerate(
                intelligence.decisions,
                start=1
            )
        ]
    )


    if not decisions_text:

        decisions_text = (
            "No confirmed decisions identified."
        )


    actions_text = "\n\n".join(
        [
            (
                f"{i}. {item.action}\n"
                f"Owner: "
                f"{item.owner or 'Not specified'}\n"
                f"Deadline: "
                f"{item.deadline or 'Not specified'}\n"
                f"Status: {item.status}\n"
                f"Evidence: {item.evidence}\n"
                f"Confidence: "
                f"{item.confidence:.0%}"
            )

            for i, item
            in enumerate(
                intelligence.action_items,
                start=1
            )
        ]
    )


    if not actions_text:

        actions_text = (
            "No action items identified."
        )


    unresolved_text = "\n".join(
        [
            f"- {issue}"
            for issue
            in intelligence.unresolved_issues
        ]
    )


    if not unresolved_text:

        unresolved_text = (
            "No unresolved issues identified."
        )


    ambiguity_text = "\n".join(
        [
            f"- {item}"
            for item
            in intelligence.ambiguities
        ]
    )


    if not ambiguity_text:

        ambiguity_text = (
            "No significant ambiguities identified."
        )


    return f"""
AI MEETING INTELLIGENCE SYSTEM (AIMIS)
======================================

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

{decisions_text}


ACTION ITEMS
------------

{actions_text}


UNRESOLVED ISSUES
-----------------

{unresolved_text}


AMBIGUITIES
-----------

{ambiguity_text}


MEETING TRANSCRIPT
------------------

{transcript}


SYSTEM PROCESSING INFORMATION
-----------------------------

Whisper Model:
{WHISPER_MODEL_NAME}

Whisper Device:
{DEVICE}

Gemini Model:
{GEMINI_MODEL}

Audio Duration:
{audio_duration:.2f} seconds

Transcript Word Count:
{len(transcript.split())}

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


# ============================================================
# 15. SIDEBAR
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
**1. Upload Audio**

↓

**2. Audio Validation**

↓

**3. Whisper ASR**

↓

**4. Review Transcript**

↓

**5. Gemini Analysis**

↓

**6. Meeting Intelligence**
"""
    )


    st.divider()


    st.subheader(
        "System Components"
    )


    st.write(
        f"**Whisper:** {WHISPER_MODEL_NAME}"
    )


    st.write(
        f"**Whisper device:** {DEVICE}"
    )


    st.write(
        f"**Gemini:** {GEMINI_MODEL}"
    )


    st.divider()


    if st.button(
        "🔄 Reset Meeting",
        use_container_width=True
    ):

        reset_analysis()

        st.session_state.upload_signature = None

        st.rerun()


# ============================================================
# 16. MAIN HEADER
# ============================================================

st.title(
    "🎙️ AI Meeting Intelligence System"
)


st.markdown(
    """
<div class="aimis-subtitle">
Transform meeting audio into transcripts, summaries,
confirmed decisions, action items, responsible owners,
deadlines and follow-up intelligence.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 17. WORKFLOW METRICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "🎧 Input",
        "Meeting Audio"
    )


with col2:

    st.metric(
        "📝 Transcription",
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
# 18. STEP 1 — UPLOAD AUDIO
# ============================================================

st.header(
    "1. Upload Meeting Audio"
)


uploaded_audio = st.file_uploader(

    "Upload an MP3, WAV or M4A meeting recording",

    type=[
        "mp3",
        "wav",
        "m4a"
    ]
)


# ============================================================
# 19. AUDIO UPLOADED
# ============================================================

if uploaded_audio is not None:


    current_signature = (
        f"{uploaded_audio.name}:"
        f"{uploaded_audio.size}"
    )


    if (
        st.session_state.upload_signature
        != current_signature
    ):

        reset_analysis()

        st.session_state.upload_signature = (
            current_signature
        )


    st.audio(
        uploaded_audio
    )


    file_size_mb = (
        uploaded_audio.size /
        (1024 * 1024)
    )


    file_col1, file_col2 = (
        st.columns(2)
    )


    with file_col1:

        st.metric(
            "File",
            uploaded_audio.name
        )


    with file_col2:

        st.metric(
            "Size",
            f"{file_size_mb:.2f} MB"
        )


    # ========================================================
    # 20. TRANSCRIBE BUTTON
    # ========================================================

    if st.button(
        "🎧 Transcribe Meeting",
        type="primary",
        use_container_width=True
    ):


        if shutil.which(
            "ffmpeg"
        ) is None:

            st.error(
                "FFmpeg is not available. "
                "Ensure packages.txt contains ffmpeg."
            )

            st.stop()


        temporary_directory = (
            tempfile.mkdtemp(
                prefix="aimis_"
            )
        )


        raw_extension = (
            os.path.splitext(
                uploaded_audio.name
            )[1]
        )


        raw_audio_path = os.path.join(
            temporary_directory,
            f"uploaded_audio{raw_extension}"
        )


        normalised_audio_path = os.path.join(
            temporary_directory,
            "whisper_input.wav"
        )


        try:


            # ------------------------------------------------
            # SAVE UPLOAD
            # ------------------------------------------------

            with open(
                raw_audio_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_audio.getbuffer()
                )


            # ------------------------------------------------
            # NORMALISE AUDIO
            # ------------------------------------------------

            with st.spinner(
                "Preparing audio for Whisper..."
            ):

                duration = (
                    normalise_audio_for_whisper(
                        raw_audio_path,
                        normalised_audio_path
                    )
                )


            # ------------------------------------------------
            # TRANSCRIBE
            # ------------------------------------------------

            with st.spinner(
                "Whisper is transcribing the meeting..."
            ):

                transcript, whisper_time = (
                    transcribe_meeting(
                        normalised_audio_path
                    )
                )


            st.session_state.transcript = (
                transcript
            )


            st.session_state.editable_transcript = (
                transcript
            )


            st.session_state.audio_duration = (
                duration
            )


            st.session_state.whisper_time = (
                whisper_time
            )


            st.session_state.intelligence = (
                None
            )


            st.success(
                "✅ Meeting transcription completed successfully."
            )


        except Exception as error:


            st.error(
                "❌ The audio could not be transcribed."
            )


            st.warning(
                "Please confirm that the uploaded file "
                "contains audible speech and is not corrupted."
            )


            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


        finally:

            shutil.rmtree(
                temporary_directory,
                ignore_errors=True
            )


# ============================================================
# 21. NO AUDIO
# ============================================================

else:

    st.info(
        "Upload a meeting recording to begin."
    )


# ============================================================
# 22. STEP 2 — REVIEW TRANSCRIPT
# ============================================================

if st.session_state.transcript:


    st.divider()


    st.header(
        "2. Review Meeting Transcript"
    )


    st.caption(
        "You can correct any transcription errors "
        "before sending the transcript to Gemini."
    )


    edited_transcript = st.text_area(

        "Meeting Transcript",

        key="editable_transcript",

        height=300
    )


    transcript_words = len(
        edited_transcript.split()
    )


    transcript_col1, transcript_col2, transcript_col3 = (
        st.columns(3)
    )


    with transcript_col1:

        st.metric(
            "Words",
            transcript_words
        )


    with transcript_col2:

        st.metric(
            "Audio Duration",
            (
                f"{st.session_state.audio_duration:.1f} sec"
                if st.session_state.audio_duration
                else "N/A"
            )
        )


    with transcript_col3:

        st.metric(
            "Whisper Time",
            (
                f"{st.session_state.whisper_time:.2f} sec"
                if st.session_state.whisper_time
                is not None
                else "N/A"
            )
        )


    st.download_button(

        "⬇️ Download Transcript",

        data=edited_transcript,

        file_name=(
            "AIMIS_meeting_transcript.txt"
        ),

        mime="text/plain",

        use_container_width=True
    )


    # ========================================================
    # 23. GEMINI BUTTON
    # ========================================================

    if st.button(
        "🧠 Generate Meeting Intelligence",
        type="primary",
        use_container_width=True
    ):


        if not edited_transcript.strip():

            st.warning(
                "The transcript is empty."
            )


        else:

            try:


                with st.spinner(
                    "Gemini is analysing the meeting..."
                ):


                    intelligence, gemini_time = (
                        analyse_meeting(
                            edited_transcript
                        )
                    )


                st.session_state.intelligence = (
                    intelligence
                )


                st.session_state.gemini_time = (
                    gemini_time
                )


                st.success(
                    "✅ Meeting intelligence generated."
                )


            except Exception as error:

                show_gemini_error(
                    error
                )


# ============================================================
# 24. STEP 3 — MEETING INTELLIGENCE
# ============================================================

if (
    st.session_state.intelligence
    is not None
):


    intelligence = (
        st.session_state.intelligence
    )


    final_transcript = (
        st.session_state.editable_transcript
    )


    st.divider()


    st.header(
        "3. AI Meeting Intelligence Report"
    )


    st.subheader(
        f"📌 {intelligence.meeting_title}"
    )


    # ========================================================
    # 25. RESULT METRICS
    # ========================================================

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )


    with result_col1:

        st.metric(
            "Key Topics",
            len(
                intelligence.key_topics
            )
        )


    with result_col2:

        st.metric(
            "Confirmed Decisions",
            len(
                intelligence.decisions
            )
        )


    with result_col3:

        st.metric(
            "Action Items",
            len(
                intelligence.action_items
            )
        )


    with result_col4:

        st.metric(
            "Gemini Time",
            (
                f"{st.session_state.gemini_time:.2f} sec"
            )
        )


    # ========================================================
    # 26. RESULT TABS
    # ========================================================

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


    # ========================================================
    # SUMMARY
    # ========================================================

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
                    f"- {topic}"
                )

        else:

            st.info(
                "No key topics identified."
            )


    # ========================================================
    # DECISIONS
    # ========================================================

    with decisions_tab:


        st.subheader(
            "Confirmed Decisions"
        )


        if intelligence.decisions:


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
                    f"Model confidence: "
                    f"{decision.confidence:.0%}"
                )


                with st.expander(
                    "Supporting transcript evidence"
                ):

                    st.write(
                        decision.evidence
                    )


        else:

            st.info(
                "No confirmed decisions were identified."
            )


    # ========================================================
    # ACTION ITEMS
    # ========================================================

    with actions_tab:


        st.subheader(
            "Action Items"
        )


        if intelligence.action_items:


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
                            f"{item.confidence:.0%}"
                    }
                )


            st.dataframe(

                pd.DataFrame(
                    action_rows
                ),

                use_container_width=True,

                hide_index=True
            )


            st.subheader(
                "Supporting Evidence"
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


    # ========================================================
    # FOLLOW-UP
    # ========================================================

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
                "No unresolved issues identified."
            )


        st.divider()


        st.subheader(
            "Ambiguities"
        )


        if intelligence.ambiguities:

            for item in (
                intelligence.ambiguities
            ):

                st.markdown(
                    f"- {item}"
                )

        else:

            st.success(
                "No significant ambiguities identified."
            )


    # ========================================================
    # TECHNICAL
    # ========================================================

    with technical_tab:


        total_ai_time = (

            (
                st.session_state.whisper_time
                or 0
            )

            +

            (
                st.session_state.gemini_time
                or 0
            )
        )


        technical_data = pd.DataFrame(
            {

                "Component": [

                    "Speech Recognition",

                    "Whisper Device",

                    "Language Model",

                    "Audio Duration",

                    "Transcript Words",

                    "Whisper Processing Time",

                    "Gemini Processing Time",

                    "Total AI Processing Time"
                ],


                "Value": [

                    WHISPER_MODEL_NAME,

                    DEVICE,

                    GEMINI_MODEL,

                    (
                        f"{st.session_state.audio_duration:.2f} sec"
                    ),

                    len(
                        final_transcript.split()
                    ),

                    (
                        f"{st.session_state.whisper_time:.2f} sec"
                    ),

                    (
                        f"{st.session_state.gemini_time:.2f} sec"
                    ),

                    f"{total_ai_time:.2f} sec"
                ]
            }
        )


        st.dataframe(

            technical_data,

            use_container_width=True,

            hide_index=True
        )


    # ========================================================
    # 27. DOWNLOAD REPORT
    # ========================================================

    st.divider()


    report = create_report(

        intelligence,

        final_transcript,

        st.session_state.whisper_time,

        st.session_state.gemini_time,

        st.session_state.audio_duration
    )


    st.download_button(

        "📥 Download Complete Meeting Report",

        data=report,

        file_name=(
            "AIMIS_meeting_intelligence_report.txt"
        ),

        mime="text/plain",

        type="primary",

        use_container_width=True
    )


# ============================================================
# 28. FOOTER
# ============================================================

st.divider()


st.markdown(
    """
<div class="aimis-footer">
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
