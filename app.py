import streamlit as st
import pandas as pd
from pathlib import Path
import whisper
from google import genai
import shutil
import subprocess
import os


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Meeting Intelligence System",
    page_icon="🎙️",
    layout="wide"
)


st.title("🎙️ AI Meeting Intelligence System (AIMIS)")

st.write(
    """
    An AI-based system for automatic meeting transcription,
    decision extraction, action-item identification,
    and evaluation.
    """
)


# =====================================================
# LOAD WHISPER MODEL
# =====================================================

@st.cache_resource
def load_whisper():

    model = whisper.load_model(
        "small.en"
    )

    return model



# =====================================================
# GEMINI CLIENT
# =====================================================

def gemini_client():

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )



# =====================================================
# SECTION 1
# AUDIO UPLOAD AND PROCESSING
# =====================================================

st.header("1. Upload and Process Meeting")


uploaded_audio = st.file_uploader(
    "Upload meeting audio file",
    type=[
        "mp3",
        "wav",
        "m4a"
    ]
)



if uploaded_audio:


    st.audio(
        uploaded_audio
    )


    if st.button(
        "🚀 Process Meeting"
    ):


        # ---------------------------------------------
        # Save uploaded file
        # ---------------------------------------------

        extension = uploaded_audio.name.split(".")[-1]


        audio_path = (
            f"meeting_audio.{extension}"
        )


        with open(
            audio_path,
            "wb"
        ) as f:

            f.write(
                uploaded_audio.getbuffer()
            )


        st.info(
            f"Audio saved: {audio_path}"
        )



        # ---------------------------------------------
        # Check FFmpeg
        # ---------------------------------------------

        if shutil.which("ffmpeg") is None:

            st.error(
                "FFmpeg is missing. Add packages.txt with ffmpeg."
            )

            st.stop()


        st.success(
            "FFmpeg detected"
        )



        # ---------------------------------------------
        # Whisper transcription
        # ---------------------------------------------


        with st.spinner(
            "Running Whisper transcription..."
        ):


            whisper_model = load_whisper()


            result = whisper_model.transcribe(
                audio_path
            )


            transcript = result["text"]



        st.success(
            "Transcription completed"
        )


        st.subheader(
            "📝 Meeting Transcript"
        )


        st.text_area(
            "Transcript",
            transcript,
            height=300
        )



        # ---------------------------------------------
        # Gemini extraction
        # ---------------------------------------------


        with st.spinner(
            "Generating meeting intelligence..."
        ):


            client = gemini_client()



            prompt = f"""

You are an AI meeting intelligence assistant.

Analyze this meeting transcript.

Extract:

1. Meeting summary

2. Key topics

3. Decisions

4. Action items

5. Responsible owners

6. Deadlines

7. Evidence quotes


Rules:

- Only include confirmed decisions.
- Do not convert suggestions into decisions.
- Include evidence from transcript.


Transcript:

{transcript}

"""



            response = client.models.generate_content(

                model="gemini-3.5-flash-lite",

                contents=prompt

            )


            intelligence = response.text



        st.success(
            "Meeting intelligence generated"
        )



        st.header(
            "🤖 AI Meeting Intelligence Report"
        )


        st.write(
            intelligence
        )



        # Save output

        with open(
            "latest_meeting_report.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                intelligence
            )



# =====================================================
# SECTION 2
# EXISTING EXPERIMENTAL RESULTS
# =====================================================


st.divider()


st.header(
    "2. AIMIS Experimental Evaluation Dashboard"
)



files = [

    "M01_generic_vs_structured_extraction_metrics.csv",

    "asr_results.csv",

    "prompt_repeatability_summary.csv"

]



for file in files:


    file_path = Path(file)


    if file_path.exists():


        st.subheader(
            file
        )


        data = pd.read_csv(
            file_path
        )


        st.dataframe(
            data,
            use_container_width=True
        )


    else:


        st.warning(
            f"{file} not found"
        )



# =====================================================
# FOOTER
# =====================================================


st.divider()


st.caption(
    "AI Meeting Intelligence System (AIMIS) - MSc Project Prototype"
)
