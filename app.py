import streamlit as st
import pandas as pd
from pathlib import Path
import whisper
from google import genai
import os


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Meeting Intelligence System",
    layout="wide"
)


st.title("AI Meeting Intelligence System (AIMIS)")

st.write(
    "AI-powered meeting transcription, analysis and evaluation dashboard"
)


# =====================================================
# LOAD WHISPER MODEL
# =====================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model(
        "small.en"
    )


# =====================================================
# GEMINI CLIENT
# =====================================================

def get_gemini_client():

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


# =====================================================
# PART 1 - LIVE MEETING PROCESSING
# =====================================================


st.header("1. Process New Meeting")


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
        "Process Meeting",
        type="primary"
    ):


        # -----------------------------------------
        # SAVE AUDIO FILE
        # -----------------------------------------

        audio_path = "uploaded_meeting.mp3"


        with open(
            audio_path,
            "wb"
        ) as f:

            f.write(
                uploaded_audio.getbuffer()
            )


        # -----------------------------------------
        # WHISPER TRANSCRIPTION
        # -----------------------------------------


        with st.spinner(
            "Running Whisper transcription..."
        ):


            model = load_whisper_model()


            transcription = model.transcribe(
                audio_path
            )


            transcript = transcription["text"]



        st.success(
            "Transcription completed"
        )


        st.subheader(
            "Meeting Transcript"
        )


        st.text_area(
            "Transcript",
            transcript,
            height=250
        )



        # -----------------------------------------
        # GEMINI ANALYSIS
        # -----------------------------------------


        with st.spinner(
            "Generating meeting intelligence..."
        ):


            client = get_gemini_client()


            prompt = f"""

You are an AI meeting intelligence assistant.

Analyse the following meeting transcript.

Extract:

1. Meeting summary

2. Key topics

3. Decisions

4. Action items

5. Owners

6. Deadlines

7. Evidence quotes


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


        st.subheader(
            "AI Meeting Intelligence Report"
        )


        st.write(
            intelligence
        )



# =====================================================
# PART 2 - EXPERIMENTAL EVALUATION DASHBOARD
# =====================================================


st.divider()


st.header(
    "2. AIMIS Experimental Evaluation"
)



evaluation_files = [

"M01_generic_vs_structured_extraction_metrics.csv",

"asr_results.csv",

"prompt_repeatability_summary.csv"

]



for file in evaluation_files:


    file_path = Path(file)


    if file_path.exists():


        st.subheader(
            file
        )


        df = pd.read_csv(
            file_path
        )


        st.dataframe(
            df,
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
    "AI Meeting Intelligence System - MSc Project Prototype"
)
