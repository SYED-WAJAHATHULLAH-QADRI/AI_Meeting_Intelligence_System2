import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import json
import whisper
from google import genai


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Meeting Intelligence System",
    layout="wide"
)


st.title(
    "AI Meeting Intelligence System (AIMIS)"
)

st.write(
    "AI-powered meeting transcription and intelligence extraction"
)


# =====================================================
# PART 1: USER MEETING PROCESSING APPLICATION
# =====================================================

st.header(
    "1. Process New Meeting"
)


uploaded_audio = st.file_uploader(
    "Upload meeting audio",
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
        "Process Meeting"
    ):


        with st.spinner(
            "Transcribing audio using Whisper..."
        ):


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            ) as temp_audio:


                temp_audio.write(
                    uploaded_audio.read()
                )

                audio_path = temp_audio.name



            # Whisper

            model = whisper.load_model(
                "small.en"
            )


            transcription = model.transcribe(
                audio_path
            )


            transcript = transcription["text"]



        st.success(
            "Audio transcription completed"
        )


        st.subheader(
            "Transcript"
        )

        st.write(
            transcript
        )



        # ==============================
        # GEMINI PROCESSING
        # ==============================


        st.spinner(
            "Generating meeting intelligence..."
        )


        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )


        prompt=f"""

You are an AI meeting intelligence assistant.

Analyze the transcript below.

Extract:

1. Meeting summary
2. Key decisions
3. Action items
4. Responsible persons
5. Deadlines
6. Evidence quotes


Transcript:

{transcript}

"""


        response = client.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=prompt

        )


        result=response.text



        st.success(
            "Meeting intelligence generated"
        )


        st.subheader(
            "Meeting Intelligence Report"
        )


        st.write(
            result
        )



# =====================================================
# PART 2: EVALUATION DASHBOARD
# =====================================================


st.divider()

st.header(
    "2. AIMIS Evaluation Dashboard"
)



files=[

"M01_generic_vs_structured_extraction_metrics.csv",

"asr_results.csv",

"prompt_repeatability_summary.csv"

]



for file in files:


    path=Path(file)


    if path.exists():

        st.subheader(
            file
        )


        df=pd.read_csv(
            path
        )


        st.dataframe(
            df,
            use_container_width=True
        )


    else:


        st.warning(
            f"{file} not found"
        )
