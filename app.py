import streamlit as st
import pandas as pd
from pathlib import Path


st.title("AI Meeting Intelligence System")


st.write(
    "Experimental dashboard for AIMIS evaluation results."
)


folder = Path("results")


files = [
    "M01_generic_vs_structured_extraction_metrics.csv",
    "asr_results.csv",
    "prompt_repeatability_summary.csv"
]


for f in files:

    path = folder / f

    if path.exists():

        st.subheader(f)

        df = pd.read_csv(path)

        st.dataframe(df)

    else:

        st.warning(f + " not found")