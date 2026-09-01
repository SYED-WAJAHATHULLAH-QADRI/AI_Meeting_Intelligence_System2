# AI Meeting Intelligence System (AIMIS)

## Overview

AIMIS is an AI-based meeting analysis system that converts meeting
transcripts into structured intelligence.

## Features

- Automatic speech transcription
- Meeting summarization
- Decision extraction
- Action-item extraction
- Evidence-based evaluation

## Technologies

Speech Recognition:
OpenAI Whisper small.en

Language Model:
Google Gemini Flash

## Evaluation Metrics

- Word Error Rate (WER)
- Character Error Rate (CER)
- Precision
- Recall
- F1-score
- Hallucination rate
- Owner accuracy
- Deadline accuracy

## Repository Contents

AIMIS/
    
    app.py
    README.md
    requirements.txt
    
    results/
        M01_generic_vs_structured_extraction_metrics.csv
        asr_results.csv
        prompt_repeatability_summary.csv


## Installation

Install dependencies:

pip install -r requirements.txt


## Run Application

streamlit run app.py


## Research Purpose

This repository contains the experimental implementation
and evaluation results of an AI Meeting Intelligence System.