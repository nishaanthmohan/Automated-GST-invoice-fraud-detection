@echo off
echo ======================================================================
echo Starting AI-Driven GST Invoice Anomaly Detection & XAI Dashboard
echo BIP Project ID: 2026MIN500 - Bannari Amman Institute of Technology
echo ======================================================================
cd /d "%~dp0"
py -m streamlit run app.py
pause
