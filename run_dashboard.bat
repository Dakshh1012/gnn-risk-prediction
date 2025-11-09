@echo off
echo Installing required packages for Streamlit app...
pip install streamlit==1.25.0 plotly==5.15.0 pillow==10.0.0

echo Starting Supply Chain Risk Intelligence Dashboard...
streamlit run streamlit_app.py

pause