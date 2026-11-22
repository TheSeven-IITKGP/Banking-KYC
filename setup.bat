@echo off
REM One-step setup for Windows. Run from inside the kyc-rag folder:  setup.bat
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
pip install -r requirements.txt || goto :error
pip install -e . || goto :error
if not exist .env copy .env.example .env
python -m kycrag.cli download || goto :error
python -m kycrag.cli build || goto :error
python -m kycrag.cli inspect CB-KYC-2025
echo.
echo Done. Next: put your free Groq key (GROQ_API_KEY) in .env, then run:  streamlit run app\streamlit_app.py
goto :eof
:error
echo Setup stopped at the step above. If "download" failed, save the files manually into data\raw (see README).
exit /b 1
