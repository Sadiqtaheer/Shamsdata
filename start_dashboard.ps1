Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\app.py" --server.port 8501
