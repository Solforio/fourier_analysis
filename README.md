# Fourier Analysis Studio
## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```


## V4 notes

- Explicit **Run Fourier analysis** button to avoid re-running expensive analysis on every interaction.
- Safer error metrics and grouped error tables for empty or fully-masked series.
- Extra regression tests for empty-array and all-NaN edge cases.
