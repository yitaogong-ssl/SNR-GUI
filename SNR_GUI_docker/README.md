# SNR GUI

SNR GUI is a Streamlit app for marker quality control using per-cell CSV files from multiple ROIs.

## Local Run

```bash
pip install streamlit pandas numpy matplotlib
streamlit run snr_gui.py
```

## Internal SPU Deployment (PIPEX Linux, Docker only)

```bash
docker compose up -d --build
```

Then open:

- On PIPEX host: `http://localhost:8501`
- From SciLifeLab internal network: `http://pipex-linux:8501`

Stop:

```bash
docker compose down
```

## SNR Labels

- `Pass`: SNR >= 3
- `Good`: SNR >= 10
