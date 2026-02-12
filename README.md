# SNR GUI

SNR GUI is a Streamlit app for marker quality control using per-cell CSV files from multiple ROIs.

## Project Website

This repository includes a GitHub Pages site in `docs/`.

- Local preview: open `docs/index.html`
- Online URL pattern: `https://<your-user>.github.io/<your-repo>/`

## App Run

```bash
pip install streamlit pandas numpy matplotlib
streamlit run "# snr_gui.py"
```

Optional: rename `# snr_gui.py` to `snr_gui.py` for cleaner commands.

## SNR Labels

- `Pass`: SNR >= 3
- `Good`: SNR >= 10

## GitHub Pages Deployment

The workflow file `.github/workflows/deploy-pages.yml` is already configured.

1. Push to branch `main`.
2. In GitHub repo settings, set Pages source to `GitHub Actions`.
3. Wait for `Deploy GitHub Pages` workflow to finish.
