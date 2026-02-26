FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STARDIST_MODEL_DIR=/root/.keras/models/StarDist2D

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Preload StarDist model during build so runtime does not depend on network.
# Upstream zip extracts into a nested *_extracted folder that includes config.json.
RUN python - <<'PY'
import os
import urllib.request
import zipfile
from pathlib import Path
from stardist.models import StarDist2D

root = Path(os.environ.get('STARDIST_MODEL_DIR', '/root/.keras/models/StarDist2D'))
target = root / '2D_versatile_fluo'
extracted = target / '2D_versatile_fluo_extracted'
zip_path = target / '2D_versatile_fluo.zip'
target.mkdir(parents=True, exist_ok=True)
if not extracted.exists():
    url = 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip'
    print('Downloading StarDist model:', url)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extracted)

# Validate local loading path used by segmentation.py
m = StarDist2D(None, name=extracted.name, basedir=str(extracted.parent))
print('StarDist model ready at', m.logdir)
PY

COPY snr_gui.py /app/snr_gui.py
COPY snr_gui_ome.py /app/snr_gui_ome.py
COPY snr_gui_ome_mvp.py /app/snr_gui_ome_mvp.py
COPY segmentation.py /app/segmentation.py
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "snr_gui.py", "--server.address=0.0.0.0", "--server.port=8501"]
