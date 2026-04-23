FROM --platform=linux/amd64 python:3.10-slim-bullseye

WORKDIR /app

RUN uname -m

ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "setuptools<70.0.0" pip wheel Cython

# ============================================================================
# MAIN ENVIRONMENT
# ============================================================================

RUN pip install --no-cache-dir --prefer-binary \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.14.1 \
    matplotlib==3.8.3 \
    seaborn==0.13.2 \
    openpyxl==3.1.2

RUN pip install --no-cache-dir --prefer-binary \
    prophet==1.1.5 \
    catboost==1.2.3

RUN pip install --no-cache-dir --prefer-binary \
    xgboost==2.0.3 \
    lightgbm==4.3.0

RUN pip install --no-cache-dir --prefer-binary \
    tsfresh==0.21.0 \
    tslearn==0.6.3 \
    dtaidistance==2.3.13 \
    pingouin==0.5.5 \
    dieboldmariano==1.1.0

RUN pip install --no-cache-dir --prefer-binary \
    "https://github.com/etna-team/etna/archive/refs/tags/2.10.0.tar.gz#egg=etna"

# ============================================================================
# FEDOT SUB-ENVIRONMENT (/opt/fedot_env)
# ============================================================================

COPY requirements_fedot.txt .
RUN python -m venv /opt/fedot_env && \
    /opt/fedot_env/bin/pip install --upgrade pip "setuptools<70.0.0" wheel Cython setuptools-scm && \
    /opt/fedot_env/bin/pip install --no-cache-dir --prefer-binary \
        "numpy==1.26.4" \
        "scipy==1.12.0" \
        "statsmodels==0.14.0" \
        "SALib==1.4.7" \
        "six==1.16.0" \
        "urllib3<2.0.0" \
        "certifi==2024.2.2" \
        "idna==3.6" \
        "pyaml==23.12.0" && \
    /opt/fedot_env/bin/pip install --no-cache-dir --prefer-binary --no-build-isolation fedot==0.7.5

# ============================================================================
# SKTIME SUB-ENVIRONMENT (/opt/sktime_env)
# ============================================================================

COPY requirements_sktime.txt .
RUN python -m venv /opt/sktime_env && \
    /opt/sktime_env/bin/pip install --upgrade pip "setuptools<70.0.0" wheel Cython \
        "numpy==1.24.3" "statsmodels==0.14.0" && \
    /opt/sktime_env/bin/pip install --no-cache-dir --prefer-binary --no-build-isolation \
        -r requirements_sktime.txt

RUN /opt/sktime_env/bin/python -c "from sktime.clustering.k_means import TimeSeriesKMeans; print('SKTIME IS FULLY OPERATIONAL')"

ENV SKTIME_VENV_PATH=/opt/sktime_env/bin/python
ENV FEDOT_VENV_PATH=/opt/fedot_env/bin/python

RUN pip install --no-cache-dir "setuptools<70.0.0"

COPY . /app/

CMD ["python", "powercons_etna.py"]