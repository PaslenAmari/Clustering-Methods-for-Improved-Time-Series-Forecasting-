FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app

# Diagnostic: Verify we are indeed running in amd64 emulation
RUN uname -m

# Increase pip resilience against slow network connections
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Pin setuptools because newer versions remove pkg_resources, which some legacy tools may need.
RUN pip install --upgrade "setuptools<70.0.0" pip wheel Cython

# ============================================================================
# MAIN ENVIRONMENT (ETNA, Fedot, TSFresh, DTW)
# ============================================================================

# 1. Base Scientific Stack (Strong Pinning to avoid numpy 2.x conflicts)
RUN pip install --no-cache-dir --prefer-binary \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.14.1 \
    matplotlib==3.8.3 \
    seaborn==0.13.2

# 2. Heavy ML Backends (Split to isolate failures and reduce memory pressure)
RUN pip install --no-cache-dir --prefer-binary \
    prophet==1.1.5 \
    catboost==1.2.3

RUN pip install --no-cache-dir --prefer-binary \
    xgboost==2.0.3 \
    lightgbm==4.3.0

# 3. Time Series & Statistical Utilities
RUN pip install --no-cache-dir --prefer-binary \
    tsfresh==0.21.0 \
    tslearn==0.6.3 \
    dtaidistance==2.3.13 \
    pingouin==0.5.5 \
    dieboldmariano==1.1.0

# 3. Fedot Sub-Environment
COPY requirements_fedot.txt .
RUN python -m venv /opt/fedot_env && \
    /opt/fedot_env/bin/pip install --upgrade pip "setuptools<70.0.0" wheel Cython setuptools-scm && \
    /opt/fedot_env/bin/pip install --no-cache-dir --prefer-binary \
    "numpy==1.26.4" \
    "statsmodels==0.14.0" \
    "SALib==1.4.7" \
    "six==1.16.0" \
    "urllib3<2.0.0" \
    "certifi==2024.2.2" \
    "idna==3.6" \
    "pyaml==23.12.0"

RUN /opt/fedot_env/bin/pip install --no-cache-dir --prefer-binary --no-build-isolation fedot==0.7.5
RUN /opt/fedot_env/bin/pip install --no-cache-dir --prefer-binary --no-build-isolation fedot_ind==0.4.2

# 4. ETNA Git Install
RUN git config --global http.postBuffer 1048576000 && \
    git config --global http.version HTTP/1.1 && \
    pip install --no-cache-dir --prefer-binary git+https://github.com/etna-team/etna.git@master

# 5. sktime Sub-Environment
COPY requirements_sktime.txt .
RUN python -m venv /opt/sktime_env && \
    /opt/sktime_env/bin/pip install --upgrade pip "setuptools<70.0.0" wheel Cython "numpy==1.24.3" "statsmodels==0.14.0" && \
    /opt/sktime_env/bin/pip install --no-cache-dir --prefer-binary --no-build-isolation -r requirements_sktime.txt

# Environment Finalization
ENV SKTIME_VENV_PATH=/opt/sktime_env/bin/python
ENV FEDOT_VENV_PATH=/opt/fedot_env/bin/python

# Ensure the main environment also has the compatible setuptools
RUN pip install --no-cache-dir "setuptools<70.0.0"

# Copy the entire context
COPY . /app/
CMD ["python", "powercons_etna.py"]
