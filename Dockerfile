FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy pandas scikit-learn scipy matplotlib seaborn
RUN pip install --no-cache-dir prophet catboost xgboost lightgbm
RUN pip install --no-cache-dir tsfresh tslearn dtaidistance pingouin sktime dieboldmariano
RUN git config --global http.postBuffer 1048576000 && \
    git config --global http.version HTTP/1.1 && \
    pip install --no-cache-dir git+https://github.com/etna-team/etna.git@master
COPY powercons_etna.py /app/powercons_etna.py
CMD ["python", "powercons_etna.py"]
