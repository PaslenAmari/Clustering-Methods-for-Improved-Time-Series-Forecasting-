FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy pandas scikit-learn scipy matplotlib seaborn
RUN pip install prophet catboost xgboost lightgbm
RUN pip install tsfresh tslearn dtaidistance pingouin sktime dieboldmariano
RUN pip install git+https://github.com/etna-team/etna.git@master
COPY powercons_etna.py /app/powercons_etna.py
CMD ["python", "powercons_etna.py"]
