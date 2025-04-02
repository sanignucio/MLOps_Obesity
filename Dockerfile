FROM python:3.11.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

# Instalar dependencias de catboost y pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/list/*

COPY . . 

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080" ]
