FROM python:3.6-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Install your package
RUN python setup.py install

EXPOSE 8081

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8081", "volume_estimation_server:app"]
