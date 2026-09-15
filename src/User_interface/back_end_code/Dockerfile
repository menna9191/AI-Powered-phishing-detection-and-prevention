
FROM python:3.11-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install torch==2.2.2 --default-timeout=1000

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["python", "main.py"]