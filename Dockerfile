FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "test.py"]