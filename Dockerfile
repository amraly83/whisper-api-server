FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Change the exposed port to 8083
EXPOSE 8083

# Update the command to use port 8083
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8083"]