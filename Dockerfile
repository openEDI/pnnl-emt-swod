FROM python:3.10.6-slim-bullseye
RUN apt-get update && apt-get install -y git ssh && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
EXPOSE 5684/tcp
CMD ["python", "-m", "pnnl_emt_swod.server"]
