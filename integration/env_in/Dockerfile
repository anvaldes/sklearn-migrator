FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r integration/env_in/requirements.txt \
 && pip install --no-cache-dir -e .

CMD ["bash"]
