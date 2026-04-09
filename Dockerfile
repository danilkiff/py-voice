FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer — rebuilt only when lock file changes
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY *.py ./

EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]
