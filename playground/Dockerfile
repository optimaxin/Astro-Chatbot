# One image, one service: the React build is served by the FastAPI backend.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    PLAYGROUND_DB_PATH=/var/data/playground.db \
    PLAYGROUND_STATIC_DIR=/app/static
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/config ./config
COPY backend/scripts ./scripts
COPY --from=web /web/dist ./static

EXPOSE 10000
# One worker on purpose: SQLite, the in-memory rate limiter and the ai-core
# session tokens all assume a single process. Plenty for 1–10 testers.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --workers 1"]
