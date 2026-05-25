# WB Data Catalog — multi-stage: Vite frontend + FastAPI backend
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
COPY --from=frontend-build /build/dist /app/static
ENV FRONTEND_DIST=/app/static
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["./start.sh"]
