FROM python:3.12-slim

WORKDIR /app
COPY server.py .

ENV CHATBOT_HOST=0.0.0.0
ENV CHATBOT_PORT=8000

EXPOSE 8000
CMD ["python", "server.py"]
