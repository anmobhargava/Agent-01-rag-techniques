# The chat app in a container. Works the same on Hugging Face Spaces (Docker SDK), Cloud Run, Fly, or your laptop.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/docs ./data/docs
RUN chown -R app:app /home/app
USER app

ENV PYTHONPATH=/home/app/src \
    HAYSTACK_TELEMETRY_ENABLED=False \
    GRADIO_ANALYTICS_ENABLED=False \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PORT=7860

# Download the embedding model while the image is built, so a visitor never waits through a download on first load.
# (If this step cannot reach the internet the build fails loudly, which is what I want.)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 7860
CMD ["python", "-m", "rag_agent.app"]
