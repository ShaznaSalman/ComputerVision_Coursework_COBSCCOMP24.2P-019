# RetinaTrace AI: Gradio prototype for five-stage DR grading (research use only).
# Build: docker build -t retinatrace .
# Run:   docker run --rm -p 7860:7860 retinatrace   then open http://localhost:7860
# Memory: the app peaks at about 1.5 GB while it loads the models and analyses an image,
# so the host needs at least 2 GB of RAM.
FROM python:3.11-slim

# OpenCV needs libGL and GLib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    GRADIO_ANALYTICS_ENABLED=False \
    MPLCONFIGDIR=/tmp/matplotlib

# Hugging Face Spaces runs containers as user 1000; use the same user everywhere.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Code, the two weight files the app loads, the sample and retrieval images, the embeddings
# and the saved test metrics shown in the app.
COPY --chown=appuser app.py ./
COPY --chown=appuser core/ core/
COPY --chown=appuser app/ app/
COPY --chown=appuser checkpoints/final_model.weights.h5 checkpoints/unet_pseudomask.weights.h5 checkpoints/
COPY --chown=appuser embeddings.npz ./
COPY --chown=appuser report_images/test_loss_and_metrics.csv report_images/

USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
