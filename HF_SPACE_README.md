---
title: RetinaTrace AI
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Research prototype for five-stage diabetic retinopathy grading
---

# RetinaTrace AI (Hugging Face Space)

Research prototype for five-stage diabetic retinopathy grading from fundus photographs
(EfficientNetB3, Grad-CAM, similar-case retrieval and a safety gate). It is not a medical
device and must not be used for clinical decisions.

**Status: alternative setup, not used.** Hugging Face Spaces requires a paid plan for Docker Spaces. The live deployment is on Render: https://retinatrace-cv-coursework-cobsccomp24-2p.onrender.com/

The Space is built from the repository's `Dockerfile`. The YAML header above tells Hugging Face
to use the Docker SDK and to route traffic to port 7860.

## Steps to deploy

1. Create a new Space on Hugging Face and choose **Docker** as the SDK (blank template).
   Pick hardware with at least 2 GB of RAM; the app peaks at about 1.6 GB.
2. Clone the Space and copy in the files the Dockerfile uses: `Dockerfile`, `.dockerignore`,
   `requirements-deploy.txt`, `app.py`, `core/`, `app/`, `embeddings.npz`,
   `checkpoints/final_model.weights.h5`, `checkpoints/unet_pseudomask.weights.h5` and
   `report_images/test_loss_and_metrics.csv`.
3. Rename this file to `README.md` in the Space, so the YAML header is at the top of the Space's README.
4. Files over 10 MB must go through Git LFS (both weight files are larger: 45.7 MB and 23.4 MB):
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git add .
   git commit -m "Add RetinaTrace app"
   git push
   ```
5. The Space builds the image and starts `python app.py`. The first start takes a few minutes
   while the models load. Check the build and container logs on the Space page if it does not start.

The app creates no public share link (`share=False`) and keeps uploads only in temporary files.
