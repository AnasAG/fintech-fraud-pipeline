# Deploying to GCP Cloud Run

This guide deploys the FastAPI scoring service to Cloud Run.
Cloud Run is a managed serverless container platform — you push a container image,
Cloud Run handles scaling, HTTPS, and load balancing.

**Cost:** Cloud Run has a generous free tier (2M requests/month). Running this
portfolio project at low traffic costs ~$0/month.

---

## Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed
- Project has been built locally (`make train` run successfully)

---

## Step 1: Set up GCP project

```bash
# Create a new project (or use existing)
gcloud projects create fintech-fraud-pipeline --name="Fintech Fraud Pipeline"
gcloud config set project fintech-fraud-pipeline

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

---

## Step 2: Create an Artifact Registry repository

```bash
gcloud artifacts repositories create fraud-pipeline \
  --repository-format=docker \
  --location=us-central1 \
  --description="Fraud detection pipeline images"
```

---

## Step 3: Build and push the API image

```bash
# Configure Docker to use gcloud for authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build the image
docker build -f docker/Dockerfile.api \
  -t us-central1-docker.pkg.dev/fintech-fraud-pipeline/fraud-pipeline/api:v1 .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/fintech-fraud-pipeline/fraud-pipeline/api:v1
```

---

## Step 4: Store the trained model in GCS

Cloud Run containers are stateless — the model can't live in the container image
(it changes every time you retrain). Store it in Cloud Storage and mount it at startup.

```bash
# Create a bucket
gcloud storage buckets create gs://fintech-fraud-models --location=us-central1

# Upload your trained champion model
gcloud storage cp -r models/champion/ gs://fintech-fraud-models/champion/
```

---

## Step 5: Deploy to Cloud Run

```bash
gcloud run deploy fraud-api \
  --image us-central1-docker.pkg.dev/fintech-fraud-pipeline/fraud-pipeline/api:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars "MODEL_PATH=/app/models/champion,THRESHOLD_APPROVE=0.3,THRESHOLD_DECLINE=0.7" \
  --set-secrets "POSTGRES_PASSWORD=postgres-password:latest"
```

**Note on memory:** LightGBM models can be 50–200MB. With the feature pipeline,
you need at least 1Gi RAM. Set 2Gi to be safe.

---

## Step 6: Verify deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe fraud-api --region us-central1 --format 'value(status.url)')
echo $SERVICE_URL

# Health check
curl $SERVICE_URL/health

# Test prediction
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 149.50, "ProductCD": "W", "card1": 9500}'
```

---

## Step 7: Set up Cloud SQL (PostgreSQL) for production

For the monitoring dashboard to work in production, you need a managed PostgreSQL instance.

```bash
# Create Cloud SQL instance
gcloud sql instances create fraud-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database and user
gcloud sql databases create fraud_db --instance=fraud-postgres
gcloud sql users create fraud_user --instance=fraud-postgres --password=<your-password>

# Store password in Secret Manager
echo -n "<your-password>" | gcloud secrets create postgres-password --data-file=-
```

---

## CI/CD with Cloud Build (optional next step)

For automated deploys on every push to main:

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.api', '-t', '$_IMAGE_TAG', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '$_IMAGE_TAG']

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - gcloud
      - run
      - deploy
      - fraud-api
      - --image=$_IMAGE_TAG
      - --region=us-central1
      - --platform=managed

substitutions:
  _IMAGE_TAG: 'us-central1-docker.pkg.dev/fintech-fraud-pipeline/fraud-pipeline/api:$COMMIT_SHA'
```

---

## Cost estimate

| Resource | Tier | Est. monthly cost |
|---|---|---|
| Cloud Run | 2M req/month free | $0 |
| Cloud SQL (db-f1-micro) | ~730 hours/month | ~$7 |
| Cloud Storage (model files) | < 1GB | < $0.03 |
| Artifact Registry | < 1GB | < $0.10 |
| **Total** | | **~$7/month** |

For a portfolio project with low traffic, Cloud Run alone (without Cloud SQL) is effectively free.
