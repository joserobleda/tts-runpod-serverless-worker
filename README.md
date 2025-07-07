# TTS RunPod Serverless Worker

This is a serverless worker for text-to-speech generation using RunPod infrastructure with Cloudflare R2 storage.

## Configuration

### Cloudflare R2 Setup

To upload generated audio files to Cloudflare R2, set the following environment variables:

```bash
# Cloudflare R2 Configuration
BUCKET_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
BUCKET_ACCESS_KEY_ID=<your-r2-access-key-id>
BUCKET_SECRET_ACCESS_KEY=<your-r2-secret-access-key>
```

### How to get Cloudflare R2 credentials:

1. **Account ID**: Found in your Cloudflare dashboard sidebar
2. **R2 Access Key ID & Secret**: 
   - Go to Cloudflare Dashboard → R2 Object Storage → Manage R2 API tokens
   - Create a new API token with R2 read/write permissions
   - Copy the Access Key ID and Secret Access Key

### Environment Variables

- `BUCKET_ENDPOINT_URL`: Your R2 endpoint URL in the format `https://<account-id>.r2.cloudflarestorage.com`
- `BUCKET_ACCESS_KEY_ID`: Your R2 access key ID
- `BUCKET_SECRET_ACCESS_KEY`: Your R2 secret access key
- `WORKER_MODEL_DIR`: Directory containing the TTS model (default: `/model`)

#### Example Configuration:
```bash
# Example environment variables for Cloudflare R2
export BUCKET_ENDPOINT_URL=https://abc123def456.r2.cloudflarestorage.com
export BUCKET_ACCESS_KEY_ID=1234567890abcdef1234567890abcdef
export BUCKET_SECRET_ACCESS_KEY=abcdef1234567890abcdef1234567890abcdef12
export WORKER_MODEL_DIR=/model
```

### Fallback Behavior

If R2 credentials are not configured, the worker will return audio files as base64-encoded strings instead of uploading to R2.

## Usage

The worker accepts text-to-speech generation requests and returns either:
- A URL to the uploaded audio file in R2 (if configured)
- Base64-encoded audio data (fallback)

## Model Configuration

The worker uses advanced quality parameters for optimal audio generation:
- Temperature: 0.7 (balanced creativity/stability)
- Repetition penalty: 5.0 (prevents repetition)
- Top-k: 50, Top-p: 0.8 (good diversity with nucleus sampling)
- Enhanced audio processing enabled by default

## RunPod Endpoint

This repository contains the worker for the xTTSv2 AI Endpoints.

## Docker Image

```bash
docker build .
```
 or

 ```bash
 docker pull devbes/tts-runpod-serverless-worker:latest
 ```

## Continuous Deployment
This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional SD models to be built and pushed.

## API

```json
{
  "input": {
      "language": <language:str>,
      "voice": {
          "speaker_0": "url"
          "speaker_1": "url"
          },
      "text": [
          ["speaker_0", "text"],
          ["speaker_1", "text"],
          ...
          ["speaker_1", "text"],
      ],
      "gpt_cond_len": <gpt_cond_len:int>,
      "max_ref_len": <max_ref_len:int>,
      "speed": <speed:float>
      "enhance_audio": <enhance_audio:bool>
  }
}
```
