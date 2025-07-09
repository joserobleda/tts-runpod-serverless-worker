import io
import os
import argparse
import base64
import numpy as np
# runpod utils
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object
from runpod.serverless.utils import rp_download, rp_cleanup
# predictor
import predict
from rp_schema import INPUT_SCHEMA
# utils
from scipy.io.wavfile import write
# direct S3 upload
import boto3
from botocore.exceptions import ClientError


# Model params
model_dir = os.getenv("WORKER_MODEL_DIR", "/model")


def upload_audio(wav, sample_rate, key):
    """ Uploads audio to Cloudflare R2 bucket if it is available, otherwise returns base64 encoded audio. """
    # Ensure wav is numpy array and in correct format
    if hasattr(wav, 'cpu'):  # If it's a torch tensor
        wav = wav.cpu().numpy()
    
    # Ensure correct shape and dtype
    if wav.ndim > 1:
        wav = wav.squeeze()  # Remove extra dimensions
    wav = np.clip(wav, -1.0, 1.0)  # Clip to valid range
    wav = (wav * 32767).astype(np.int16)  # Convert to 16-bit integers
    
    # Convert wav to bytes
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, wav)
    wav_bytes = wav_io.getvalue()

    # Upload to Cloudflare R2 (S3-compatible) - Direct upload to avoid date folders
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        try:
            # Parse endpoint and bucket from URL
            endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
            
            # If the URL contains a bucket name, extract it
            if '/' in endpoint_url.split('://', 1)[1]:
                # URL format: https://account-id.r2.cloudflarestorage.com/bucket-name
                base_url, bucket_name = endpoint_url.rsplit('/', 1)
                actual_endpoint = base_url
            else:
                # URL format: https://account-id.r2.cloudflarestorage.com
                actual_endpoint = endpoint_url
                bucket_name = os.environ.get('BUCKET_NAME', 'tts')
            
            # Create S3 client for direct upload
            s3_client = boto3.client(
                's3',
                endpoint_url=actual_endpoint,
                aws_access_key_id=os.environ.get('BUCKET_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('BUCKET_SECRET_ACCESS_KEY'),
                region_name='auto'  # Cloudflare R2 uses 'auto' as region
            )
            
            # Upload directly to root of bucket (no date folders)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,  # File will go directly to root with this key
                Body=wav_bytes,
                ContentType='audio/wav'
            )
            
            # Return the public URL
            return f"{actual_endpoint}/{bucket_name}/{key}"
            
        except Exception as e:
            print(f"Direct S3 upload failed: {e}")
            print("Falling back to RunPod upload function...")
            # Fallback to original RunPod function
            return upload_in_memory_object(
                key,
                wav_bytes,
                bucket_creds = {
                    "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                    "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                    "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
                }
            )
    
    # Base64 encode for direct return
    return base64.b64encode(wav_bytes).decode('utf-8')


def run(job):
    try:
        job_input = job['input']
        print(f"Processing job {job['id']} with input: {job_input}")

        # Input validation
        validated_input = validate(job_input, INPUT_SCHEMA)

        if 'errors' in validated_input:
            print(f"Validation errors: {validated_input['errors']}")
            return {"error": validated_input['errors']}
        validated_input = validated_input['validated_input']

        # Download input objects
        print(f"Downloading voice files: {list(validated_input['voice'].keys())}")
        for k, v in validated_input["voice"].items():
            validated_input["voice"][k] = rp_download.download_files_from_urls(
                job['id'],
                [v]
            )[0]  # Take the first downloaded file

        print(f"Processing text segments: {len(validated_input['text'])}")
        
        # Inference text-to-audio with optimal quality settings
        wave, sr = MODEL.predict(
            language=validated_input["language"],
            speaker_wav=validated_input["voice"],
            text=validated_input["text"],
            # Basic parameters
            gpt_cond_len=validated_input.get("gpt_cond_len", 30),  # Higher for better quality
            max_ref_len=validated_input.get("max_ref_len", 60),    # Higher for longer reference
            speed=validated_input.get("speed", 1.0),
            enhance_audio=validated_input.get("enhance_audio", True),
            # Advanced quality parameters for maximum quality
            temperature=validated_input.get("temperature", 0.3),           # Balanced creativity/stability
            length_penalty=validated_input.get("length_penalty", 1.0),     # No length bias
            repetition_penalty=validated_input.get("repetition_penalty", 5.0),  # Prevent repetition
            top_k=validated_input.get("top_k", 40),                        # Good diversity
            top_p=validated_input.get("top_p", 0.7),                       # Nucleus sampling  
            num_gpt_outputs=validated_input.get("num_gpt_outputs", 1),     # Single output
            gpt_cond_chunk_len=validated_input.get("gpt_cond_chunk_len", 4), # Stable chunking
            sound_norm_refs=validated_input.get("sound_norm_refs", False), # No normalization
            enable_text_splitting=validated_input.get("enable_text_splitting", True)  # Better long text
        )

        if wave is None:
            return {"error": "Failed to generate audio"}

        print(f"Generated audio: shape={wave.shape if hasattr(wave, 'shape') else len(wave)}, sr={sr}")

        # Upload output object
        audio_return = upload_audio(wave, sr, f"{job['id']}.wav")
        job_output = {
            # "audio": audio_return
        }

        # Remove downloaded input objects
        rp_cleanup.clean(['input_objects'])

        print(f"Job {job['id']} completed successfully")
        return job_output

    except Exception as e:
        print(f"Error processing job {job.get('id', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Internal error: {str(e)}"}


if __name__ == "__main__":
    MODEL = predict.Predictor(model_dir=model_dir)
    MODEL.setup()

    runpod.serverless.start({"handler": run})
