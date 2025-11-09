"""
FastAPI application for DR Binary Classification
Deployed on AWS Lambda via Docker container
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from contextlib import asynccontextmanager
import io
from PIL import Image
import logging
import sys

from model_handler import DRModelHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global model handler
model_handler: Optional[DRModelHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global model_handler
    
    logger.info("=" * 60)
    logger.info("STARTING DR CLASSIFICATION API")
    logger.info("=" * 60)
    
    try:
        logger.info("Step 1: Testing imports...")
        import torch
        import timm
        import albumentations
        logger.info(f"  ✓ PyTorch {torch.__version__}")
        logger.info(f"  ✓ TIMM {timm.__version__}")
        logger.info(f"  ✓ Albumentations {albumentations.__version__}")
        
        logger.info("Step 2: Initializing model handler (S3-backed)...")
        model_handler = DRModelHandler(device="cpu")
        logger.info("  ✓ Model loaded successfully")
        
        logger.info("=" * 60)
        logger.info("API READY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("STARTUP FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    model_handler = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Binary classification API for DR screening (Negative vs Positive)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    interpretation: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DR Classification API is running",
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": model_handler is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_handler is not None else "unhealthy",
        "model_loaded": model_handler is not None,
        "device": "cpu"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR classification from uploaded fundus image
    """
    
    if model_handler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        logger.info(f"Processing image: {file.filename} ({image.size})")
        
        result = model_handler.predict(image)
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        await file.close()


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Predict DR classification for multiple images"""
    
    if model_handler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            result = model_handler.predict(image)
            result["filename"] = file.filename
            results.append(result)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            await file.close()
    
    return JSONResponse(content={"results": results})


# ============================================
# AWS LAMBDA HANDLER (IMPORTANT!)
# ============================================
from mangum import Mangum

# This is the Lambda handler
handler = Mangum(app)

# For local testing with uvicorn
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")