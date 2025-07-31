from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import torch
import logging
import os
import time
from typing import Optional, Dict, Any
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class QuestionRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    processing_time: float
    model_info: str

def get_api_key(request: Request) -> str:
    """Extract the API key from the x-api-key header."""
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    return api_key

def get_expected_api_key() -> str:
    """Return the expected API key from the environment variable."""
    key = os.environ.get("LLM_API_KEY", "demo-api-key-12345")
    if not key:
        logger.warning("LLM_API_KEY environment variable not set; requests will always fail authentication")
    return key

@lru_cache()
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer (cached)."""
    model_path = os.environ.get("MODEL_PATH", "models/fine_tuned")
    
    logger.info("Loading model from: %s", model_path)
    
    try:
        # Try to load fine-tuned model
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            # Create pipeline
            generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Successfully loaded fine-tuned model")
            return generator, "fine-tuned"
            
    except Exception as e:
        logger.warning("Failed to load fine-tuned model: %s", e)
    
    # Fallback to base model
    try:
        base_model = os.environ.get("BASE_MODEL", "google/flan-t5-base")
        logger.info("Loading base model: %s", base_model)
        
        generator = pipeline(
            "text2text-generation",
            model=base_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Successfully loaded base model")
        return generator, "base"
        
    except Exception as e:
        logger.error("Failed to load any model: %s", e)
        raise HTTPException(status_code=500, detail="Model loading failed")

def verify_api_key(provided_key: str = Depends(get_api_key)) -> str:
    """Verify the provided API key against the expected key."""
    expected_key = get_expected_api_key()
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return provided_key

# Initialize FastAPI app
app = FastAPI(
    title="EV Charging QA API",
    description="Fine-tuned language model API for electric vehicle charging questions",
    version="1.0.0"
)

@app.post("/query", response_model=QueryResponse)
def query_model(
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    """Generate an answer to a question about electric vehicle charging."""
    start_time = time.time()
    
    try:
        generator, model_type = load_model_and_tokenizer()
        
        # Format the input for the model
        input_text = f"Question: {request.question}"
        
        # Generate response
        result = generator(
            input_text,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        answer = result[0]['generated_text'].strip()
        
        # Clean up the response if needed
        if answer.startswith("Answer: "):
            answer = answer[8:]
        
        processing_time = time.time() - start_time
        
        logger.info("Generated response for question: %s (%.3fs)", request.question[:50], processing_time)
        
        return QueryResponse(
            answer=answer,
            processing_time=processing_time,
            model_info=f"{model_type}_model"
        )
        
    except Exception as e:
        logger.error("Error generating response: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate response")

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        generator, model_type = load_model_and_tokenizer()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_type,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/model-info")
def model_info(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        generator, model_type = load_model_and_tokenizer()
        model_path = os.environ.get("MODEL_PATH", "models/fine_tuned")
        
        return {
            "model_type": model_type,
            "model_path": model_path,
            "base_model": os.environ.get("BASE_MODEL", "google/flan-t5-base"),
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")

@app.get("/metrics")
def get_metrics(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Get basic metrics about the service."""
    return {
        "model_loaded": True,
        "uptime_seconds": time.time(),
        "cuda_available": torch.cuda.is_available(),
        "memory_usage": "N/A"  # Could add actual memory monitoring here
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP exception: %s - %s", exc.status_code, exc.detail)
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unexpected error: %s", exc)
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "../config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        host = config.get('host', '0.0.0.0')
        port = config.get('port', 8000)
    else:
        host = '0.0.0.0'
        port = 8000
    
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port) 