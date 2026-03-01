from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import boto3
import json
import pandas as pd
from io import StringIO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "sentiment-analysis-endpoint")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

logger.info(f"API initialized using endpoint: {ENDPOINT_NAME} in region: {AWS_REGION}")

class TweetRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    tweets: List[str]

class SentimentResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    sentiment_emoji: str
    interpretation: str

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }
    return emojis.get(sentiment, '❓')

def get_sentiment_color(sentiment: str) -> str:
    """Get color class for sentiment"""
    colors = {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative'
    }
    return colors.get(sentiment, 'neutral')

def get_interpretation(sentiment: str, confidence: float) -> str:
    """Get interpretation of sentiment"""
    if confidence > 0.8:
        strength = "very strong"
    elif confidence > 0.6:
        strength = "strong"
    else:
        strength = "moderate"
    
    descriptions = {
        'positive': f"This text expresses {strength} positive sentiment. The message is upbeat and favorable.",
        'neutral': f"This text has {strength} neutral sentiment. The message is balanced or factual.",
        'negative': f"This text expresses {strength} negative sentiment. The message conveys dissatisfaction or criticism."
    }
    
    return descriptions.get(sentiment, "Unable to determine sentiment")

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "model": "DistilBERT for Sentiment Classification",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_csv": "/predict/csv",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        return {
            "status": "healthy",
            "endpoint": ENDPOINT_NAME,
            "region": AWS_REGION,
            "model": "DistilBERT Sentiment Classifier"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/predict", response_model=SentimentResponse)
async def predict_single(tweet: TweetRequest):
    """Predict sentiment for a single text"""
    try:
        runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        payload = json.dumps({'text': tweet.text})
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        
        sentiment = result['prediction']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        return SentimentResponse(
            prediction=sentiment,
            confidence=confidence,
            probabilities=probabilities,
            sentiment_emoji=get_sentiment_emoji(sentiment),
            interpretation=get_interpretation(sentiment, confidence)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """Predict sentiment for multiple texts"""
    try:
        runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        results = []
        for idx, text in enumerate(request.tweets):
            payload = json.dumps({'text': text})
            
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            
            sentiment = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            results.append({
                "id": idx + 1,
                "text": text,
                "prediction": sentiment,
                "confidence": confidence,
                "probabilities": probabilities,
                "sentiment_emoji": get_sentiment_emoji(sentiment),
                "sentiment_color": get_sentiment_color(sentiment),
                "interpretation": get_interpretation(sentiment, confidence)
            })
        
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict sentiment from CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Look for common column names
        text_column = None
        for col in ['text', 'tweet', 'message', 'content', 'review', 'comment']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            if len(df.columns) == 1:
                text_column = df.columns[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find text column. Available columns: {list(df.columns)}"
                )
        
        runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        results = []
        for idx, row in df.iterrows():
            text = str(row[text_column])
            
            payload = json.dumps({'text': text})
            
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            
            sentiment = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            results.append({
                "row": int(idx) + 1,
                "text": text,
                "prediction": sentiment,
                "confidence": confidence,
                "probabilities": probabilities,
                "sentiment_emoji": get_sentiment_emoji(sentiment),
                "sentiment_color": get_sentiment_color(sentiment),
                "interpretation": get_interpretation(sentiment, confidence)
            })
        
        return {
            "total_texts": len(results),
            "predictions": results,
            "summary": {
                "positive": sum(1 for r in results if r['prediction'] == 'positive'),
                "neutral": sum(1 for r in results if r['prediction'] == 'neutral'),
                "negative": sum(1 for r in results if r['prediction'] == 'negative'),
                "average_confidence": sum(r['confidence'] for r in results) / len(results) if results else 0
            }
        }
    
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
