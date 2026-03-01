# 💬 Sentiment Analysis Web Application

Modern, professional web interface for sentiment analysis using DistilBERT and AWS SageMaker.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)

## 🌟 Features

### 💭 Single Text Analysis
- Interactive form for instant sentiment analysis
- Real-time prediction results with confidence scores
- Visual sentiment indicators with emojis
- Detailed probability distribution for each sentiment class
- Example texts for quick testing

### 📊 Batch/CSV Processing
- Upload CSV files with multiple texts
- Bulk prediction processing
- Visual summary statistics dashboard
- Interactive results table
- Downloadable results in CSV format
- Sample CSV template included

### 🔌 RESTful API
- `/predict` - Single text prediction
- `/predict/batch` - Multiple text predictions
- `/predict/csv` - CSV file upload
- `/health` - Health check endpoint
- Full API documentation at `/docs`

### 🎨 Professional UI/UX
- Modern, responsive design
- Smooth animations and transitions
- Color-coded sentiment indicators
- Drag-and-drop file upload
- Mobile-friendly interface

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd frontend-app
```

**2. Set environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your credentials
nano .env
```

**3. Start the services:**
```bash
docker-compose up -d
```

**4. Access the application:**
- **Web UI:** http://localhost:5000
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

**5. Stop the services:**
```bash
docker-compose down
```

### Option 2: Local Development

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Set environment variables:**
```bash
export SAGEMAKER_ENDPOINT=sentiment-analysis-endpoint
export AWS_REGION=ap-south-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

**3. Start API server (Terminal 1):**
```bash
cd backend
python api.py
```

**4. Start web server (Terminal 2):**
```bash
cd backend
export API_URL=http://localhost:8000
python app.py
```

**5. Access:**
- Web UI: http://localhost:5000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The service was excellent and staff was very helpful!"
  }'
```

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.95,
  "probabilities": {
    "positive": 0.95,
    "neutral": 0.03,
    "negative": 0.02
  },
  "sentiment_emoji": "😊",
  "interpretation": "This text expresses very strong positive sentiment..."
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "tweets": [
      "Great product!",
      "Okay service",
      "Terrible experience"
    ]
  }'
```

### CSV Upload

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@texts.csv"
```

## 📄 CSV Format

Your CSV file should contain a column with text data. The API automatically detects columns named:
- `text` (recommended)
- `tweet`
- `message`
- `content`
- `review`
- `comment`

**Example CSV:**
```csv
text
"The service was excellent and staff was very helpful!"
"Flight departed on time."
"Terrible experience, would not recommend."
```

**Download Sample:** Use the "Download Sample CSV" button in the Batch page.

## 🏗️ Project Structure

```
frontend-app/
├── backend/
│   ├── api.py              # FastAPI backend
│   └── app.py              # Flask web server
├── frontend/
│   ├── templates/
│   │   ├── index.html      # Single prediction UI
│   │   └── batch.html      # Batch prediction UI
│   └── static/
│       ├── css/
│       │   └── style.css   # Styles
│       └── js/
│           ├── app.js      # Single prediction JS
│           └── batch.js    # Batch prediction JS
├── docker-compose.yml      # Docker orchestration
├── Dockerfile.api          # API container
├── Dockerfile.web          # Web container
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

## ⚙️ Configuration

### Environment Variables

**API Service (backend/api.py):**
- `SAGEMAKER_ENDPOINT` - SageMaker endpoint name
- `AWS_REGION` - AWS region (default: `ap-south-1`)
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_SESSION_TOKEN` - AWS session token (optional)

**Web Service (backend/app.py):**
- `API_URL` - API server URL (default: `http://localhost:8000`)

### GitHub Secrets/Variables

For CI/CD integration:

**GitHub Variables:**
- `ENDPOINT_NAME` - Your SageMaker endpoint name
- `AWS_REGION` - Your AWS region
- `S3_BUCKET` - Your S3 bucket name
- `PROJECT_NAME` - Your project name

**GitHub Secrets:**
- `AWS_ROLE_ARN` - IAM role for GitHub Actions
- `SAGEMAKER_ROLE_ARN` - IAM role for SageMaker

## 🎯 Model Details

- **Architecture:** DistilBERT (66M parameters)
- **Training Data:** Twitter Airline Sentiment (14,640 samples)
- **Classes:** Positive, Neutral, Negative
- **Framework:** PyTorch + Hugging Face Transformers
- **Deployment:** AWS SageMaker ml.m5.large endpoint
- **Inference Time:** ~200ms per prediction

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~80% |
| F1-Score | ~0.78 |
| Precision | ~0.79 |
| Recall | ~0.80 |

## 🔒 Security Considerations

### For Production Deployment:

1. **Authentication & Authorization**
   - Add user authentication
   - Implement API key authentication
   - Rate limiting per user/IP

2. **HTTPS/SSL**
   - Use HTTPS in production
   - Get SSL certificates (Let's Encrypt)
   - Configure secure headers

3. **Input Validation**
   - Validate text length
   - Sanitize inputs
   - Check file sizes

4. **CORS Configuration**
   - Update allowed origins
   - Remove wildcard (`*`) from production

5. **Environment Security**
   - Use AWS Secrets Manager
   - Never commit `.env` files
   - Rotate credentials regularly

6. **Monitoring & Logging**
   - Set up CloudWatch monitoring
   - Log all API requests
   - Track endpoint usage

## 🐛 Troubleshooting

### Endpoint Not Found

```bash
# Check if endpoint exists
aws sagemaker describe-endpoint \
  --endpoint-name sentiment-analysis-endpoint

# Update environment variable
export SAGEMAKER_ENDPOINT=your-actual-endpoint-name
```

### AWS Credentials Error

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=ap-south-1
```

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different ports in docker-compose.yml
ports:
  - "5001:5000"  # Use port 5001 instead
```

### Docker Issues

```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# View logs
docker-compose logs -f

# Check container status
docker-compose ps
```

### Model Prediction Errors

**Symptoms:** 500 errors, timeout
**Solutions:**
- Ensure endpoint is InService: `aws sagemaker describe-endpoint --endpoint-name <name>`
- Check endpoint logs in CloudWatch
- Verify IAM permissions
- Check AWS credentials

## 📊 API Rate Limits

- **Single prediction:** No limit
- **Batch prediction:** Up to 100 texts per request
- **CSV upload:** Max 10MB file size, ~1000 rows recommended

## 💰 Cost Estimation

Running the SageMaker endpoint:
- **Instance:** ml.m5.large
- **Cost:** ~$0.115/hour (~$83/month if running 24/7)
- **Recommendation:** Stop endpoint when not in use

```bash
# Stop endpoint to save costs
aws sagemaker delete-endpoint --endpoint-name sentiment-analysis-endpoint
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - See LICENSE file for details

## 🙋 Support

For issues or questions:
- Open a GitHub issue
- Check the [troubleshooting section](#-troubleshooting)
- Review API documentation at `/docs`

## 🔗 Related Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

---

Made with ❤️ using DistilBERT and AWS SageMaker
