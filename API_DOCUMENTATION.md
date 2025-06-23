# API Documentation

## Health Check
- `GET /api/health`  
Returns system status.

## Predict Risk
- `POST /api/predict_risk`  
Requires `student_data` JSON payload.  
Returns `risk_score` and `risk_level`.
