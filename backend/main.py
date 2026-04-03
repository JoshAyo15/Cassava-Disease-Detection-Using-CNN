"""
FastAPI Backend for Cassava Disease Classification
Serves ML model predictions and provides analytics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import json
import io
from pathlib import Path
import logging
from datetime import datetime
import sqlite3
from backend.predictor import DiseasePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cassava Disease Classifier API",
    description="ML-powered API for cassava leaf disease classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None

# Database setup
DB_PATH = Path(__file__).parent / "predictions.db"


def init_db():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_filename TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(disease: str, confidence: float, filename: str):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (disease, confidence, image_filename)
            VALUES (?, ?, ?)
        """, (disease, confidence, filename))
        conn.commit()
        conn.close()
        logger.info(f"✓ Prediction saved: {disease} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"✗ Database error: {e}")


@app.on_event("startup")
async def startup():
    """Initialize model and database on startup"""
    global predictor
    logger.info("Initializing application...")

    try:
        predictor = DiseasePredictor()
        init_db()
        logger.info("✓ Application initialized successfully")
    except Exception as e:
        logger.error(f"✗ Startup error: {e}")
        raise


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }


@app.get("/info", tags=["Info"])
async def get_info():
    """Get API and model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "title": "Cassava Disease Classifier",
        "version": "1.0.0",
        "model_architecture": "ConvNeXt Small",
        "classes": predictor.classes,
        "supported_formats": ["jpg", "jpeg", "png"],
        "max_file_size_mb": 10
    }


@app.post("/predict", tags=["Prediction"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Classify cassava leaf disease from uploaded image

    - **file**: Image file (jpg, jpeg, png)

    Returns:
    - **disease**: Predicted disease class
    - **confidence**: Prediction confidence (0-1)
    - **all_predictions**: All class probabilities
    """

    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    try:
        # Read image
        img_data = await file.read()
        img_stream = io.BytesIO(img_data)

        # Get prediction
        result = predictor.predict(img_stream)

        # Save to database
        save_prediction(
            result["disease"],
            result["confidence"],
            file.filename
        )

        return {
            "success": True,
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics", tags=["Analytics"])
async def get_analytics():
    """Get prediction analytics and statistics"""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        # Disease breakdown
        cursor.execute("""
            SELECT disease, COUNT(*) as count, 
                   AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY disease
            ORDER BY count DESC
        """)
        disease_stats = []
        for row in cursor.fetchall():
            disease_stats.append({
                "disease": row[0],
                "count": row[1],
                "avg_confidence": round(row[2], 4)
            })

        # Recent predictions
        cursor.execute("""
            SELECT disease, confidence, prediction_time
            FROM predictions
            ORDER BY prediction_time DESC
            LIMIT 10
        """)
        recent = []
        for row in cursor.fetchall():
            recent.append({
                "disease": row[0],
                "confidence": round(row[1], 4),
                "time": row[2]
            })

        conn.close()

        return {
            "total_predictions": total,
            "disease_breakdown": disease_stats,
            "recent_predictions": recent
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Analytics unavailable")


@app.get("/predictions/export", tags=["Export"])
async def export_predictions():
    """Export all predictions as JSON"""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, disease, confidence, prediction_time, image_filename
            FROM predictions
            ORDER BY prediction_time DESC
        """)

        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                "id": row[0],
                "disease": row[1],
                "confidence": round(row[2], 4),
                "timestamp": row[3],
                "image": row[4]
            })

        conn.close()

        return {
            "total": len(predictions),
            "data": predictions
        }

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Export failed")


@app.get("/predictions/export.xlsx", tags=["Export"])
async def export_predictions_xlsx():
    try:
        import pandas as pd
        import io
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, disease, confidence, prediction_time, image_filename
            FROM predictions
            ORDER BY prediction_time DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(
            rows, columns=["ID", "Disease", "Confidence", "Timestamp", "Image"])
        df["Confidence"] = df["Confidence"].round(4)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Predictions")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=cassava-predictions.xlsx"},
        )
    except Exception as e:
        logger.error(f"XLSX export error: {e}")
        raise HTTPException(status_code=500, detail="XLSX export failed")


@app.get("/predictions/export.pdf", tags=["Export"])
async def export_predictions_pdf():
    try:
        import io
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, disease, confidence, prediction_time, image_filename
            FROM predictions
            ORDER BY prediction_time DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        data = [["ID", "Disease", "Confidence", "Timestamp", "Image"]]
        for r in rows:
            data.append(
                [r[0], r[1], f"{round(r[2]*100, 1)}%", r[3], r[4] or ""])
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=landscape(
            letter), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
        styles = getSampleStyleSheet()
        title = Paragraph("Cassava Predictions Report", styles["Title"])
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ]))
        story = [title, Spacer(1, 12), tbl]
        doc.build(story)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=cassava-predictions.pdf"},
        )
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        raise HTTPException(status_code=500, detail="PDF export failed")


@app.delete("/predictions/clear", tags=["Admin"])
async def clear_predictions():
    """Clear all prediction history (use with caution)"""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": "All predictions cleared"
        }

    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail="Clear failed")


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Cassava Disease Classifier API",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
