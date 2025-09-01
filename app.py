from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from fastapi import HTTPException
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from slc.Grain_Detector import analyze_rice_image
from slc.Quality_Analyzer import analyze_grain_quality
from slc.Commodity_Classifier import classify_commodity
from slc.Variety_Identifier import identify_rice_variety
from slc.invoice import extract_invoice_entities
from slc.number_plate import extract_number_plate
from slc.quatity_detection import count_bags
from slc.weigh_bridge import detect_weigh_bridge_slip

class GrainAnalysisResponse(BaseModel):
    result: dict 
@app.post("/api/demo_backend_v2/analyze_rice_grains")
async def analyze_rice_grains_api(
    image: UploadFile = File(..., description="Upload a rice grain image for analysis")
):
    """
    API endpoint to analyze rice grains in an uploaded image using Gemini API.

    This endpoint accepts an image file containing rice grains and returns a detailed analysis including:
    - Total number of grains detected
    - Classification of grain quality (whole, broken, discolored)
    - Percentage of each quality category
    - Any visible defects or abnormalities
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Save uploaded image temporarily
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        # Call the rice grain analysis function
        analysis_result = analyze_rice_image(file_path)

        # Clean up the temporary file
        os.remove(file_path)

        return GrainAnalysisResponse(result=analysis_result)

    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing rice grains: {str(e)}")

# Grain Quality Analyzer setup

class GrainQualityResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/analyze_grain_quality")
async def analyze_grain_quality_api(
    image: UploadFile = File(..., description="Upload a rice grain image for quality analysis")
):
    """
    API endpoint to analyze rice grain quality using Gemini API.

    Returns:
    - Overall quality rating (Excellent, Good, Fair, Poor)
    - Distribution percentages
    - Issues detected
    - Recommendations
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        analysis_result = analyze_grain_quality(file_path)
        os.remove(file_path)
        return GrainQualityResponse(result=analysis_result)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing grain quality: {str(e)}")
# Rice Variety Identifier setup

class RiceVarietyResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/identify_rice_variety")
async def identify_rice_variety_api(
    image: UploadFile = File(..., description="Upload a rice image for variety identification")
):
    """
    API endpoint to identify rice variety (e.g., Basmati, Jasmine, etc.) using Gemini API.

    Returns:
    - Predicted rice variety
    - Confidence level
    - Top 3 possible varieties (if uncertain)
    - Features used for classification
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        analysis_result = identify_rice_variety(file_path)
        os.remove(file_path)
        return RiceVarietyResponse(result=analysis_result)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error identifying rice variety: {str(e)}")

# Commodity Classifier setup

class CommodityClassifierResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/classify_commodity")
async def classify_commodity_api(
    image: UploadFile = File(..., description="Upload an image to classify commodity type")
):
    """
    API endpoint to classify commodity type (Rice, Dal, Corn, Wheat, etc.) using Gemini API.

    Returns:
    - Predicted commodity
    - Confidence score
    - Top 3 possible classes
    - Features used for classification
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        analysis_result = classify_commodity(file_path)
        os.remove(file_path)
        return CommodityClassifierResponse(result=analysis_result)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error classifying commodity: {str(e)}")


@app.post("/invoice_extraction/")
async def invoice_extraction_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for invoice entity extraction.
    """
    try:
        image_bytes = await file.read()
        result = extract_invoice_entities(image_bytes)
        return {"result": result}
    except Exception as e:
        return {"error": f"Error processing the invoice: {str(e)}"}

@app.post("/number_plate_extraction/")
async def number_plate_extraction_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for number plate extraction.
    """
    try:
        image_bytes = await file.read()
        result = extract_number_plate(image_bytes)
        return {"result": result}
    except Exception as e:
        return {"error": f"Error processing the number plate image: {str(e)}"}

@app.post("/rice_bags_detection/")
async def rice_bags_detection_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for counting rice bags in an image.
    """
    try:
        image_bytes = await file.read()
        result = count_bags(image_bytes)
        return {"result": result}
    except Exception as e:
        return {"error": f"Error processing the rice bag image: {str(e)}"}

@app.post("/weigh_bridge_slip_detection/")
async def weigh_bridge_slip_detection_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for extracting data from a weighbridge slip.
    """
    try:
        image_bytes = await file.read()
        result = detect_weigh_bridge_slip(image_bytes)
        return {"result": result}
    except Exception as e:
        return {"error": f"Error processing the weighbridge slip: {str(e)}"}
