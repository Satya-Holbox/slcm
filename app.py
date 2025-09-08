from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from fastapi import HTTPException
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from fastapi.responses import JSONResponse


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

# Grain Analysis setup

# Grain Analysis setup

class GrainAnalysisResponse(BaseModel):
    result: dict 
@app.post("/api/demo_backend_v2/analyze_rice_grains")
async def analyze_rice_grains_api(file: UploadFile = File(...)):
    """
    Endpoint to analyze rice grains in an uploaded image using Gemini API.
    """
    try:
        image_bytes = await file.read()
        result = analyze_rice_image(image_bytes)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error analyzing rice grains: {str(e)}"},
            status_code=500
        )

# Grain Quality Analyzer setup

class GrainQualityResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/analyze_grain_quality", response_model=GrainQualityResponse)
async def analyze_grain_quality_api(file: UploadFile = File(...)):
    """
    Endpoint to analyze rice grain quality using Gemini API.
    """
    try:
        image_bytes = await file.read()
        result = analyze_grain_quality(image_bytes)  # pass bytes, not path
        return {"result": result}
    except Exception as e:
        return {"error": f"Error analyzing grain quality: {str(e)}"}
# Rice Variety Identifier setup

class RiceVarietyResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/identify_rice_variety", response_model=RiceVarietyResponse)
async def identify_rice_variety_api(file: UploadFile = File(...)):
    """
    Endpoint to identify rice variety (e.g., Basmati, Jasmine, etc.) using Gemini API.
    """
    try:
        image_bytes = await file.read()
        result = identify_rice_variety(image_bytes)  # pass bytes, not path
        return {"result": result}
    except Exception as e:
        return {"error": f"Error identifying rice variety: {str(e)}"}

# Commodity Classifier setup
class CommodityResponse(BaseModel):
    result: dict

@app.post("/api/demo_backend_v2/classify_commodity", response_model=CommodityResponse)
async def classify_commodity_api(file: UploadFile = File(...)):
    """
    Endpoint to classify commodity type (Rice, Dal, Corn, Wheat, etc.) using Gemini API.
    """
    try:
        image_bytes = await file.read()
        result = classify_commodity(image_bytes)  # pass bytes
        return {"result": result}
    except Exception as e:
        return {"error": f"Error classifying commodity: {str(e)}"}

@app.post("/api/demo_backend_v2/invoice_extraction")
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

@app.post("/api/demo_backend_v2/number_plate_extraction")
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

@app.post("/api/demo_backend_v2/rice_bags_detection")
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

@app.post("/api/demo_backend_v2/weigh_bridge_slip_detection")
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


@app.get("/health")
def healthcheck():
    return "Hello"