from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from slc.invoice import extract_invoice_entities
from slc.number_plate import extract_number_plate
from slc.quatity_detection import count_bags
from slc.weigh_bridge import detect_weigh_bridge_slip

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
