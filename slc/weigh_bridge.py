from PIL import Image
from google import genai
import os
from google.genai.types import HttpOptions
from io import BytesIO
import json

try:
    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_location = "global"
    
    if not gcp_project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set.")
    
    client = genai.Client(
        vertexai=True,
        project=gcp_project_id,
        location=gcp_location,
        http_options=HttpOptions(api_version="v1")
    )
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")

def detect_weigh_bridge_slip(image_bytes):
    """
    Extracts data from a weighbridge slip image.
    Args:
        image_bytes (bytes): The raw bytes of the image file.
    Returns:
        dict: A dictionary containing the extracted weighbridge slip data.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        prompt = """
        Analyze the provided image of a weighbridge slip. Extract the following key entities:
        - Slip Number
        - Vehicle Number
        - Date
        - Gross Weight
        - Tare Weight
        - Net Weight
        - Material/Product
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"response": response.text}

    except Exception as e:
        return {"error": str(e)}
