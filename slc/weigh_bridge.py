from PIL import Image
from google import genai
import os
from google.genai.types import HttpOptions
from io import BytesIO
import json
import re

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
        Analyze the provided image of a weighbridge slip. Extract the following key entities and format the output as a JSON object. Ensure the values are correctly parsed to their specified data types (e.g., numbers for weights, strings for text). Handle missing or unreadable fields by using `null`.",
        "entities": {
            "slip_number": {
            "type": "string",
            "description": "The unique identification number of the weighbridge slip."
            },
            "vehicle_number": {
            "type": "string",
            "description": "The registration number of the vehicle."
            },
            "print_date": {
            "type": "string",
            "format": "YYYY-MM-DD",
            "description": "The date the slip was printed. Convert to YYYY-MM-DD format if necessary."
            },
            "material": {
            "type": "string",
            "description": "The name of the material or product being weighed."
            },
            "gross_weight": {
            "type": "object",
            "description": "The total weight of the loaded vehicle.",
            "properties": {
                "value": {
                "type": "number",
                "description": "The numerical value of the gross weight."
                },
                "unit": {
                "type": "string",
                "description": "The unit of measurement for the gross weight (e.g., 'kg', 'tonnes')."
                }
            }
            },
            "tare_weight": {
            "type": "object",
            "description": "The empty weight of the vehicle.",
            "properties": {
                "value": {
                "type": "number",
                "description": "The numerical value of the tare weight."
                },
                "unit": {
                "type": "string",
                "description": "The unit of measurement for the tare weight."
                }
            }
            },
            "net_weight": {
            "type": "object",
            "description": "The calculated weight of the material (Gross Weight - Tare Weight).",
            "properties": {
                "value": {
                "type": "number",
                "description": "The numerical value of the net weight."
                },
                "unit": {
                "type": "string",
                "description": "The unit of measurement for the net weight."
                }
            }
            }
        },
        Return response strictly in JSON format:
        
            "slip_info": {
            "slip_number": "string | null",
            "vehicle_number": "string | null",
            "print_date": "string | null",
            "material": "string | null",
            "gross_weight": {
                "value": "number | null",
                "unit": "string | null"
            },
            "tare_weight": {
                "value": "number | null",
                "unit": "string | null"
            },
            "net_weight": {
                "value": "number | null",
                "unit": "string | null"
            }
            }
    """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        try:
            cleaned_response = re.sub(r"```json|```", "", response.text).strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"response": response.text}

    except Exception as e:
        return {"error": str(e)}
