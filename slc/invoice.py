from PIL import Image
from google import genai
import os
from google.genai.types import HttpOptions
from io import BytesIO
import json
import re
client = None
try:
    api_version = "v1alpha"
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if api_key:
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version=api_version)
        )
    elif gcp_project_id:
        client = genai.Client(
            vertexai=True,
            project=gcp_project_id,
            location=gcp_location,
            http_options=HttpOptions(api_version=api_version)
        )
    else:
        raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) or GOOGLE_CLOUD_PROJECT.")
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")

def extract_invoice_entities(image_bytes):
    """
    Extracts key entities from an invoice image.
    Args:
        image_path (str): The path to the image file.
    Returns:
        dict: A dictionary containing the extracted invoice data.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        prompt = """
        Analyze the provided image of an invoice. 
        Extract the following fields and return the result as valid JSON only, without explanations, comments, or code block formatting:
        {
        "invoice_number": "<string>",
        "date_of_issue": "<string in YYYY-MM-DD format>",
        "vendor_name": "<string>",
        "total_amount_due": "<number>",
        "line_items": [
            {
            "description": "<string>",
            "quantity": <number>,
            "unit_price": <number>,
            "line_total": <number>
            }
        ]
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
