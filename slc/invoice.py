from PIL import Image
from google import genai
import os
from google.genai.types import HttpOptions
from io import BytesIO

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
        Analyze the provided image of an invoice. Extract the following key entities:
        - Invoice Number
        - Date of Issue
        - Vendor Name
        - Total Amount Due
        - A list of line items, where each line item includes:
            - Description
            - Quantity
            - Unit Price
            - Line Total
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        
        return response.text

    except Exception as e:
        return {"error": str(e)}
