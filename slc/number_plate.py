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

def extract_number_plate(image_bytes):
    """
    Extracts the number plate from a vehicle in an image.
    Args:
        image_bytes (bytes): The raw bytes of the image file.
    Returns:
        str: The extracted alphanumeric characters of the number plate, or an error message.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        prompt = """
        Extract the number plate from the vehicle shown in this image. 
        Only return the exact alphanumeric characters of the license plate number.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        
        return response.text.strip()

    except Exception as e:
        return {"error": str(e)}
