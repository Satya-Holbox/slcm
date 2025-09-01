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

def count_bags(image_bytes):
    """
    Counts the number of rice bags in an image.
    Args:
        image_bytes (bytes): The raw bytes of the image file.
    Returns:
        int or str: The count of rice bags as an integer, or an error message.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        prompt = """
        Identify and extract the numerical weight reading displayed on the weighbridge's digital screen.
        Return only the number.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        
        return int(response.text.strip())

    except Exception as e:
        return {"error": str(e)}
