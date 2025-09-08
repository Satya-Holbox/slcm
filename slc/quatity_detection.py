from PIL import Image
from google import genai
import os
from google.genai.types import HttpOptions
from io import BytesIO

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
        Count the total number of rice bags visible in this image. 
        Provide the final count as a single number.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )
        
        return int(response.text.strip())

    except Exception as e:
        return {"error": str(e)}
