from PIL import Image
from google import genai
import os
import json
from io import BytesIO

# Initialize Gemini Client with API key
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    client = genai.Client(api_key=api_key)

except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    client = None


# def analyze_rice_image(image_bytes):
#     """
#     Analyze rice grains in an image using Gemini API (API key authentication).
    
#     Args:
#         image_bytes (bytes): Raw image bytes.
        
#     Returns:
#         dict: Structured JSON with counts, quality classification, and defects.
#     """
#     if not client:
#         return {"error": "Gemini client not initialized."}

#     try:
#         # Load image with PIL
#         image = Image.open(BytesIO(image_bytes))
        
#         # Prompt for structured JSON output
#         prompt = """
#         Analyze the rice grains in this image. 
#         Tasks:
#         1. Count total grains.
#         2. Classify into whole, broken, discolored.
#         3. List visible defects.

#         Return strictly in JSON format:
#         {
#           "total_grains": number,
#           "grain_quality": {
#               "whole_grains": {"count": number, "percentage": number},
#               "broken_grains": {"count": number, "percentage": number},
#               "discolored_grains": {"count": number, "percentage": number}
#           },
#           "defects": ["list of defects"],
#           "analysis_summary": "short summary in one sentence"
#         }
#         """

#         response = client.models.generate_content(
#             model="gemini-1.5-flash",   # API key supported model
#             contents=[image, prompt]
#         )

#         return json.loads(response.text.strip())

#     except json.JSONDecodeError:
#         return {"error": "Failed to parse JSON from Gemini response", "raw_output": response.text}
#     except Exception as e:
#         return {"error": f"Error analyzing rice grains: {str(e)}"}
def analyze_rice_image(image_bytes):
    """
    Analyze rice grains in an image using Gemini API (API key authentication).
    
    Args:
        image_bytes (bytes): Raw image bytes.
        
    Returns:
        dict: Structured JSON with counts, quality classification, and defects.
    """
    if not client:
        return {"error": "Gemini client not initialized."}

    try:
        image = Image.open(BytesIO(image_bytes))

        prompt = """
        Analyze the rice grains in this image. 
        Tasks:
        1. Count total grains.
        2. Classify into whole, broken, discolored.
        3. List visible defects.

        Return strictly in JSON format (without markdown, without ```).
        Example:
        {
          "total_grains": 100,
          "grain_quality": {
              "whole_grains": {"count": 90, "percentage": 90.0},
              "broken_grains": {"count": 8, "percentage": 8.0},
              "discolored_grains": {"count": 2, "percentage": 2.0}
          },
          "defects": ["example defect"],
          "analysis_summary": "short text"
        }
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[image, prompt]
        )

        text_response = response.text.strip()

        # 🧹 Clean Gemini output (remove ```json ... ``` if present)
        if text_response.startswith("```"):
            text_response = text_response.strip("`")
            if text_response.lower().startswith("json"):
                text_response = text_response[4:].strip()

        return json.loads(text_response)

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from Gemini response", "raw_output": response.text}
    except Exception as e:
        return {"error": f"Error analyzing rice grains: {str(e)}"}
