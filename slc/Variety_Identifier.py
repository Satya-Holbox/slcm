from PIL import Image
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

prompt = """Identify the variety of rice in the provided image.  

Tasks:
1. Detect the rice variety (e.g., Basmati, Jasmine, Sona Masoori, Arborio, Brown rice, Wild rice, Parboiled rice, etc.)
2. Provide confidence level (%) for the classification
3. Mention distinguishing features used to identify the variety (e.g., long-grain, short-grain, color, texture)
4. If uncertain, list top 3 possible varieties with confidence levels

Return response strictly in JSON:
{
    "predicted_variety": "Basmati | Jasmine | Sona Masoori | Arborio | Brown Rice | Wild Rice | Other",
    "confidence": number,
    "possible_varieties": [
        {"variety": "name", "confidence": number},
        {"variety": "name", "confidence": number},
        {"variety": "name", "confidence": number}
    ],
    "features_used": "short explanation of key features observed"
}
"""

def identify_rice_variety(image_path):
    """Identify rice variety using Gemini API."""
    try:
        # Open image
        image = Image.open(image_path)
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
            [image, prompt]
        )

        # Clean response
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`")
            if cleaned_text.lower().startswith("json"):
                cleaned_text = cleaned_text[4:].strip()

        # Parse JSON
        try:
            result = json.loads(cleaned_text)
            if isinstance(result, dict):
                return result
            else:
                return {"error": "Unexpected response type", "raw_response": cleaned_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": cleaned_text}

    except Exception as e:
        return {"error": f"Error identifying rice variety: {str(e)}"}
