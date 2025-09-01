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

prompt = """Identify the commodity shown in the provided image.  

Tasks:
1. Detect the primary commodity type (e.g., Rice, Dal/Lentils, Corn, Wheat, Barley, Pulses, Seeds, Other).
2. Provide confidence score (%) for the classification.
3. If multiple commodities are present, list the top 3 possible classes with confidence levels.
4. Highlight visual features used for classification (e.g., shape, color, size, texture).

Return response strictly in JSON format:
{
    "predicted_commodity": "Rice | Dal | Corn | Wheat | Barley | Pulses | Seeds | Other",
    "confidence": number,
    "possible_classes": [
        {"commodity": "name", "confidence": number},
        {"commodity": "name", "confidence": number},
        {"commodity": "name", "confidence": number}
    ],
    "features_used": "short explanation of observed features"
}
"""

def classify_commodity(image_path):
    """Classify commodity from image using Gemini API."""
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
        return {"error": f"Error classifying commodity: {str(e)}"}
