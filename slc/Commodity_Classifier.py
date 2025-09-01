from PIL import Image
import os
import json
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)


def classify_commodity(image_bytes: bytes):
    """
    Classify commodity using Gemini API with structured JSON output.
    
    Args:
        image_bytes (bytes): Raw image bytes.
    
    Returns:
        dict: JSON result with predicted commodity, confidence, possible classes, and features.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        model = genai.GenerativeModel("gemini-1.5-flash")  # API key supported model

        prompt = """
        Identify the commodity in this image. Provide:
        1. Predicted commodity (Rice, Dal, Corn, Wheat, Barley, Pulses, Seeds, Other)
        2. Confidence score (%)
        3. Top 3 possible commodities with confidence
        4. Features used (short description)

        Return strictly in JSON format (without markdown fences):
        {
            "predicted_commodity": "Rice | Dal | Corn | Wheat | Barley | Pulses | Seeds | Other",
            "confidence": number,
            "possible_classes": [
                {"commodity": "name", "confidence": number},
                {"commodity": "name", "confidence": number},
                {"commodity": "name", "confidence": number}
            ],
            "features_used": "short explanation"
        }
        """

        response = model.generate_content([image, prompt])
        text_response = response.text.strip()

        # 🧹 Clean if Gemini wrapped output in ```json ... ```
        if text_response.startswith("```"):
            text_response = text_response.strip("`")
            if text_response.lower().startswith("json"):
                text_response = text_response[4:].strip()

        return json.loads(text_response)

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from Gemini response", "raw_output": response.text}
    except Exception as e:
        return {"error": f"Error classifying commodity: {str(e)}"}
