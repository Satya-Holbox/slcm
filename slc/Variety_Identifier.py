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


def identify_rice_variety(image_bytes: bytes):
    """
    Identify rice variety using Gemini API with structured JSON output.
    
    Args:
        image_bytes (bytes): Raw image bytes.
    
    Returns:
        dict: JSON result with predicted variety, confidence, possible varieties, and features.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        model = genai.GenerativeModel("gemini-1.5-flash")  # API key supported model

        prompt = """
        Identify the variety of rice in this image. Provide:
        1. Predicted variety (Basmati, Jasmine, Sona Masoori, Arborio, Brown Rice, Wild Rice, Other)
        2. Confidence level (%)
        3. Top 3 possible varieties with confidence
        4. Features used (short description)

        Return strictly in JSON format (without markdown fences):
        {
            "predicted_variety": "Basmati | Jasmine | Sona Masoori | Arborio | Brown Rice | Wild Rice | Other",
            "confidence": number,
            "possible_varieties": [
                {"variety": "name", "confidence": number},
                {"variety": "name", "confidence": number},
                {"variety": "name", "confidence": number}
            ],
            "features_used": "short explanation"
        }
        """

        response = model.generate_content(
            [image, prompt]
        )

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
        return {"error": f"Error identifying rice variety: {str(e)}"}
