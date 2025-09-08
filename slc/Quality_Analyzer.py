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


def analyze_grain_quality(image_bytes: bytes):
    """
    Analyze rice grain quality using Gemini API with structured JSON output.
    
    Args:
        image_bytes (bytes): Raw image file bytes.
    
    Returns:
        dict: JSON result with quality scores, issues, and recommendations.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        model = genai.GenerativeModel("gemini-1.5-flash")  # API-key supported model

        prompt = """
        Analyze the rice grain quality in this image. Provide:
        1. Overall quality (Excellent, Good, Fair, Poor)
        2. Percentage of each quality category
        3. List common issues observed (broken, discolored, chalky, shriveled)
        4. Recommendations to improve quality

        Return strictly in JSON format (without markdown fences):
        {
            "overall_quality": "Excellent | Good | Fair | Poor",
            "quality_distribution": {
                "excellent": {"percentage": number},
                "good": {"percentage": number},
                "fair": {"percentage": number},
                "poor": {"percentage": number}
            },
            "issues_detected": [list of issues],
            "recommendations": "short actionable advice"
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
        return {"error": f"Error analyzing grain quality: {str(e)}"}
