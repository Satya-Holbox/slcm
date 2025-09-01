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

prompt = """Analyze the provided image of rice grains for quality assessment. 
Do not focus on total grain count. Instead, evaluate the *quality aspects* of the grains.  

Provide the following details:
1. Classification of grain quality (Excellent, Good, Fair, Poor)
2. Percentage of each category (if multiple categories exist in the image)
3. Common quality issues observed (e.g., broken grains, discolored grains, chalky grains, shriveled grains)
4. Recommendations to improve quality (if applicable)

Return the result in strict JSON format:
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

def analyze_grain_quality(image_path):
    """Analyze rice grain quality using Gemini API."""
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
        return {"error": f"Error analyzing grain quality: {str(e)}"}
