from PIL import Image
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini API client with API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

prompt = """Analyze the provided image of rice grains. Detect and count the number of rice grains visible in the image. 
Provide the following information:
1. Total number of rice grains detected
2. Classification of grain quality (e.g., whole grains, broken grains, discolored grains)
3. Percentage of each quality category
4. Any visible defects or abnormalities in the grains

Format the output as a JSON object with the following structure:
{
    "total_grains": number,
    "grain_quality": {
        "whole_grains": {"count": number, "percentage": number},
        "broken_grains": {"count": number, "percentage": number},
        "discolored_grains": {"count": number, "percentage": number}
    },
    "defects": [list of any detected defects],
    "analysis_summary": "brief text summary of the overall grain quality"
}
"""

def analyze_rice_image(image_path):
    """Analyze rice grains in the provided image using Gemini API."""
    try:
        # Open the image
        image = Image.open(image_path)
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
            [image, prompt]
        )

        # Clean the response text
        cleaned_text = response.text.strip()

        # Remove markdown code fences if present
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`")  # remove backticks
            if cleaned_text.lower().startswith("json"):
                cleaned_text = cleaned_text[4:].strip()

        # Try parsing JSON
        try:
            result = json.loads(cleaned_text)
            if isinstance(result, dict):
                return result
            else:
                return {"error": "Unexpected response type", "raw_response": cleaned_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": cleaned_text}

    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}
