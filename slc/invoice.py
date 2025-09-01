from PIL import Image
from google import genai

client = genai.Client()

prompt = "Analyze the provided image of an invoice. Extract the following key entities: **Invoice Number**, **Date of Issue**, **Vendor Name**, **Total Amount Due**, and a list of **line items**. For each line item, extract the **Description**, **Quantity**, **Unit Price**, and **Line Total**. Format the output as a JSON object."
image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image, prompt]
)
print(response.text)