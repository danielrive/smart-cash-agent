from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
import base64
import mimetypes
import json

vision_llm = init_chat_model(
    "us.amazon.nova-lite-v1:0", 
    model_provider="bedrock_converse"
)

def extract_text(img_path: str) -> dict:
    """
    Extract structured expense data from a receipt image using a multimodal model.

    Args:
        img_path (str): Path to the image file (JPG, PNG, etc.).

    Returns:
        dict: {
            "store": str,
            "date": str,
            "items": list[{"name": str, "price": float}],
            "total": float,
            "currency": str
        }

    Notes:
        - The function will attempt to parse the LLM output as JSON.
        - If parsing fails, the raw text is returned in {"raw_text": "..."}.
    """
    try:
        # Read image
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Encode as base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect mime type (default to png if unknown)
        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type:
            mime_type = "image/png"

        # Construct message
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "You are an OCR and information extraction system. "
                            "Extract structured data from this receipt image. "
                            "Return the result as a valid JSON object with the following fields:\n"
                            "{'store': str, 'date': str, 'items': [{'name': str, 'price': float}], "
                            "'total': float, 'currency': str}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call vision model
        response = vision_llm.invoke(message)

        # Handle model response
        raw_output = response.content if isinstance(response.content, str) else str(response.content)

        try:
            # Try parsing JSON directly
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback to returning raw text
            return {"raw_text": raw_output}

    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
