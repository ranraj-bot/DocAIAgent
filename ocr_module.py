from PIL import Image
import io

def extract_text(image_bytes: bytes) -> str:
    """Extracts text from an image provided as bytes. Placeholder function."""
    # In a real scenario, you would use Tesseract, AWS Textract, Google Vision API, etc.
    # For the demo, we return mock text based perhaps on the first few bytes
    # or just a static string.
    print(f"Received {len(image_bytes)} bytes for OCR (using mock).")
    # Simulate some processing
    try:
        # Try to open the image to check if it's valid (optional)
        img = Image.open(io.BytesIO(image_bytes))
        print(f"Mock OCR: Image size {img.size}, format {img.format}")
    except Exception as e:
        print(f"Mock OCR: Could not open image - {e}")
        return "MOCK OCR ERROR: Could not read image data."

    # Return consistent mock text for the demo
    return "MOCK TEXT:\nINVOICE\nNumber: INV-000123\nDate: 2025-05-01\nVendor: ACME Corp\nTotal Amount: $1,234.56\nLine Items:\nItem A - $1000.00\nItem B - $234.56"

# Simple test (optional)
if __name__ == "__main__":
    # Create a dummy image bytes object (e.g., a small black square)
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        print("Testing OCR stub:")
        print(extract_text(img_bytes))
    except Exception as e:
        print(f"Error in OCR stub test: {e}") 