import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipForQuestionAnswering, AutoProcessor


def load_model(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BlipForQuestionAnswering.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def load_image(image_source):
    if image_source.startswith('http://') or image_source.startswith('https://'):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_source).convert('RGB')
    
    return image


def predict_answer(image, question, model, processor, device):
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return answer


def main():
    parser = argparse.ArgumentParser(description='Run inference on BLIP-VQA model')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or URL')
    parser.add_argument('--question', type=str, required=True,
                       help='Question about the image')
    parser.add_argument('--model_path', type=str,
                       default='MohammadAlameenArtan/BLIP_Model_VizWiz',
                       help='Path to trained model or HuggingFace model name')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, processor, device = load_model(args.model_path)
    
    print("Loading image...")
    image = load_image(args.image)
    
    print(f"\nQuestion: {args.question}")
    print("Generating answer...")
    
    answer = predict_answer(image, args.question, model, processor, device)
    
    print(f"Answer: {answer}")


if __name__ == '__main__':
    main()