import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# MobileNet class dictionary
dct = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# Load MobileNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = models.mobilenet_v3_large(weights=None)
mobilenet.classifier[3] = torch.nn.Linear(in_features=1280, out_features=6)
mobilenet.load_state_dict(torch.load("../mobilenet/classifier.pth", map_location=device))
mobilenet.eval().to(device)

# Load DeepLab model
deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
num_classes = 60
in_channels = deeplab.classifier[4].in_channels
deeplab.classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
deeplab.load_state_dict(torch.load("../deeplab/segmentation.pth", map_location=device), strict=False)
deeplab.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of Canadian provinces and territories
provinces = [
    'Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador',
    'Nova Scotia', 'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan',
    'Northwest Territories', 'Nunavut', 'Yukon'
]

# Garbage classification tips dictionary
garbage_tips = {
    'cardboard': {
        'Tip': 'Cardboard goes into the blue bin. Flatten boxes to save space and remove any non-paper packing materials.',
        'Extra Info': 'Some areas may have dedicated cardboard recycling bins; check your local waste management rules.'
    },
    'glass': {
        'Tip': 'Glass items should be placed in the blue bin for recycling. Ensure glass is clean and free from food residues.',
        'Extra Info': 'Some municipalities may have specific guidelines for different types of glass, like colored or treated glass, so it’s wise to confirm local rules.'
    },
    'metal': {
        'Tip': 'Recyclable metals should be placed in the blue bin. Ensure items are empty and free from significant contamination before recycling.',
        'Extra Info': 'Check local guidelines to confirm specific metal items accepted and any preparation required.'
    },
    'paper': {
        'Tip': 'Paper, including newspapers, office paper, and mail, goes into the blue bin. Avoid recycling paper soiled with food or oil.',
        'Extra Info': 'Check local guidelines to see if shredded paper is accepted and how it should be prepared (often it should be bagged).'
    },
    'plastic': {
        'Tip': 'Clean and rinse plastic containers before recycling them in the blue bin. Caps can often be left on bottles.',
        'Extra Info': 'Verify the types of plastic accepted locally, as recycling capabilities vary.'
    },
    'trash': {
        'Tip': 'Dispose of non-recyclable and non-organic waste in the black bin. Evaluate each item to decide the appropriate disposal method based on its material.',
        'Extra Info': 'For organic food waste, use the green bin. Check if certain wastes like yard waste have specific collection rules in your area.'
    }
}

# Province websites dictionary
province_websites = {
    'Alberta': 'https://www.alberta.ca/recycling',
    'British Columbia': 'https://recyclebc.ca/',
    'Manitoba': 'https://simplyrecycle.ca/recycling-guides/',
    'New Brunswick': 'https://www.circularmaterials.ca/resident-provinces/new-brunswick/',
    'Newfoundland and Labrador': 'https://mmsb.nl.ca/',
    'Nova Scotia': 'https://divertns.ca/',
    'Ontario': 'https://www.toronto.ca/services-payments/recycling-organics-garbage/houses/what-goes-in-my-blue-bin/',
    'Prince Edward Island': 'https://iwmc.pe.ca/sort/',
    'Quebec': 'https://www.recyc-quebec.gouv.qc.ca/citoyens/mieux-recuperer/quest-ce-qui-va-dans-le-bac/',
    'Saskatchewan': 'https://www.saskwastereduction.ca/recycle/',
    'Northwest Territories': 'https://www.gov.nt.ca/ecc/en/services/waste-reduction-and-recycling',
    'Nunavut': 'https://www.iqaluit.ca/fr/news/recycling',
    'Yukon': 'https://yukon.ca/en/recycling-materials-facilities'
}

# MobileNet classification function
def classify_image(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = mobilenet(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence_score = probabilities[predicted_class].item()
    class_name = dct[predicted_class]
    return class_name, confidence_score

# DeepLab segmentation function
def segment_image(image):
    # Save original image dimensions
    original_width, original_height = image.size

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]  # Shape: [num_classes, H, W]
        # Transpose dimensions for computation
        output = output.permute(1, 2, 0)  # Shape: [H, W, num_classes]
        # Compute class probabilities for each pixel
        probabilities = torch.nn.functional.softmax(output, dim=-1)  # Softmax over num_classes
        # Get predicted classes for each pixel
        predicted_classes = probabilities.argmax(dim=-1)  # Shape: [H, W]
        # Get confidence scores for predicted classes
        predicted_probs = probabilities.gather(-1, predicted_classes.unsqueeze(-1)).squeeze(-1)  # Shape: [H, W]
        # Calculate average confidence score
        mean_confidence = predicted_probs.mean().item()
        # Convert predicted classes to NumPy array
        output_predictions = predicted_classes.cpu().numpy()

    # Resize segmentation mask to original image size
    resized_mask = cv2.resize(output_predictions.astype(np.uint8), (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # Create color mask
    mask_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color
    color_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    color_mask[resized_mask > 0] = mask_color

    # Convert original image to NumPy array
    image_np = np.array(image)

    # Ensure image is in RGB format
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Ensure image and mask sizes match
    if image_np.shape[:2] != color_mask.shape[:2]:
        color_mask = cv2.resize(color_mask, (image_np.shape[1], image_np.shape[0]))

    # Overlay mask on image
    overlay = cv2.addWeighted(image_np, 1.0, color_mask, 0.4, 0)

    # Save result
    result_path = "segmentation_result.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, overlay_bgr)

    return result_path, mean_confidence

# Combined function
def process_image(image, province):
    class_result, class_confidence_score = classify_image(image)
    segmentation_result, segmentation_confidence_score = segment_image(image)
    user_location = province  # Use the selected province or territory

    # Get garbage tips for the classified object
    class_name_lower = class_result.lower()
    tips = garbage_tips.get(class_name_lower, {})
    tip_text = tips.get('Tip', 'No tips available.')
    extra_info = tips.get('Extra Info', '')

    # Format tips as markdown
    tip_markdown = f"**Tip:** {tip_text}"
    extra_info_markdown = f"**Extra Info:** {extra_info}"

    # Get website for the selected province or territory
    province_website_url = province_websites.get(province, 'No website available.')
    province_website_markdown = f"**Your Province or Territory's Recycling Website:** [{province_website_url}]({province_website_url})"

    return (class_result, class_confidence_score,
            segmentation_result, segmentation_confidence_score,
            tip_markdown, extra_info_markdown, province_website_markdown, user_location)

# Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=provinces, label="Select Your Province or Territory")
    ],
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Number(label="Classification Confidence Score"),
        gr.Image(type="filepath", label="Segmentation Result"),
        gr.Number(label="Segmentation Confidence Score"),
        gr.Markdown(label="Tip"),
        gr.Markdown(label="Extra Info"),
        gr.Markdown(label="Province Recycling Website"),
        gr.Textbox(label="Your Location")
    ],
    title="Garbage Classification and Segmentation",
    description="Upload an image of garbage. Select your province or territory. The system will classify the image, perform segmentation, display recycling tips, and your location's recycling website."
)

if __name__ == "__main__":
    interface.launch()