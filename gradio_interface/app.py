import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# MobileNet 类别字典
dct = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# 加载 MobileNet 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet = models.mobilenet_v3_large(weights=None)
mobilenet.classifier[3] = torch.nn.Linear(in_features=1280, out_features=6)
mobilenet.load_state_dict(torch.load("../mobilenet/classifier.pth", map_location=device))
mobilenet.eval().to(device)

# 加载 DeepLab 模型
deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
num_classes = 60
in_channels = deeplab.classifier[4].in_channels
deeplab.classifier[4] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
deeplab.load_state_dict(torch.load("../deeplab/segmentation.pth", map_location=device), strict=False)
deeplab.eval().to(device)

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MobileNet 分类函数
def classify_image(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = mobilenet(input_tensor)
        predicted_class = torch.argmax(output[0]).item()
    class_name = dct[predicted_class]
    return class_name

# DeepLab 分割函数
def segment_image(image):
    # 保存原始图片的尺寸
    original_width, original_height = image.size

    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 生成分割掩码
    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
        output_predictions = torch.argmax(output, dim=0).cpu().numpy()
    
    # 将分割掩码调整为原始图片的尺寸
    resized_mask = cv2.resize(output_predictions.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 创建颜色掩码
    mask_color = np.array([255, 0, 0], dtype=np.uint8)  # 红色
    color_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    color_mask[resized_mask > 0] = mask_color

    # 将原始图像转换为 numpy 数组
    image_np = np.array(image)

    # 检查通道数，如果是灰度图像，转换为 RGB
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # 确保图像和掩码的尺寸一致
    if image_np.shape[:2] != color_mask.shape[:2]:
        color_mask = cv2.resize(color_mask, (image_np.shape[1], image_np.shape[0]))

    # 叠加掩码到图像
    overlay = cv2.addWeighted(image_np, 1.0, color_mask, 0.4, 0)

    # 保存结果
    result_path = "segmentation_result.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, overlay_bgr)
    
    return result_path

# 整合函数
def process_image(image):
    class_result = classify_image(image)
    segmentation_result = segment_image(image)
    return class_result, segmentation_result

# Gradio 界面
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Image(type="filepath", label="Segmentation Result")
    ],
    title="Garbage Classification and Segmentation",
    description="Upload an image of garbage. The system will classify it and visualize the segmented result."
)

if __name__ == "__main__":
    interface.launch()
