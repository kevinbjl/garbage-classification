import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import colorsys
from skimage import measure
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
    resized_mask = cv2.resize(output_predictions, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 准备绘图
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Segmentation Result with Single Mask Color")

    # 固定掩码颜色（例如红色）
    mask_color = (1, 0, 0)  # RGB, 红色
    unique_classes = np.unique(resized_mask)

    # 遍历每个类别，绘制掩码轮廓
    for class_id in unique_classes:
        if class_id == 0:  # 跳过背景类别（假设 0 是背景）
            continue
        class_mask = (resized_mask == class_id)
        
        # 找到掩码的轮廓
        contours = measure.find_contours(class_mask.astype(np.float64), level=0.5)
        for contour in contours:
            poly = Polygon(contour, closed=True, facecolor=mask_color + (0.4,), edgecolor=mask_color, linewidth=2)
            ax.add_patch(poly)
    
    # 保存绘制结果到临时文件
    result_path = "segmentation_result.png"
    plt.tight_layout()
    plt.savefig(result_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return result_path

# 整合函数
def process_image(image):
    class_result = classify_image(image)
    segmentation_result = segment_image(image)
    return class_result, segmentation_result

# Gradio 界面
interface = gr.Interface(
    fn=process_image,
    inputs=gr.components.Image(type="pil"),
    outputs=[
        gr.components.Textbox(label="Classification Result"),
        gr.components.Image(type="filepath", label="Segmentation Result")
    ],
    title="Garbage Classification and Segmentation",
    description="Upload an image of garbage. The system will classify it and visualize the segmented result."
)

if __name__ == "__main__":
    interface.launch()
