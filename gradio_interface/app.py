import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr

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

# 定义图片预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MobileNet 推理函数
def classify_image(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = mobilenet(input_tensor)
        predicted_class = torch.argmax(output[0]).item()
    class_name = dct[predicted_class]
    return class_name

# DeepLab 推理函数
def segment_image(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
        output_predictions = torch.argmax(output, dim=0).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(output_predictions, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    return "segmentation_result.png"

# 定义整合函数 process_image
def process_image(image):
    # 调用 MobileNet 分类
    class_result = classify_image(image)
    
    # 调用 DeepLab 分割
    segmentation_result = segment_image(image)
    
    # 返回分类和分割结果
    return class_result, segmentation_result

# Gradio 界面
interface = gr.Interface(
    fn=process_image,
    inputs=gr.components.Image(type="pil"),
    outputs=[
        gr.components.Textbox(label="Classification Result"),
        gr.components.Image(type="filepath", label="Segmentation Result")  # 使用 filepath
    ],
    title="Garbage Classification and Segmentation",
    description="Upload an image of garbage. The system will classify it and visualize the segmented result."
)

if __name__ == "__main__":
    interface.launch()
