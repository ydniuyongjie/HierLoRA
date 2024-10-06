import os
import csv
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP ViT-L/14 模型
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# 加载 SDXL pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipeline = pipeline.to(device)

# 获取 SDXL 的文本编码器
text_encoder = pipeline.text_encoder
text_encoder_2 = pipeline.text_encoder_2
tokenizer = pipeline.tokenizer
tokenizer_2 = pipeline.tokenizer_2

def compute_similarity(image1_path, image2_path, text):
    # 加载和预处理图片
    image1 = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)
    
    # 处理文本
    text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    text_input_2 = tokenizer_2(text, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").to(device)
    
    # 计算特征
    with torch.no_grad():
        image1_features = clip_model.encode_image(image1)
        image2_features = clip_model.encode_image(image2)
        
        text_embeddings = text_encoder(**text_input).last_hidden_state
        text_embeddings_2 = text_encoder_2(**text_input_2).last_hidden_state
        
        # 使用 CLIP 的文本编码器来获得与图像特征相同维度的文本特征
        clip_text_features = clip_model.encode_text(clip.tokenize(text).to(device))
    
    # 归一化特征
    image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
    clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    image_similarity = torch.cosine_similarity(image1_features, image2_features).item()
    text_similarity1 = torch.cosine_similarity(image1_features, clip_text_features).item()
    text_similarity2 = torch.cosine_similarity(image2_features, clip_text_features).item()
    
    return image_similarity, text_similarity1, text_similarity2

# 图片目录和文本描述
image_dir = "layer_outputs"
all_layers_image = "all_layers.png"
text_description = "Red house, graffiti"  # 替换为实际使用的文本描述

# 获取所有层图片
layer_images = [f for f in os.listdir(image_dir) if f.endswith('.png') and f != all_layers_image]

# 计算相似度并保存结果
results = []
for layer_image in layer_images:
    layer_image_path = os.path.join(image_dir, layer_image)
    all_layers_path = os.path.join(image_dir, all_layers_image)
    
    image_similarity, layer_text_similarity, all_text_similarity = compute_similarity(
        layer_image_path, all_layers_path, text_description)
    
    results.append({
        'layer': layer_image.replace('.png', ''),
        'image_similarity': image_similarity,
        'layer_text_similarity': layer_text_similarity,
        'all_text_similarity': all_text_similarity
    })

# 保存结果为CSV
csv_file = 'similarity_results.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['layer', 'image_similarity', 'layer_text_similarity', 'all_text_similarity'])
    writer.writeheader()
    writer.writerows(results)

# 绘制结果图
layers = [r['layer'] for r in results]
image_similarities = [r['image_similarity'] for r in results]
layer_text_similarities = [r['layer_text_similarity'] for r in results]

plt.figure(figsize=(12, 6))
plt.plot(layers, image_similarities, label='Image Similarity with All Layers')
plt.plot(layers, layer_text_similarities, label='Text Similarity')
plt.xlabel('Layers')
plt.ylabel('Similarity Score')
plt.title('Image and Text Similarities Across Layers')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('similarity_plot.png')
plt.show()

print(f"Results saved to {csv_file}")
print("Plot saved as similarity_plot.png")
