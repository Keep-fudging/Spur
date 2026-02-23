import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from torch import nn
from torchvision import models
import pandas as pd
import os
import tempfile


def load_my_model(model_path='best_model_fold9.pth'):
    # 方法：修改第一个卷积层，接受1通道输入
    model = models.resnet18(weights=None)

    # 关键修改：将第一层卷积改为接受1通道
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 修改全连接层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model


model = load_my_model()

# 对应的预处理：保持灰度图，不需要转3通道
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Grayscale(num_output_channels=1),  # 保持单通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图只有一个通道
])


def predict_single_image(img, temperature=1.0):
    """
    单张图片预测，返回详细结果（只关注类别1）
    """
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs[0]

        # 原始概率 (温度=1.0)
        original_probs = torch.nn.functional.softmax(logits, dim=0)

        # 温度调整后的概率
        tempered_probs = torch.nn.functional.softmax(logits / temperature, dim=0)

    return {
        # 只返回类别1相关的信息
        'prob1_original': float(original_probs[1]),  # 类别1的原始概率
        'prob1_tempered': float(tempered_probs[1]),  # 类别1的温度调整概率
        'logit1': float(logits[1]),  # 类别1的logit值
        'logit0': float(logits[0]),  # 保留类别0的logit用于计算差值
        'prob0_original': float(original_probs[0])  # 保留用于内部计算
    }


def predict_batch(images, temperature):
    """
    批量预测多张图片
    """
    if not images:
        return None, "请上传图片"

    results = []
    for i, img in enumerate(images):
        try:
            result = predict_single_image(img, temperature)
            result['图片序号'] = f"图片 {i + 1}"
            result['文件名'] = getattr(img, 'filename', f'image_{i + 1}.jpg') if hasattr(img,
                                                                                         'filename') else f'image_{i + 1}.jpg'
            # 计算类别1的置信度（如果原始概率>0.5则预测为类别1）
            result['prediction'] = '类别1' if result['prob1_original'] > 0.5 else '类别0'
            result['confidence'] = result['prob1_original'] if result['prob1_original'] > 0.5 else 1 - result[
                'prob1_original']
            results.append(result)
        except Exception as e:
            results.append({
                '图片序号': f"图片 {i + 1}",
                '文件名': f'image_{i + 1}.jpg',
                'prob1_original': 0,
                'prob1_tempered': 0,
                'logit1': 0,
                'logit0': 0,
                'prob0_original': 0,
                'prediction': '错误',
                'confidence': 0,
                'error': str(e)
            })

    return results, "批量预测完成"


def create_results_dataframe(results, temperature):
    """
    创建结果数据表格（只显示类别1相关信息）
    """
    df_data = []
    for r in results:
        if 'error' in r:
            row = {
                '图片': r['文件名'],
                f'类别1概率(温度={temperature})': 'N/A',
                '类别1原始概率': 'N/A',
                '类别1 logit值': 'N/A',
                '类别0 logit值': 'N/A',
                'logit差值': 'N/A',
                '预测结果': '错误',
                '置信度': 'N/A',
                '状态': f'错误: {r["error"]}'
            }
        else:
            # 计算logit差值
            logit_diff = abs(r['logit1'] - r['logit0'])

            row = {
                '图片': r['文件名'],
                f'类别1概率(温度={temperature})': f'{r["prob1_tempered"]:.4f} ({r["prob1_tempered"] * 100:.2f}%)',
                '类别1原始概率': f'{r["prob1_original"]:.4f} ({r["prob1_original"] * 100:.2f}%)',
                '类别1 logit值': f'{r["logit1"]:.2f}',
                '类别0 logit值': f'{r["logit0"]:.2f}',
                'logit差值': f'{logit_diff:.2f}',
                '预测结果': r['prediction'],
                '置信度': f'{r["confidence"] * 100:.2f}%'
            }
        df_data.append(row)

    return pd.DataFrame(df_data)


def export_results_to_csv(results_df):
    """
    导出结果为CSV文件
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as f:
        results_df.to_csv(f, index=False)
        return f.name


# 创建Gradio界面
with gr.Blocks(title="二分类ResNet18 - 批量图片分类系统") as demo:
    gr.Markdown("# 🖼️ 二分类ResNet18批量图片分类系统")
    gr.Markdown("上传多张56×56的图片，系统将批量分类并生成对比结果（重点关注**类别1**）。")

    with gr.Row():
        with gr.Column(scale=1):
            # 输入区域
            image_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="📤 上传多张图片 (可拖拽或点击选择)"
            )

            gr.Markdown("支持jpg、png等格式，多张图片批量处理")

            temperature_slider = gr.Slider(
                minimum=0.2,
                maximum=5.0,
                value=1.0,
                step=0.1,
                label="🌡️ 温度参数 (Temperature)"
            )
            gr.Markdown("温度=1.0: 原始输出 | 温度>1.0: 类别1概率更平滑")

            predict_btn = gr.Button("🚀 开始批量预测", variant="primary")

            # 状态显示
            status_text = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        with gr.Column(scale=2):
            # 结果表格 - 只显示类别1相关信息
            results_table = gr.Dataframe(
                label="📊 分类结果对比表 (类别1相关信息)",
                interactive=False,
                wrap=True
            )

    with gr.Row():
        with gr.Column():
            # 导出按钮
            export_btn = gr.Button("📥 导出结果为CSV", variant="secondary")
            download_file = gr.File(label="点击下载CSV文件")

    with gr.Row():
        with gr.Column():
            # 统计信息
            gr.Markdown("### 📈 批量统计信息 (类别1)")
            stats_text = gr.Textbox(
                label="统计摘要",
                value="等待预测...",
                interactive=False,
                lines=4
            )


    # 预测函数
    def batch_predict(files, temperature):
        if not files:
            return None, "请上传图片", "无数据", None

        # 加载图片
        images = []
        for file in files:
            try:
                if hasattr(file, 'name'):
                    img = Image.open(file.name).convert('L')
                    img.filename = os.path.basename(file.name)
                else:
                    img = Image.open(file).convert('L')
                    img.filename = os.path.basename(file)
                images.append(img)
            except Exception as e:
                print(f"加载图片失败: {e}")

        if not images:
            return None, "没有成功加载的图片", "无数据", None

        # 批量预测
        results, status = predict_batch(images, temperature)

        # 创建数据表格
        df = create_results_dataframe(results, temperature)

        # 生成统计信息（只关注类别1）
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            # 类别1的平均概率
            avg_prob1_original = sum(r['prob1_original'] for r in successful_results) / len(successful_results)
            avg_prob1_tempered = sum(r['prob1_tempered'] for r in successful_results) / len(successful_results)

            # 预测为类别1的数量
            class1_count = sum(1 for r in successful_results if r['prediction'] == '类别1')
            class0_count = len(successful_results) - class1_count

            # 平均logit值
            avg_logit1 = sum(r['logit1'] for r in successful_results) / len(successful_results)

            # 平均置信度
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)

            stats = f"✅ 成功处理: {len(successful_results)}/{len(results)} 张图片\n"
            stats += f"📊 类别1平均概率: {avg_prob1_original * 100:.2f}% (原始) | {avg_prob1_tempered * 100:.2f}% (温度={temperature})\n"
            stats += f"📊 类别1平均logit值: {avg_logit1:.2f}\n"
            stats += f"📈 平均置信度: {avg_confidence * 100:.2f}%\n"
            stats += f"📌 分类统计: 类别0={class0_count}张, 类别1={class1_count}张"
        else:
            stats = "❌ 没有成功处理的图片"

        return df, status, stats, None


    # 导出函数
    def export_results(df):
        if df is None or len(df) == 0:
            return None
        csv_path = export_results_to_csv(df)
        return csv_path


    # 绑定事件
    predict_btn.click(
        fn=batch_predict,
        inputs=[image_input, temperature_slider],
        outputs=[results_table, status_text, stats_text, download_file]
    )

    export_btn.click(
        fn=export_results,
        inputs=[results_table],
        outputs=[download_file]
    )

    # 添加使用说明
    gr.Markdown("---")
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 批量处理功能说明 (重点关注类别1)

        1. **上传图片**：
           - 点击上传区域选择多张图片，或直接拖拽图片到上传区域
           - 支持jpg、png、bmp等常见格式
           - 图片会自动转换为56×56灰度图

        2. **设置温度参数**：
           - 温度=1.0：模型的原始输出
           - 温度>1.0：类别1的概率分布更平滑

        3. **查看结果**：
           - **类别1原始概率**：模型对这张图片属于类别1的原始判断
           - **类别1温度调整概率**：经过温度平滑后的概率
           - **类别1 logit值**：模型输出的原始分数（正值倾向于类别1）
           - **类别0 logit值**：用于对比和计算差值
           - **logit差值**：绝对值越大，模型判断越确定

        4. **导出数据**：
           - 点击"导出结果为CSV"按钮下载CSV文件

        5. **批量统计**：
           - 显示类别1的平均概率和logit值
           - 各类别分布统计
        """)

# 启动应用
if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
