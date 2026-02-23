import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import tempfile

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="二分类ResNet18",
    page_icon="🧠",
    layout="wide"
)

# ---------- 模型定义 ----------
class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        # 修改第一层为单通道输入
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改全连接层为二分类
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.model(x)

# ---------- 加载模型（带缓存）----------
@st.cache_resource
def load_model():
    """加载切分后的模型文件（自动添加model.前缀）"""
    
    # 1. 确定分片数量
    num_parts = 4  # 你用的是4份
    
    # 2. 加载所有分片
    state_dict = {}
    progress_bar = st.progress(0, text="正在加载模型分片...")
    
    for i in range(num_parts):
        filename = f'model_part_{i}.pth'
        if os.path.exists(filename):
            part = torch.load(filename, map_location='cpu')
            state_dict.update(part)
            progress_bar.progress((i + 1) / num_parts)
        else:
            st.error(f"找不到模型分片文件: {filename}")
            # 如果是分片3找不到，尝试继续
            if i == 3:
                st.warning("分片3未找到，尝试只加载前3个分片")
                num_parts = 3
    
    progress_bar.empty()
    
    # 3. 检查键名
    sample_keys = list(state_dict.keys())[:5]
    st.write(f"加载的权重键示例: {sample_keys}")
    
    # 4. 创建模型
    model = ResNet18Binary()
    model_keys = list(model.state_dict().keys())[:5]
    st.write(f"模型期望的键示例: {model_keys}")
    
    # 5. 关键修复：给所有键名加上 'model.' 前缀
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # 如果键名不以 'model.' 开头，就加上
        if not key.startswith('model.'):
            new_key = 'model.' + key
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    st.write(f"修复后的键示例: {list(fixed_state_dict.keys())[:5]}")
    
    # 6. 加载权重
    try:
        model.load_state_dict(fixed_state_dict)
        st.success("✅ 模型加载成功！")
    except Exception as e:
        st.error(f"加载失败: {e}")
        # 尝试非严格模式
        model.load_state_dict(fixed_state_dict, strict=False)
        st.warning("使用非严格模式加载，部分层可能未初始化")
    
    model.eval()
    return model

# ---------- 图像预处理 ----------
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 预测函数 ----------
def predict_images(uploaded_files, temperature):
    """批量预测图片"""
    
    if not uploaded_files:
        return None
    
    # 加载模型
    with st.spinner("🔄 正在加载模型..."):
        model = load_model()
    
    results = []
    progress_bar = st.progress(0, text="正在处理图片...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # 读取图片
            image = Image.open(uploaded_file).convert('L')
            
            # 预处理
            img_tensor = transform(image).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                outputs = model(img_tensor)
                logits = outputs[0]
                
                # 原始概率
                probs = torch.nn.functional.softmax(logits, dim=0)
                # 温度调整概率
                probs_temp = torch.nn.functional.softmax(logits / temperature, dim=0)
            
            # 记录结果
            results.append({
                '图片名称': uploaded_file.name,
                f'类别1概率(温度={temperature})': f'{float(probs_temp[1])*100:.2f}%',
                '类别1原始概率': f'{float(probs[1])*100:.2f}%',
                '类别1 logit': f'{float(logits[1]):.2f}',
                '类别0 logit': f'{float(logits[0]):.2f}',
                'logit差值': f'{abs(float(logits[0]) - float(logits[1])):.2f}',
                '预测结果': '类别1' if probs[1] > 0.5 else '类别0'
            })
            
        except Exception as e:
            results.append({
                '图片名称': uploaded_file.name,
                '错误': str(e)
            })
        
        # 更新进度
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# ---------- 页面UI ----------
st.title("🧠 二分类ResNet18 演示")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置参数")
    
    temperature = st.slider(
        "🌡️ 温度参数",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="温度>1.0使概率分布更平滑，<1.0更尖锐"
    )
    
    st.markdown("---")
    st.markdown("### 📁 模型文件状态")
    
    # 检查模型文件是否存在
    model_files_exist = all(
        os.path.exists(f'model_part_{i}.pth') for i in range(3)
    )
    
    if model_files_exist:
        st.success("✅ 模型文件已就绪")
        for i in range(3):
            size = os.path.getsize(f'model_part_{i}.pth') / (1024 * 1024)
            st.text(f"   model_part_{i}.pth: {size:.2f}MB")
    else:
        st.error("❌ 模型文件不存在，请上传")

# 主界面
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 上传图片")
    uploaded_files = st.file_uploader(
        "选择图片（可多选）",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="支持jpg、png等格式，可同时选择多张"
    )
    
    if uploaded_files:
        st.success(f"已选择 {len(uploaded_files)} 张图片")
        
        # 显示缩略图
        with st.expander("查看缩略图"):
            cols = st.columns(3)
            for i, file in enumerate(uploaded_files[:6]):  # 最多显示6张
                with cols[i % 3]:
                    image = Image.open(file)
                    st.image(image, caption=file.name, width=100)
        
        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在处理..."):
                results_df = predict_images(uploaded_files, temperature)
                
                if results_df is not None and not results_df.empty:
                    st.session_state['results'] = results_df
                    st.session_state['has_results'] = True
                else:
                    st.error("预测失败")

with col2:
    st.header("📊 预测结果")
    
    if 'has_results' in st.session_state and st.session_state['has_results']:
        # 显示结果表格
        st.dataframe(
            st.session_state['results'],
            use_container_width=True,
            hide_index=True
        )
        
        # 下载按钮
        csv = st.session_state['results'].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载结果CSV",
            data=csv,
            file_name="预测结果.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # 统计信息
        st.markdown("### 📈 统计摘要")
        col_a, col_b, col_c = st.columns(3)
        
        df = st.session_state['results']
        valid_results = df[df['错误'].isna() if '错误' in df.columns else pd.Series(True, index=df.index)]
        
        if not valid_results.empty:
            class1_count = len(valid_results[valid_results['预测结果'] == '类别1'])
            class0_count = len(valid_results) - class1_count
            
            with col_a:
                st.metric("总处理图片", len(valid_results))
            with col_b:
                st.metric("类别1数量", class1_count)
            with col_c:
                st.metric("类别0数量", class0_count)
    else:
        st.info("👈 请在左侧上传图片并开始预测")

st.markdown("---")
st.markdown("🚀 基于Streamlit + PyTorch，所有计算在服务器完成")
