# P2Txt - 点云可视化与智能分析工具

## 项目简介

P2Txt 是一个功能强大的点云数据可视化与智能分析工具，结合了计算机视觉技术和人工智能大模型，为点云数据处理提供了直观易用的图形界面。

## 主要功能

### 📊 点云数据可视化
- **多角度2D视图生成**：自动生成俯视图、前视图、侧视图
- **3D点云可视化**：支持Open3D库进行三维点云渲染
- **分类标签着色**：根据不同分类标签使用不同颜色显示
- **高质量图像输出**：生成高分辨率PNG格式的可视化图像

### 🤖 AI智能分析
- **视觉语言模型集成**：集成阿里云通义千问VL-Max模型
- **多模态分析**：结合图像和文本进行智能分析
- **自定义提示词**：支持用户自定义系统提示词和用户提示词
- **流式输出**：实时显示AI分析结果

### 🛠️ 数据处理工具
- **点云数据格式转换**：支持多种点云数据格式
- **标签数据处理**：处理和修正点云标签信息
- **批量处理**：支持批量处理多个点云文件

### 💻 用户界面
- **现代化GUI**：基于PyQt6构建的直观用户界面
- **多语言支持**：支持中英文界面切换
- **设置保存**：自动保存用户配置和API密钥
- **进度提示**：实时显示处理进度

## 支持的点云分类

项目预定义了以下点云分类标签：
- **Other (其他)** - 灰色 (#A9A9A9)
- **Buildings (建筑物)** - 红色 (#FF0000)
- **Trees (树木)** - 绿色 (#228B22)
- **Cars (汽车)** - 蓝色 (#0000FF)
- **Roads (道路)** - 黄色 (#FFFF00)
- **Poles (电线杆)** - 橙色 (#FFA500)

## 系统要求

### Python版本
- Python 3.8+

### 必需依赖
```
PyQt6
numpy
matplotlib
open3d
dashscope
```

## 安装指南

1. **克隆项目**
```bash
git clone https://github.com/your-username/P2Txt.git
cd P2Txt
```

2. **安装依赖**
```bash
pip install PyQt6 numpy matplotlib open3d dashscope
```

3. **运行程序**
```bash
python P2Txt_new.py
```

## 使用说明

### 基本使用流程

1. **加载点云文件**
   - 点击"选择文件"按钮选择点云数据文件（.txt格式）
   - 支持的数据格式：x y z intensity label（5列数据）

2. **生成可视化视图**
   - 程序自动生成三个角度的2D视图
   - 保存在`output_views`或`output_views_zh`目录中

3. **AI智能分析**
   - 配置阿里云API密钥
   - 设置分析提示词
   - 点击"分析"按钮获取AI分析结果

### 数据格式要求

点云数据文件应为文本格式，每行包含5列数据：
```
x_coordinate y_coordinate z_coordinate intensity label
```

示例：
```
1.234 5.678 9.012 0.8 1
2.345 6.789 0.123 0.6 2
3.456 7.890 1.234 0.9 0
```

## 项目结构

```
P2Txt/
├── P2Txt_new.py          # 主程序文件
├── orgtxt2txt.py         # 数据格式转换工具
├── label_process.py      # 标签数据处理工具
├── ico.png              # 程序图标
├── scene_1.txt          # 示例点云数据
├── output_views/        # 英文界面输出目录
├── output_views_zh/     # 中文界面输出目录
├── P2Txt_new.spec       # PyInstaller打包配置
└── README.md           # 项目说明文档
```

## 核心模块说明

### 点云加载模块 (`load_point_cloud`)
- 安全加载点云数据
- 数据完整性验证
- 错误处理和异常捕获

### 视图渲染模块 (`render_view`, `render_point_cloud_views`)
- 多角度2D视图生成
- 分类标签着色
- 高质量图像保存

### AI分析模块 (`ApiWorker`)
- 异步API调用
- 多模态数据处理
- 流式结果输出

### GUI界面模块 (`PointCloudAnalyzerApp`)
- 用户界面管理
- 事件处理
- 设置保存和加载

## 辅助工具

### 数据转换工具 (orgtxt2txt.py)
用于将原始点云数据转换为标准格式：
```bash
python orgtxt2txt.py
```

### 标签处理工具 (label_process.py)
用于处理和修正点云标签数据：
```bash
python label_process.py
```

## API配置

使用AI分析功能需要配置阿里云通义千问API：

1. 获取API密钥：访问[阿里云控制台](https://dashscope.console.aliyun.com/)
2. 在程序中输入API密钥
3. API密钥会自动保存，下次使用时无需重新输入

## 常见问题

### Q: 点云文件加载失败怎么办？
A: 请检查文件格式是否正确，确保每行包含5列数据（x y z intensity label）

### Q: 生成的视图图像在哪里？
A: 图像保存在`output_views`（英文界面）或`output_views_zh`（中文界面）目录中

### Q: AI分析功能无法使用？
A: 请确认已正确配置阿里云API密钥，并检查网络连接

### Q: 如何自定义分类标签？
A: 修改`P2Txt_new.py`文件中的`LABEL_COLORS`字典

## 致谢

感谢以下开源项目：
- [PyQt6](https://pypi.org/project/PyQt6/) - GUI框架
- [Open3D](https://github.com/isl-org/Open3D) - 3D数据处理
- [matplotlib](https://matplotlib.org/) - 数据可视化
- [numpy](https://numpy.org/) - 数值计算
- [阿里云通义千问](https://dashscope.aliyun.com/) - AI分析能力

---
