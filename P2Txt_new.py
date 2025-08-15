import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 忽略弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import open3d as o3d  # <<-- 添加Open3D库导入

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit,
    QTabWidget, QMessageBox, QSplitter, QProgressDialog,
    QComboBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QPalette, QColor, QIcon, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
# 新增 cast
from typing import Protocol, Any, cast

from pylab import mpl
# 设置matplotlib的中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

LABEL_COLORS = {
    0: ("Other", "#A9A9A9"), 1: ("Buildings", "#FF0000"), 2: ("Trees", "#228B22"),
    3: ("Cars", "#0000FF"), 4: ("Roads", "#FFFF00"), 5: ("Poles", "#FFA500"),
}

def load_point_cloud(file_path):
    try:
        # 尝试加载，允许文件为空或仅包含注释导致数据为空
        data = np.loadtxt(file_path)

        # 检查数据是否为空（例如文件只有注释或空白，loadtxt会跳过）
        if data.size == 0:
            # 如果loadtxt结果为空数组（如文件只有注释或为空）
            # 取决于np版本和文件内容，loadtxt可能报错或返回空
            # 这个显式检查处理返回空的情况
            raise ValueError(
                "点云文件中未找到数据。文件可能为空或仅包含注释。"
            )

        if data.ndim == 1:  # 处理文件中只有一个点的情况
            if data.shape[0] == 5:
                data = data.reshape(1, 5)
            else:
                raise ValueError(
                    f"文件中单行数据不是5列。实际为{data.shape[0]}列。应为x y z intensity label。")
        elif data.shape[1] != 5:
            raise ValueError(
                f"点云数据必须正好有5列（x y z intensity label）。实际为{data.shape[1]}列。")

        points = data[:, :3]  # x, y, z
        intensity = data[:, 3]  # 强度
        labels = data[:, 4].astype(int)  # 标签，确保为整数

        # 返回单独的x,y,z用于2D视图（兼容旧代码）
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]

        return x_coords, y_coords, z_coords, labels, intensity, points

    except ValueError as ve:  # 捕获loadtxt的特定错误或自定义ValueError
        print(f"加载点云出错（ValueError）：{ve}")
        raise  # 重新抛出以便调用方捕获
    except Exception as e:  # 捕获加载过程中的其他潜在错误
        print(f"加载点云时发生意外错误：{e}")
        raise  # 重新抛出以便调用方捕获


def render_view(x_coords, y_coords, labels, view_name, save_dir, i18n_texts):
    plt.figure(figsize=(10, 8))
    # unique_labels将是NumPy数组。如果labels为None或空，np.unique([])为np.array([])
    unique_labels = np.unique(labels) if labels is not None and labels.size > 0 else np.array([])

    for label_id in unique_labels:  # 如果unique_labels为空，此循环不会执行
        if label_id in LABEL_COLORS:
            label_name, color = LABEL_COLORS[label_id]
            mask = labels == label_id
            plt.scatter(x_coords[mask], y_coords[mask], c=color, s=1, label=label_name, marker='.')
        else:
            print(f"警告：标签ID {label_id} 不在LABEL_COLORS中，跳过。")

    # 如果没有找到唯一标签（如labels数组为空或所有标签未知）
    # 但有点存在，则用默认颜色绘制
    # 检查unique_labels是否为空
    if unique_labels.size == 0 and (x_coords is not None and x_coords.size > 0):
        plt.scatter(x_coords, y_coords, c=LABEL_COLORS[0][1], s=1, label=LABEL_COLORS[0][0], marker='.')

    plt.axis('equal')
    plt.title(f'{view_name} {i18n_texts["view_title_suffix"]}')
    plt.xlabel('X')
    plt.ylabel('Y')
    handles, legend_labels = [], []
    sorted_label_colors = sorted(LABEL_COLORS.items())

    # 根据实际唯一标签生成图例，或如果没有特定标签但有点则显示默认图例
    for label_id, (label_name, color) in sorted_label_colors:
        # 条件：label_id在数据中出现的唯一标签中
        # 或唯一标签数组为空且为默认标签（0），且有点可显示默认图例
        if label_id in unique_labels or \
                (unique_labels.size == 0 and label_id == 0 and (x_coords is not None and x_coords.size > 0)):
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=label_name, markerfacecolor=color, markersize=10))
            legend_labels.append(label_name)

    if handles:
        plt.legend(handles, legend_labels, markerscale=1, fontsize=8, loc='upper right', frameon=True)
    else:
        # 如果x_coords为空，或没有标签匹配LABEL_COLORS，可能会出现这种情况
        print(f"{view_name}中未找到已知标签或无数据用于图例。")
    plt.tight_layout()
    file_path = os.path.join(save_dir, f'{view_name.lower().replace(" ", "_")}_view.png')
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"{i18n_texts['saved_view_message']}: {view_name}")
    return file_path


def render_point_cloud_views(file_path, save_dir, i18n_texts):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    x, y, z, label_data, _, _points_for_3d = load_point_cloud(file_path)
    paths = {}
    # 仅当label_data不为None且不为空时渲染2D视图
    if label_data is not None and label_data.size > 0:
        paths['top'] = render_view(x, y, label_data, i18n_texts['top_view_name'], save_dir, i18n_texts)
        paths['front'] = render_view(x, z, label_data, i18n_texts['front_view_name'], save_dir, i18n_texts)
        paths['side'] = render_view(y, z, label_data, i18n_texts['side_view_name'], save_dir, i18n_texts)
        print(f"{i18n_texts['render_complete_message']} {save_dir}")
    else:
        # 该消息也适用于load_point_cloud返回空label_data数组的情况
        print("点云加载无有效标签或无数据点，跳过二维视图生成。")
    return paths


# 新增：信号协议，帮助类型检查器识别 emit/connect
class SignalLike(Protocol):
    def connect(self, slot: Any) -> Any: ...
    def emit(self, *args: Any, **kwargs: Any) -> None: ...


class ApiWorker(QThread):  # 确保这是健壮版本
    # 为信号添加类型注解，消除“在 'pyqtSignal | pyqtSignal' 中找不到引用 'emit'”告警
    result_ready: SignalLike = pyqtSignal(str)
    error_occurred: SignalLike = pyqtSignal(str)
    # 移除对子类 finished 的重新声明，使用基类 QThread.finished

    def __init__(self, api_key, image_paths, system_prompt_text, user_prompt_text):
        super().__init__()
        self.api_key = api_key
        self.image_paths = image_paths  # 这是一个字典
        self.system_prompt_text = system_prompt_text
        self.user_prompt_text = user_prompt_text

    def run(self):
        import dashscope
        from dashscope.api_entities.dashscope_response import Role

        try:
            content_for_api = []
            # 检查image_paths字典是否为空。如果2D视图被跳过会出现这种情况
            if not self.image_paths:
                self.error_occurred.emit(
                    "无可用于VLM分析的二维视图（点云可能无标签、无数据或视图生成失败）。")
                self.finished.emit()
                return

            for view_type in ["front", "side", "top"]:
                path = self.image_paths.get(view_type)
                if path and os.path.exists(path):
                    content_for_api.append({"image": f"file://{os.path.abspath(path)}"})
                else:
                    # 这种情况意味着image_paths已填充但某个路径缺失/无效
                    # 如果render_view成功，这种情况一般不会发生
                    # 若视图未生成，前面的if not self.image_paths会捕获
                    self.error_occurred.emit(
                        f"{view_type}视图的图片路径缺失或无效，无法进行VLM分析。路径: {path}")
                    self.finished.emit()
                    return

            # 确保确实有图片内容，如果走到这里
            # 如果上面的循环确保所有图片都存在，这里检查可能是多余的
            if not any("image" in item for item in content_for_api):
                self.error_occurred.emit("检查路径后未找到可用于VLM分析的有效图片内容。")
                self.finished.emit()
                return

            content_for_api.append({"text": self.user_prompt_text})
            messages = [{"role": Role.SYSTEM, "content": [{"text": self.system_prompt_text}]},
                        {"role": Role.USER, "content": content_for_api}]
            responses = dashscope.MultiModalConversation.call(api_key=self.api_key, model='qwen-vl-max',
                                                              messages=messages, stream=True, incremental_output=True)
            for response in responses:
                if response.status_code == 200:
                    text_content = ""
                    if response.output and response.output.choices and len(response.output.choices) > 0:
                        choice = response.output.choices[0]
                        if choice.message and choice.message.content and len(choice.message.content) > 0:
                            for part in choice.message.content:
                                if "text" in part: text_content += part.get("text", "")
                    if text_content: self.result_ready.emit(text_content)
                else:
                    error_detail = f"Code: {response.code}, Message: {response.message}"
                    if hasattr(response, 'request_id'): error_detail += f", Request ID: {response.request_id}"
                    self.error_occurred.emit(f"Dashscope API错误: {error_detail}")
                    return
        except ImportError:
            self.error_occurred.emit("未安装Dashscope SDK。请安装：pip install dashscope")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Dashscope API调用异常：{e}\n{error_trace}")
            self.error_occurred.emit(f"Dashscope API调用处理失败: {str(e)}")
        finally:
            self.finished.emit()


MORANDI_LIGHT = {
    "bg": "#EAE0D5", "bg_alt": "#DCD0C0", "fg": "#5D5C61", "accent": "#B6A693",
    "button": "#C9B7A8", "button_fg": "#4A4A48", "border": "#A99A8D", "text_area_bg": "#F5F5F5",
}
MORANDI_DARK = {
    "bg": "#3C3F41", "bg_alt": "#4A4E51", "fg": "#E0E0E0", "accent": "#6E7F80",
    "button": "#5A5E61", "button_fg": "#E0E0E0", "border": "#2A2D2F", "text_area_bg": "#2B2B2B",
}

I18N_TEXTS = {
    "en": {
        "window_title": "Point Cloud Scene Analyzer", "load_button": "Load Point Cloud (.txt)",
        "analyze_button": "Analyze Scene with VLM", "clear_button": "Clear Output & Views",
        "dark_mode_button": "Switch to Dark Mode", "light_mode_button": "Switch to Light Mode",
        "api_key_label": "Dashscope API Key:", "output_label": "VLM Analysis:",
        "views_tab_label": "Generated Views",
        "threed_view_tab_label": "3D View",
        "launch_3d_button": "Launch 3D Viewer",
        "threed_placeholder": "Load a point cloud and click 'Launch 3D Viewer'.",
        "status_ready": "Ready. Load a point cloud file.",
        "status_loading_file": "Loading file: {file_path}", "status_generating_views": "Generating 2D views...",
        "status_views_generated": "2D views generated. Ready for analysis or 3D view.",
        "status_views_skipped": "Point cloud loaded. 2D views skipped (no labels/data). Ready for 3D view or analysis if applicable.",
        # New/Updated
        "status_analyzing": "Analyzing scene with VLM... Please wait.",
        "status_analysis_complete": "Analysis complete.",
        "status_analysis_partial": "Receiving analysis...", "error_title": "Error",
        "error_no_file": "Please load a point cloud file first.",
        "error_no_views": "Please generate views first (load a file).",
        "error_no_api_key": "Please enter your Dashscope API Key.",
        "error_loading_point_cloud": "Failed to load point cloud: {error}",
        "error_generating_views": "Failed to generate 2D views: {error}",
        "error_loading_image": "Error loading image for display.",
        "error_launching_3d": "Failed to launch 3D viewer: {error}",
        "output_dir_name": "output_views", "top_view_name": "Top", "front_view_name": "Front", "side_view_name": "Side",
        "view_title_suffix": "View", "saved_view_message": "Saved",
        "render_complete_message": "2D view rendering complete, images saved in:",
        "language_select_label": "Language:",
        "views_placeholder": "Load a point cloud to generate 2D views (requires labels/data).",  # Modified
        "system_prompt": """You are a helpful AI assistant specializing in point cloud scene understanding. Given three orthogonal 2D projected views (top, front, side) of a 3D point cloud scene, describe the scene in detail. Identify major objects, their spatial relationships, and the overall environment type if possible. Be concise and informative.""",
        "user_prompt": """Please analyze these three views of a point cloud scene and provide a comprehensive description."""
    },
    "zh": {
        "window_title": "点云场景分析器", "load_button": "加载点云文件 (.txt)",
        "analyze_button": "调用VLM分析场景", "clear_button": "清除输出和视图",
        "dark_mode_button": "切换深色模式", "light_mode_button": "切换浅色模式",
        "api_key_label": "Dashscope API 密钥:", "output_label": "VLM分析结果:",
        "views_tab_label": "生成的视图",
        "threed_view_tab_label": "三维视图",
        "launch_3d_button": "启动三维查看器",
        "threed_placeholder": "加载点云后，点击“启动三维查看器”。",
        "status_ready": "就绪。请加载点云文件。",
        "status_loading_file": "正在加载文件: {file_path}", "status_generating_views": "正在生成二维视图...",
        "status_views_generated": "二维视图已生成。可以开始分析或查看三维视图。",
        "status_views_skipped": "点云已加载。二维视图已跳过（无标签/数据）。可进行三维查看或VLM分析（若适用）。",
        # New/Updated
        "status_analyzing": "正在调用VLM分析场景... 请稍候。",
        "status_analysis_complete": "分析完成。", "status_analysis_partial": "正在接收分析结果...",
        "error_title": "错误", "error_no_file": "请先加载点云文件。",
        "error_no_views": "请先生成视图 (加载文件)。", "error_no_api_key": "请输入您的Dashscope API密钥。",
        "error_loading_point_cloud": "加载点云失败: {error}", "error_generating_views": "生成二维视图失败: {error}",
        "error_loading_image": "错误：无法加载图片用于显示。",
        "error_launching_3d": "启动三维查看器失败: {error}",
        "output_dir_name": "output_views_zh", "top_view_name": "俯视图", "front_view_name": "正视图",
        # output_dir_name changed for zh
        "side_view_name": "侧视图",
        "view_title_suffix": "视图", "saved_view_message": "已保存",
        "render_complete_message": "二维视图渲染完成，图像保存在：",
        "language_select_label": "语言:", "views_placeholder": "加载点云以生成二维视图（需要标签/数据）。",  # Modified
        "system_prompt": """你是一个精通点云场景理解的AI助手。给定一个三维点云场景的三个正交二维投影视图（俯视图、正视图、侧视图），请详细描述这个场景。识别主要的物体，它们的空间关系，如果可能的话，判断整体环境类型。请做到简洁且信息丰富。""",
        "user_prompt": """请分析这三张点云场景的视图，并提供一个全面的描述。"""
    }
}

class PointCloudAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "PointCloudAnalyzer")
        self.point_cloud_file = None
        self.loaded_point_cloud_data_for_3d = None
        self.generated_view_paths = {}
        self.original_pixmaps = {}
        self.raw_vlm_output_buffer = ""

        self.load_settings()  # 这会设置self.i18n
        # output_views_dir应在i18n加载后设置
        self.output_views_dir = os.path.join(os.getcwd(), self.i18n.get("output_dir_name", "output_views"))

        self.initUI()

        if not os.path.exists(self.output_views_dir):
            os.makedirs(self.output_views_dir, exist_ok=True)

    def load_settings(self):
        self.setWindowIcon(QIcon("ico.png"))
        self.current_lang = self.settings.value("language", "zh")
        self.i18n = I18N_TEXTS[self.current_lang]
        theme_name = self.settings.value("theme", "light")
        self.current_theme = MORANDI_DARK if theme_name == "dark" else MORANDI_LIGHT
        self.api_key_input_default = self.settings.value("api_key", "")

    def save_settings(self):
        self.settings.setValue("language", self.current_lang)
        self.settings.setValue("theme", "dark" if self.current_theme == MORANDI_DARK else "light")
        if hasattr(self, 'api_key_input'):
            self.settings.setValue("api_key", self.api_key_input.text())

    def initUI(self):
        self.setGeometry(100, 100, 1200, 700)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        top_controls_layout = QHBoxLayout()
        self.lang_label_widget = QLabel()
        top_controls_layout.addWidget(self.lang_label_widget)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("中文", "zh")
        self.lang_combo.addItem("English", "en")
        current_idx = self.lang_combo.findData(self.current_lang)
        if current_idx != -1: self.lang_combo.setCurrentIndex(current_idx)
        # 原：self.lang_combo.currentIndexChanged.connect(self.change_language)
        cast(SignalLike, self.lang_combo.currentIndexChanged).connect(self.change_language)
        top_controls_layout.addWidget(self.lang_combo)
        self.api_key_label = QLabel()
        top_controls_layout.addWidget(self.api_key_label)
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-xxxxxxxxxxxx")
        if hasattr(self, 'api_key_input_default'): self.api_key_input.setText(self.api_key_input_default)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        top_controls_layout.addWidget(self.api_key_input)
        self.dark_mode_button = QPushButton()
        # 原：self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        cast(SignalLike, self.dark_mode_button.clicked).connect(self.toggle_dark_mode)
        top_controls_layout.addWidget(self.dark_mode_button)
        top_controls_layout.addStretch(1)
        main_layout.addLayout(top_controls_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        self.view_tabs = QTabWidget()

        initial_2d_placeholder = self.i18n.get("views_placeholder", "Load a point cloud to generate views.")
        self.top_view_label = QLabel(initial_2d_placeholder)
        self.top_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top_view_label.setScaledContents(True)
        self.top_view_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.front_view_label = QLabel(initial_2d_placeholder)
        self.front_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_view_label.setScaledContents(True)
        self.front_view_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.side_view_label = QLabel(initial_2d_placeholder)
        self.side_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_view_label.setScaledContents(True)
        self.side_view_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.view_tabs.addTab(self.top_view_label, "")
        self.view_tabs.addTab(self.front_view_label, "")
        self.view_tabs.addTab(self.side_view_label, "")

        self.threed_view_widget = QWidget()
        threed_view_layout = QVBoxLayout(self.threed_view_widget)
        threed_view_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.threed_placeholder_label = QLabel(
            self.i18n.get("threed_placeholder", "Load point cloud and click 'Launch 3D Viewer'"))
        self.threed_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threed_view_layout.addWidget(self.threed_placeholder_label)

        self.launch_3d_button = QPushButton()
        # 原：self.launch_3d_button.clicked.connect(self.launch_3d_viewer_action)
        cast(SignalLike, self.launch_3d_button.clicked).connect(self.launch_3d_viewer_action)
        self.launch_3d_button.setEnabled(False)
        threed_view_layout.addWidget(self.launch_3d_button, alignment=Qt.AlignmentFlag.AlignCenter)
        threed_view_layout.addStretch(1)

        self.view_tabs.addTab(self.threed_view_widget, "")

        left_layout.addWidget(self.view_tabs)
        self.splitter.addWidget(left_pane)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        self.output_label = QLabel()
        right_layout.addWidget(self.output_label)
        self.api_output_text = QTextEdit()
        self.api_output_text.setReadOnly(True)
        right_layout.addWidget(self.api_output_text)
        self.splitter.addWidget(right_pane)
        self.splitter.setSizes([self.width() // 2, self.width() // 2])
        main_layout.addWidget(self.splitter)

        bottom_buttons_layout = QHBoxLayout()
        self.load_button = QPushButton()
        # 原：self.load_button.clicked.connect(self.load_point_cloud_action)
        cast(SignalLike, self.load_button.clicked).connect(self.load_point_cloud_action)
        bottom_buttons_layout.addWidget(self.load_button)
        self.analyze_button = QPushButton()
        # 原：self.analyze_button.clicked.connect(self.analyze_scene_action)
        cast(SignalLike, self.analyze_button.clicked).connect(self.analyze_scene_action)
        self.analyze_button.setEnabled(False)
        bottom_buttons_layout.addWidget(self.analyze_button)
        self.clear_button = QPushButton()
        # 原：self.clear_button.clicked.connect(self.clear_all_action)
        cast(SignalLike, self.clear_button.clicked).connect(self.clear_all_action)
        bottom_buttons_layout.addWidget(self.clear_button)
        main_layout.addLayout(bottom_buttons_layout)

        self.apply_theme()
        self.update_language_ui()
        self.statusBar().showMessage(self.i18n["status_ready"])

    def update_language_ui(self):
        self.i18n = I18N_TEXTS[self.current_lang]
        self.setWindowTitle(self.i18n["window_title"])
        self.lang_label_widget.setText(self.i18n["language_select_label"])
        self.api_key_label.setText(self.i18n["api_key_label"])
        if self.current_theme == MORANDI_LIGHT:
            self.dark_mode_button.setText(self.i18n["dark_mode_button"])
        else:
            self.dark_mode_button.setText(self.i18n["light_mode_button"])

        self.view_tabs.setTabText(0, self.i18n["top_view_name"])
        self.view_tabs.setTabText(1, self.i18n["front_view_name"])
        self.view_tabs.setTabText(2, self.i18n["side_view_name"])
        self.view_tabs.setTabText(3, self.i18n["threed_view_tab_label"])

        placeholder_2d_text = self.i18n.get("views_placeholder", "Load a point cloud to generate views.")
        if self.top_view_label.pixmap() is None or self.top_view_label.pixmap().isNull(): self.top_view_label.setText(
            placeholder_2d_text)
        if self.front_view_label.pixmap() is None or self.front_view_label.pixmap().isNull(): self.front_view_label.setText(
            placeholder_2d_text)
        if self.side_view_label.pixmap() is None or self.side_view_label.pixmap().isNull(): self.side_view_label.setText(
            placeholder_2d_text)

        if hasattr(self, 'threed_placeholder_label'):
            self.threed_placeholder_label.setText(self.i18n.get("threed_placeholder", "Load point cloud..."))
        if hasattr(self, 'launch_3d_button'):
            self.launch_3d_button.setText(self.i18n.get("launch_3d_button", "Launch 3D Viewer"))

        self.output_label.setText(self.i18n["output_label"])
        self.load_button.setText(self.i18n["load_button"])
        self.analyze_button.setText(self.i18n["analyze_button"])
        self.clear_button.setText(self.i18n["clear_button"])

        # 更新状态栏文本，如果当前显示的是通用的“准备就绪”消息
        current_status_key_stem = self.statusBar().currentMessage()  # 获取当前消息
        # 检查当前状态是否为所有语言中任何一个“status_ready”消息
        is_status_ready = any(
            current_status_key_stem == I18N_TEXTS[lang_key]["status_ready"] for lang_key in I18N_TEXTS)
        if is_status_ready:
            self.statusBar().showMessage(self.i18n["status_ready"])

        # 根据新语言更新output_views_dir
        self.output_views_dir = os.path.join(os.getcwd(), self.i18n.get("output_dir_name", "output_views"))
        if not os.path.exists(self.output_views_dir):
            os.makedirs(self.output_views_dir, exist_ok=True)

    def change_language(self, index: int):
        selected_lang_code = self.lang_combo.itemData(index)
        if selected_lang_code and selected_lang_code != self.current_lang:
            self.current_lang = selected_lang_code
            # self.i18n is updated in update_language_ui
            self.update_language_ui()  # 这将更新i18n和output_views_dir
            self.apply_theme()  # 重新应用主题，以防按钮文本需要特定主题颜色
            self.save_settings()

    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self.current_theme["bg"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self.current_theme["fg"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(self.current_theme["text_area_bg"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(self.current_theme["bg_alt"]))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(self.current_theme["bg"]))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(self.current_theme["fg"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(self.current_theme["fg"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(self.current_theme["button"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(self.current_theme["button_fg"]))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#ff0000"))
        palette.setColor(QPalette.ColorRole.Link, QColor(self.current_theme["accent"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.current_theme["accent"]))
        palette.setColor(QPalette.ColorRole.HighlightedText,
                         QColor(self.current_theme["bg"]))  # Text color for highlighted items

        # 调整高亮文本颜色，以便在强调背景上有更好的对比度
        q_accent_color_for_highlight = QColor(self.current_theme["accent"])
        if q_accent_color_for_highlight.lightnessF() < 0.5:  # 深色强调
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(MORANDI_LIGHT["fg"]))  # 浅色文本
        else:  # 浅色强调
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(MORANDI_DARK["fg"]))  # 深色文本

        self.setPalette(palette)
        # QApplication.instance() 在类型检查器中被视为 QCoreApplication，显式收窄为 QApplication 以调用 setPalette
        app_inst = QApplication.instance()
        if app_inst is not None:
            cast(QApplication, app_inst).setPalette(palette)  # 应用到整个应用程序

        # 样式表以获得更详细的控制
        fg_color = self.current_theme["fg"]
        bg_color = self.current_theme["bg"]
        bg_alt_color = self.current_theme["bg_alt"]
        text_area_bg_color = self.current_theme["text_area_bg"]
        button_color = self.current_theme["button"]
        button_fg_color = self.current_theme["button_fg"]
        border_color = self.current_theme["border"]
        accent_color = self.current_theme["accent"]

        q_accent_color = QColor(accent_color)
        pressed_button_bg = q_accent_color.darker(120).name() if q_accent_color.isValid() else accent_color

        q_fg_color_obj = QColor(fg_color)
        disabled_fg_color = q_fg_color_obj.darker(
            150).name() if self.current_theme == MORANDI_LIGHT else q_fg_color_obj.lighter(150).name()
        if not q_fg_color_obj.isValid(): disabled_fg_color = fg_color  # fallback

        disabled_bg_color = QColor(button_color).lighter(110).name() if self.current_theme == MORANDI_LIGHT else QColor(
            button_color).darker(110).name()
        if not QColor(button_color).isValid(): disabled_bg_color = bg_alt_color  # fallback

        highlighted_text_color_on_accent_str = palette.color(QPalette.ColorRole.HighlightedText).name()

        common_style = f"""
            QMainWindow {{ background-color: {bg_color} }}
            QWidget {{ color: {fg_color} font-size: 10pt }}
            QFrame {{ background-color: {bg_color} }}
            QLabel {{ color: {fg_color} background-color: transparent }}
            QPushButton {{
                background-color: {button_color} color: {button_fg_color}
                border: 1px solid {border_color} padding: 5px 10px min-height: 20px border-radius: 3px
            }}
            QPushButton:hover {{ background-color: {accent_color} }}
            QPushButton:pressed {{ background-color: {pressed_button_bg} }}
            QPushButton:disabled {{
                background-color: {disabled_bg_color} color: {disabled_fg_color}
                border: 1px solid {QColor(border_color).darker(110).name() if QColor(border_color).isValid() else border_color}
            }}
            QLineEdit, QTextEdit {{
                background-color: {text_area_bg_color} color: {fg_color}
                border: 1px solid {border_color} padding: 3px border-radius: 3px
                selection-background-color: {accent_color} selection-color: {highlighted_text_color_on_accent_str}
            }}
            QTabWidget::pane {{
                border: 1px solid {border_color} background-color: {bg_alt_color}
                border-top-right-radius: 3px border-bottom-left-radius: 3px border-bottom-right-radius: 3px
            }}
            QTabWidget > QWidget {{ background-color: {bg_alt_color} }} /* Content area of tabs */
            QTabWidget > QWidget > QLabel {{ background-color: {bg_alt_color} }} /* Ensure labels inside tab content also get this bg */
            QTabBar::tab {{
                background: {button_color} color: {button_fg_color}
                border: 1px solid {border_color} border-bottom: none /* Crucial for selected tab look */
                padding: 6px 12px
                border-top-left-radius: 3px border-top-right-radius: 3px
            }}
            QTabBar::tab:selected {{
                background: {bg_alt_color} /* Match pane background */
                color: {fg_color} /* Ensure text is visible */
                /* border-bottom-color: {bg_alt_color}  Optional: make bottom border blend with pane */
            }}
            QTabBar::tab:!selected:hover {{ background: {accent_color} }}
            QSplitter::handle {{ background-color: {border_color} }}
            QSplitter::handle:horizontal {{ height: 3px margin: 1px 0 }}
            QSplitter::handle:vertical {{ width: 3px margin: 0 1px }}
            QComboBox {{
                border: 1px solid {border_color} padding: 3px 5px
                background-color: {button_color} color: {button_fg_color} border-radius: 3px
                selection-background-color: {accent_color} selection-color: {highlighted_text_color_on_accent_str}
            }}
            QComboBox:editable {{ background: {text_area_bg_color} }}
            QComboBox::drop-down {{
                subcontrol-origin: padding subcontrol-position: top right width: 20px
                border-left-width: 1px border-left-color: {border_color} border-left-style: solid
                border-top-right-radius: 3px border-bottom-right-radius: 3px
            }}
            QComboBox QAbstractItemView {{ /* Dropdown list */
                border: 1px solid {border_color}
                background-color: {text_area_bg_color} color: {fg_color}
                selection-background-color: {accent_color} selection-color: {highlighted_text_color_on_accent_str}
            }}
            QStatusBar {{ background-color: {bg_alt_color} color: {fg_color} }}
            QMessageBox, QProgressDialog {{ background-color: {bg_color} color: {fg_color} }}
            QMessageBox QLabel, QProgressDialog QLabel {{ color: {fg_color} background-color: transparent }}
        """
        self.setStyleSheet(common_style)

        # 更新按钮文本在主题更改后，因为它依赖于i18n
        if hasattr(self, 'dark_mode_button') and hasattr(self, 'i18n'):
            if self.current_theme == MORANDI_LIGHT:
                self.dark_mode_button.setText(self.i18n.get("dark_mode_button", "Switch to Dark Mode"))
            else:
                self.dark_mode_button.setText(self.i18n.get("light_mode_button", "Switch to Light Mode"))

        # 如果需要，强制重绘/更新相关小部件
        if hasattr(self, 'view_tabs'): self.view_tabs.update()
        if hasattr(self, 'top_view_label'): self.top_view_label.update()
        if hasattr(self, 'front_view_label'): self.front_view_label.update()
        if hasattr(self, 'side_view_label'): self.side_view_label.update()
        if hasattr(self, 'threed_view_widget'):
            self.threed_view_widget.setStyleSheet(f"background-color: {self.current_theme['bg_alt']}")
            self.threed_placeholder_label.setPalette(palette)  # ensure text color updates

    def toggle_dark_mode(self, checked: bool = False):
        if self.current_theme == MORANDI_LIGHT:
            self.current_theme = MORANDI_DARK
        else:
            self.current_theme = MORANDI_LIGHT
        self.apply_theme()
        self.save_settings()

    def load_point_cloud_action(self, checked: bool = False):
        file_path, _ = QFileDialog.getOpenFileName(self, self.i18n["load_button"], "",
                                                   "Text Files (*.txt)All Files (*)")
        if file_path:
            self.point_cloud_file = file_path
            self.statusBar().showMessage(
                self.i18n["status_loading_file"].format(file_path=os.path.basename(file_path)))
            QApplication.processEvents()

            progress_text = self.i18n["status_generating_views"]
            progress = QProgressDialog(progress_text, self.i18n.get("cancel_button", "Cancel"), 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle(self.i18n.get("progress_title", "Processing"))
            progress.show()
            QApplication.processEvents()

            try:
                _x, _y, _z, labels_data, _intensity_data, points_for_3d = load_point_cloud(self.point_cloud_file)
                self.loaded_point_cloud_data_for_3d = (points_for_3d, labels_data)  # Store for 3D viewer

                # 检查是否加载了点以进行3D视图
                can_launch_3d = points_for_3d is not None and points_for_3d.size > 0
                self.launch_3d_button.setEnabled(can_launch_3d)

                # 生成2D视图。 render_point_cloud_views现在处理空的labels_data。
                self.generated_view_paths = render_point_cloud_views(
                    self.point_cloud_file, self.output_views_dir, self.i18n
                )
                self._load_and_display_original_views()

                # 仅当实际生成了2D视图时启用分析按钮
                can_analyze = bool(self.generated_view_paths)
                self.analyze_button.setEnabled(can_analyze)

                if can_analyze:  # 生成了2D视图
                    status_msg = self.i18n["status_views_generated"]
                elif can_launch_3d:  # 没有2D视图，但有3D数据
                    status_msg = self.i18n["status_views_skipped"]
                else:  # 没有2D视图和3D数据（应由load_point_cloud错误捕获）
                    status_msg = self.i18n["status_ready"]  # 或更具体的错误状态

                progress.close()
                self.api_output_text.clear()
                self.raw_vlm_output_buffer = ""
                self.statusBar().showMessage(status_msg)

            except ValueError as ve:
                progress.close()
                QMessageBox.critical(self, self.i18n["error_title"],
                                     self.i18n["error_loading_point_cloud"].format(error=str(ve)))
                self.statusBar().showMessage(self.i18n["status_ready"])
                self.clear_all_action()  # 清除状态并禁用按钮
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, self.i18n["error_title"],
                                     self.i18n.get("error_processing_file",
                                                   "Error processing file: {error}").format(error=str(e)))
                self.statusBar().showMessage(self.i18n["status_ready"])
                self.clear_all_action()

    def launch_3d_viewer_action(self, checked: bool = False):
        if not self.loaded_point_cloud_data_for_3d or \
                self.loaded_point_cloud_data_for_3d[0] is None or \
                self.loaded_point_cloud_data_for_3d[0].size == 0:  # 检查点数组是否为空
            QMessageBox.warning(self, self.i18n["error_title"],
                                self.i18n["error_no_file"])  # 或更具体的“未加载点”
            return

        points, labels = self.loaded_point_cloud_data_for_3d

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if labels is not None and labels.size > 0:  # 检查标签数组是否不为空
                label_colors_int_keys = {int(k): QColor(v[1]).getRgbF()[:3] for k, v in LABEL_COLORS.items()}
                default_color_rgb = QColor(LABEL_COLORS[0][1]).getRgbF()[:3] if 0 in LABEL_COLORS else (0.5, 0.5, 0.5)

                # 确保标签数组长度与点数组相同（如果用于着色）
                if len(labels) == len(points):
                    colors = np.array([label_colors_int_keys.get(label, default_color_rgb) for label in labels])
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    print(
                        f"警告: 点数 ({len(points)}) 与标签数 ({len(labels)}) 不匹配。使用默认颜色进行3D视图。")
                    pcd.paint_uniform_color(default_color_rgb)
            else:
                default_color_rgb = QColor(LABEL_COLORS[0][1]).getRgbF()[:3] if 0 in LABEL_COLORS else (0.5, 0.5, 0.5)
                pcd.paint_uniform_color(default_color_rgb)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=self.i18n.get("threed_view_tab_label", "3D Point Cloud Viewer"), width=800,
                              height=600)
            # 将 PointCloud 显式收窄为可接受类型，避免“应为 Geometry.py，实际为 PointCloud”的类型告警
            vis.add_geometry(cast(Any, pcd))
            opt = vis.get_render_option()
            opt.background_color = np.asarray(QColor(self.current_theme["bg_alt"]).getRgbF()[:3])  # Match theme
            opt.point_size = 2.0  # Adjust point size if needed
            vis.run()
            vis.destroy_window()

        except Exception as e:
            QMessageBox.critical(self, self.i18n["error_title"], self.i18n["error_launching_3d"].format(error=str(e)))
            print(f"Error in 3D viewer: {e}")
            import traceback
            traceback.print_exc()

    def _load_and_display_original_views(self):
        self.original_pixmaps.clear()
        placeholder_text = self.i18n.get("views_placeholder", "View not available.")
        error_img_text = self.i18n.get("error_loading_image", "Error loading image.")
        paths_and_labels = [(self.generated_view_paths.get('top'), self.top_view_label, 'top'),
                            (self.generated_view_paths.get('front'), self.front_view_label, 'front'),
                            (self.generated_view_paths.get('side'), self.side_view_label, 'side')]
        for path, label_widget, key in paths_and_labels:
            if path and os.path.exists(path):  # Added os.path.exists for safety
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    self.original_pixmaps[key] = pixmap
                    label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))
                else:
                    label_widget.setPixmap(QPixmap())
                    label_widget.setText(error_img_text + f"\nPath: {path}")
            else:
                label_widget.setPixmap(QPixmap())
                label_widget.setText(placeholder_text)
        # self.apply_theme() # Called by caller or implicitly by other UI updates avoid redundant calls if possible

    def resizeEvent(self, event):  # 窗口大小变化时缩放图片
        super().resizeEvent(event)
        for key, pixmap in self.original_pixmaps.items():
            if key == 'top':
                label_widget = self.top_view_label
            elif key == 'front':
                label_widget = self.front_view_label
            elif key == 'side':
                label_widget = self.side_view_label
            else:
                continue
            if not pixmap.isNull() and label_widget.width() > 0 and label_widget.height() > 0:  # 检查控件尺寸
                label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))

    def analyze_scene_action(self, checked: bool = False):
        if not self.generated_view_paths:  # 检查字典是否为空
            QMessageBox.warning(self, self.i18n["error_title"], self.i18n.get("error_no_2d_views_for_vlm",
                                                                              "No 2D views available for VLM analysis. Please ensure point cloud has labels/data and views were generated."))
            return
        # 还要检查original_pixmaps是否已填充，意味着图像已加载。
        # 这个检查在generated_view_paths不为空且_load_and_display_original_views正常工作时大多是多余的。
        if not self.original_pixmaps:
            QMessageBox.warning(self, self.i18n["error_title"], self.i18n.get("error_no_images_loaded",
                                                                              "No images loaded to analyze. Please check view generation."))
            return

        api_key = self.api_key_input.text()
        if not api_key:
            QMessageBox.warning(self, self.i18n["error_title"], self.i18n["error_no_api_key"])
            return

        self.statusBar().showMessage(self.i18n["status_analyzing"])
        self.api_output_text.clear()
        self.raw_vlm_output_buffer = ""
        self.analyze_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        system_prompt = self.i18n["system_prompt"]  # Use self.i18n for current language prompts
        user_prompt = self.i18n["user_prompt"]
        self.api_worker = ApiWorker(api_key, self.generated_view_paths, system_prompt, user_prompt)
        self.api_worker.result_ready.connect(self.append_api_result)
        self.api_worker.error_occurred.connect(self.handle_api_error)
        self.api_worker.finished.connect(self.on_api_finished)
        self.api_worker.start()

    def consolidate_vlm_output(self, text_input: str) -> str:
        if not text_input: return ""
        text = text_input.strip()
        # 合并多余换行，但保留列表/格式化所需换行
        text = re.sub(r'\n(?!\s*([#*-]|\d+\.|\n|$))', ' ', text)  # 替换非列表的单换行为空格
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3个及以上换行缩减为2个
        return text.strip()

    def append_api_result(self, partial_result):
        self.raw_vlm_output_buffer += partial_result
        # For streaming, append raw text. SetMarkdown will be used at the end.
        self.api_output_text.insertPlainText(partial_result)  # Use insertPlainText for streaming
        self.api_output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.statusBar().showMessage(self.i18n["status_analysis_partial"])

    def handle_api_error(self, error_message):
        QMessageBox.critical(self, self.i18n["error_title"], error_message)
        # 如果有部分输出，显示格式化后的内容
        if self.raw_vlm_output_buffer:
            final_text = self.consolidate_vlm_output(self.raw_vlm_output_buffer)
            self.api_output_text.setMarkdown(final_text)  # 显示合并后的可能的markdown文本
            self.api_output_text.moveCursor(QTextCursor.MoveOperation.End)
        else:
            self.api_output_text.clear()  # 如果在错误前没有输出，则清除

        self.statusBar().showMessage(self.i18n["status_ready"])  # 或更具体的错误状态
        # 根据当前状态重新启用按钮
        self.load_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.analyze_button.setEnabled(bool(self.generated_view_paths))  # 仅当视图仍有效时

    def on_api_finished(self):
        final_text = self.consolidate_vlm_output(self.raw_vlm_output_buffer)
        self.api_output_text.setMarkdown(final_text)  # 设置最终格式化文本
        self.api_output_text.moveCursor(QTextCursor.MoveOperation.End)

        self.analyze_button.setEnabled(bool(self.generated_view_paths))
        self.load_button.setEnabled(True)
        self.clear_button.setEnabled(True)

        if not self.api_output_text.toPlainText().strip():  # 检查输出区域是否为空
            self.statusBar().showMessage(self.i18n["status_ready"])  # 或与视图就绪相关的状态
        else:
            self.statusBar().showMessage(self.i18n["status_analysis_complete"])
        self.save_settings()

    def clear_views(self):
        placeholder_text = self.i18n.get("views_placeholder", "Load a point cloud to generate views.")
        self.top_view_label.setPixmap(QPixmap())
        self.top_view_label.setText(placeholder_text)
        self.front_view_label.setPixmap(QPixmap())
        self.front_view_label.setText(placeholder_text)
        self.side_view_label.setPixmap(QPixmap())
        self.side_view_label.setText(placeholder_text)
        self.original_pixmaps.clear()

    def clear_all_action(self, checked: bool = False):
        self.point_cloud_file = None
        self.loaded_point_cloud_data_for_3d = None
        self.generated_view_paths.clear()
        self.api_output_text.clear()
        self.raw_vlm_output_buffer = ""
        self.clear_views()
        self.analyze_button.setEnabled(False)
        if hasattr(self, 'launch_3d_button'):
            self.launch_3d_button.setEnabled(False)
        self.statusBar().showMessage(self.i18n["status_ready"])

    def closeEvent(self, event):
        self.save_settings()
        if hasattr(self, 'api_worker') and self.api_worker is not None and self.api_worker.isRunning():
            self.api_worker.quit()  # Request termination
            if not self.api_worker.wait(1000):  # Wait up to 1 sec
                print("API worker did not terminate gracefully, forcing termination.")
                self.api_worker.terminate()  # Force terminate if quit doesn't work
                self.api_worker.wait()  # Wait for forced termination
        super().closeEvent(event)


if __name__ == '__main__':
    # 确保Open3D在某些系统上使用现代GL
    os.environ["OPEN3D_CPU_RENDERING"] = "false"  # 优先使用GPU渲染
    # 某些系统可能需要强制指定GL版本，如Linux下Mesa: os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

    app = QApplication(sys.argv)
    # 设置应用名称和版本，便于QSettings
    app.setApplicationName("PointCloudAnalyzer")
    app.setOrganizationName("MyCompany")

    # 字体设置（可选，为了统一）
    # font_db = QFontDatabase()
    # if font_db.hasFamily("Arial"): # 检查常用字体
    #     app.setFont(QFont("Arial", 10))
    # else:
    #     print("未找到Arial字体，使用系统默认。")

    QApplication.setStyle("Fusion")  # Fusion风格跨平台表现较好
    main_win = PointCloudAnalyzerApp()
    main_win.show()
    sys.exit(app.exec())
