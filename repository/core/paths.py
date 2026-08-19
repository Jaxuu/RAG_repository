import os

# 项目根目录（.当前目录 ..上一层目录）D:\Project\llm-project\RAG_repository\repository
REPOSITORY_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(REPOSITORY_ROOT, "temp_data")

# 前端页面静态资源目录   D:\Project\llm-project\RAG_repository\repository\front
FRONT_PAGE_DIR = os.path.join(REPOSITORY_ROOT, "front")

def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR