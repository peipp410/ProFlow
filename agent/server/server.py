import os
import subprocess # 导入 subprocess 模块
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import base64
import mimetypes

# 技能库的根目录（设定为当前脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 限制 RPC 服务的路径，增加一点安全性
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# === RemoteTool 基类或 RemoteSkillTool 需要调用的核心函数 ===

# def run(command_name: str, command_args: str) -> str:
#     """
#     执行任意的 shell 命令。
#     这个函数是 RemoteCommandTool 在服务器端的核心功能。
#     它接收命令名和命令参数，然后通过 subprocess 执行。
#     """
#     # full_command = f"{command_name} {command_args}"
#     try:
#         # 使用 subprocess.run 执行命令
#         # capture_output=True 捕获 stdout 和 stderr
#         # text=True 解码输出为文本
#         # check=True 如果命令返回非零退出码，则抛出 CalledProcessError
#         result = subprocess.run(
#             command_args,
#             shell=True,
#             capture_output=True,
#             text=True,
#             check=False # 不检查返回码，因为我们需要返回 stderr
#         )

#         output = result.stdout.strip()
#         error = result.stderr.strip()

#         if result.returncode != 0:
#             # 如果命令执行失败，返回错误信息和标准错误
#             return f"Error executing command (Exit Code: {result.returncode}): {command_args}\nStderr: {error}\nStdout: {output}"
#         elif error:
#             # 如果有标准错误但命令成功（例如某些警告信息），也返回
#             return f"Command executed with warnings/errors (Exit Code: {result.returncode}): {command_args}\nStderr: {error}\nStdout: {output}"
#         else:
#             # 命令成功且无错误
#             return output
#     except Exception as e:
#         return f"Exception during command execution: {str(e)}"

def run(language: str, command: str):
    """
    language: e.g. "bash", "python"
    command: the command string passed from client
    return: list of dict items:
        [{"type":"console","format":"output","content":"..."}]
    """
    # 1) 默认兜底：确保 full_command 永远会被定义
    full_command = ""

    # 2) 根据 language 构造 full_command（你也可以只支持 bash）
    if language == "bash":
        # 让 command 在 shell 中执行
        full_command = command
        # 可选：你也可以强制加 set -e -o pipefail
        # full_command = f"set -euo pipefail; {command}"
    else:
        # 如果只允许 bash，其他 language 直接包装成 bash 执行也可以
        # 但推荐：直接报错，让调用方修正 language
        raise ValueError(f"Unsupported language for run(): {language}")

    # 3) 执行并收集输出
    import subprocess

    try:
        proc = subprocess.run(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            executable="/bin/bash"  # mac/linux 有 bash
        )
        output_text = proc.stdout or ""

        # 4) 返回结构必须和 execute() 解析一致
        # execute() 只处理 type=console/format=output 和 type=image/format=base64
        results = []
        if output_text.strip():
            results.append({
                "type": "console",
                "format": "output",
                "content": output_text
            })

        # 若你希望把 exit code 也回传，可以额外塞一个 dict，但 execute 可能忽略它（也能放到 raw_result）
        # results.append({"type":"meta","format":"json","content": ...})

        return results

    except Exception as e:
        # 让客户端也能显示错误，而不是直接触发客户端解析异常
        return [{
            "type": "console",
            "format": "output",
            "content": str(e)
        }]

def list_categories():
    """返回所有分类（即根目录下的文件夹）"""
    categories = []
    # 确保 BASE_DIR 是一个可迭代的目录
    if not os.path.isdir(BASE_DIR):
        return []
        
    for item in os.listdir(BASE_DIR):
        # 排除以 '.' 开头的隐藏文件/目录
        if os.path.isdir(os.path.join(BASE_DIR, item)) and not item.startswith('.'):
            categories.append({"name": item, "description": f"{item} 相关的技能"})
    return categories

def get_category_description(category):
    # 可以在这里添加更复杂的逻辑来获取分类的详细描述
    return {"name": category, "description": f"这是 {category} 分类"}

def list_skills_by_category(category):
    """返回分类下的所有技能（即子文件夹）"""
    cat_path = os.path.join(BASE_DIR, category)
    if not os.path.exists(cat_path) or not os.path.isdir(cat_path):
        return []
    
    skills = []
    for item in os.listdir(cat_path):
        if os.path.isdir(os.path.join(cat_path, item)) and not item.startswith('.'):
            # 可以检查是否存在 SKILL.md 来确保是有效的技能目录
            if os.path.exists(os.path.join(cat_path, item, "SKILL.md")):
                skills.append({"name": item, "description": f"技能: {item}"})
            else:
                skills.append({"name": item, "description": f"技能: {item} (缺少SKILL.md)"})
    return skills

def get_skill_info(skill_name):
    """全局搜索技能并返回基本信息"""
    # 确保 BASE_DIR 是一个可迭代的目录
    if not os.path.isdir(BASE_DIR):
        return None

    for cat in os.listdir(BASE_DIR):
        cat_path = os.path.join(BASE_DIR, cat)
        if os.path.isdir(cat_path) and not cat.startswith('.'):
            skill_path = os.path.join(cat_path, skill_name)
            if os.path.exists(skill_path) and os.path.isdir(skill_path):
                # 检查是否存在 SKILL.md，如果是，则认为是一个完整的技能
                if os.path.exists(os.path.join(skill_path, "SKILL.md")):
                    return {
                        "name": skill_name,
                        "category": cat,
                        "version": "1.0", # 版本信息可以从 SKILL.md 中解析
                        "dir_path": skill_path,
                        "description": "详细请看 SKILL.md"
                    }
    return None

def get_directory_tree(skill_name):
    """简单的目录树展示"""
    info = get_skill_info(skill_name)
    if not info: return "Error: Skill not found"
    
    tree_lines = []
    # os.walk 会返回 (当前路径, 子目录列表, 文件列表)
    for root, dirs, files in os.walk(info['dir_path']):
        # 计算当前目录相对于技能根目录的深度
        relative_path = os.path.relpath(root, info['dir_path'])
        # 处理技能根目录本身
        if relative_path == ".":
            level = 0
            current_dir_name = os.path.basename(info['dir_path']) # 获取技能目录本身的名称
        else:
            level = relative_path.count(os.sep) + 1
            current_dir_name = os.path.basename(root)

        indent = '    ' * level # 4个空格作为缩进

        # 添加当前目录
        tree_lines.append(f"{indent}├── {current_dir_name}/") # 使用树形符号

        # 添加文件
        for f in files:
            tree_lines.append(f"{indent}│   └── {f}") # 文件缩进比目录多一级

        # 排序目录和文件，使输出更整洁
        dirs.sort()
        files.sort()
            
    return "\n".join(tree_lines)


def get_skill_content(skill_name):
    """读取 SKILL.md"""
    # info = get_skill_info(skill_name)
    # file_path = os.path.join(info['dir_path'], "SKILL.md")
    # outs = read_file(file_path)
    # return outs['content']
    filename = "SKILL.md"
    info = get_skill_info(skill_name)
    if not info: return "Error: Skill not found or directory is empty"
    
    file_path = os.path.join(info['dir_path'], filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in skill '{skill_name}' at path '{info['dir_path']}'"
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file."
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filename}' for skill '{skill_name}': {str(e)}"

def read(filepath, offset=None, limit=None, encoding='utf-8'):
    """
    读取文件内容
    
    参数：
    - filepath: 文件路径
    - offset: 起始行号（可选，从0开始）
    - limit: 读取行数（可选）
    - encoding: 文件编码（默认 utf-8）
    
    返回值：
    - 字典格式，包含 success, error, file_type, content, mime_type, bytes_read, absolute_path
    """
    try:
        # 获取绝对路径
        absolute_path = os.path.abspath(filepath)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"文件不存在: {filepath}",
                "file_type": None,
                "content": None,
                "mime_type": None,
                "bytes_read": 0,
                "absolute_path": absolute_path
            }
        
        # 检查是否是文件
        if not os.path.isfile(filepath):
            return {
                "success": False,
                "error": f"路径不是文件: {filepath}",
                "file_type": None,
                "content": None,
                "mime_type": None,
                "bytes_read": 0,
                "absolute_path": absolute_path
            }
        
        # 判断文件类型
        mime_type, _ = mimetypes.guess_type(filepath)
        is_image = mime_type and mime_type.startswith('image/')
        
        if is_image:
            # 读取图片文件（二进制）
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            # 转换为 base64 编码
            content_b64 = base64.b64encode(file_content).decode('ascii')
            
            return {
                "success": True,
                "error": None,
                "file_type": "image",
                "content": content_b64,
                "mime_type": mime_type or "application/octet-stream",
                "bytes_read": len(file_content),
                "absolute_path": absolute_path
            }
        
        else:
            # 读取文本文件
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            # 处理行范围
            if offset is not None and limit is not None:
                # 指定起始行和数量
                start = max(0, offset)
                end = min(len(lines), start + limit)
                content_lines = lines[start:end]
            elif offset is not None:
                # 只指定起始行
                start = max(0, offset)
                content_lines = lines[start:]
            elif limit is not None:
                # 只指定数量
                content_lines = lines[:limit]
            else:
                # 读取全部
                content_lines = lines
            
            # 将行列表转换为字符串
            content = ''.join(content_lines)
            
            # 计算读取的字节数
            bytes_read = len(content.encode(encoding))
            
            return {
                "success": True,
                "error": None,
                "file_type": "text",
                "content": content,
                "mime_type": mime_type or "text/plain",
                "bytes_read": bytes_read,
                "absolute_path": absolute_path
            }
    
    except UnicodeDecodeError as e:
        return {
            "success": False,
            "error": f"文件编码错误: {str(e)}。请尝试使用其他编码（如 'latin-1', 'gbk'）",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
    except PermissionError as e:
        return {
            "success": False,
            "error": f"权限被拒绝: {str(e)}",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"读取文件出错: {str(e)}",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
def read_file(filepath, offset=None, limit=None, encoding='utf-8'):
    """
    读取文件内容
    
    参数：
    - filepath: 文件路径
    - offset: 起始行号（可选，从0开始）
    - limit: 读取行数（可选）
    - encoding: 文件编码（默认 utf-8）
    
    返回值：
    - 字典格式，包含 success, error, file_type, content, mime_type, bytes_read, absolute_path
    """
    try:
        # 获取绝对路径
        absolute_path = os.path.abspath(filepath)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"文件不存在: {filepath}",
                "file_type": None,
                "content": None,
                "mime_type": None,
                "bytes_read": 0,
                "absolute_path": absolute_path
            }
        
        # 检查是否是文件
        if not os.path.isfile(filepath):
            return {
                "success": False,
                "error": f"路径不是文件: {filepath}",
                "file_type": None,
                "content": None,
                "mime_type": None,
                "bytes_read": 0,
                "absolute_path": absolute_path
            }
        
        # 判断文件类型
        mime_type, _ = mimetypes.guess_type(filepath)
        is_image = mime_type and mime_type.startswith('image/')
        
        if is_image:
            # 读取图片文件（二进制）
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            # 转换为 base64 编码
            content_b64 = base64.b64encode(file_content).decode('ascii')
            
            return {
                "success": True,
                "error": None,
                "file_type": "image",
                "content": content_b64,
                "mime_type": mime_type or "application/octet-stream",
                "bytes_read": len(file_content),
                "absolute_path": absolute_path
            }
        
        else:
            # 读取文本文件
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            # 处理行范围
            if offset is not None and limit is not None:
                # 指定起始行和数量
                start = max(0, offset)
                end = min(len(lines), start + limit)
                content_lines = lines[start:end]
            elif offset is not None:
                # 只指定起始行
                start = max(0, offset)
                content_lines = lines[start:]
            elif limit is not None:
                # 只指定数量
                content_lines = lines[:limit]
            else:
                # 读取全部
                content_lines = lines
            
            # 将行列表转换为字符串
            content = ''.join(content_lines)
            
            # 计算读取的字节数
            bytes_read = len(content.encode(encoding))
            
            return {
                "success": True,
                "error": None,
                "file_type": "text",
                "content": content,
                "mime_type": mime_type or "text/plain",
                "bytes_read": bytes_read,
                "absolute_path": absolute_path
            }
    
    except UnicodeDecodeError as e:
        return {
            "success": False,
            "error": f"文件编码错误: {str(e)}。请尝试使用其他编码（如 'latin-1', 'gbk'）",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
    except PermissionError as e:
        return {
            "success": False,
            "error": f"权限被拒绝: {str(e)}",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"读取文件出错: {str(e)}",
            "file_type": None,
            "content": None,
            "mime_type": None,
            "bytes_read": 0,
            "absolute_path": os.path.abspath(filepath)
        }
    
def write_file(filepath, content):
    """
    向文件写入内容
    
    参数：
    - filepath: 文件路径
    - content: 要写入的内容（字符串）
    
    返回值：
    - 字典格式，包含 success, error, absolute_path, bytes_written, operation
    """
    try:
        import os
        
        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            bytes_written = f.write(content)
        
        # 获取绝对路径
        absolute_path = os.path.abspath(filepath)
        
        # 返回成功格式
        return {
            "success": True,
            "error": None,
            "absolute_path": absolute_path,
            "bytes_written": bytes_written,
            "operation": "write"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "absolute_path": None,
            "bytes_written": 0,
            "operation": "write"
        }

# === 启动 RPC 服务器 ===
def start_server():
    # 绑定到 0.0.0.0 和 8899 端口
    server = SimpleXMLRPCServer(("0.0.0.0", 8899), requestHandler=RequestHandler, allow_none=True)
    
    # 注册所有核心函数
    # server.register_function(run, 'execute_remote_command') # 注册 run 函数
    server.register_function(run, 'run')
    server.register_function(list_categories, 'list_categories')
    server.register_function(get_category_description, 'get_category_description')
    server.register_function(list_skills_by_category, 'list_skills_by_category')
    server.register_function(get_skill_info, 'get_skill_info')
    server.register_function(get_directory_tree, 'get_directory_tree')
    server.register_function(get_skill_content, 'get_skill_content')
    server.register_function(read_file, 'read')
    server.register_function(read_file, 'read_file')
    server.register_function(write_file, 'write')
    
    print("🚀 技能库 RPC 服务端已启动！监听端口: 8899...")
    server.serve_forever()

if __name__ == "__main__":
    start_server()