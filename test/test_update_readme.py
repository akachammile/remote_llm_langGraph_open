import os
import re

IGNORE_DIRS = {'.git', '__pycache__', '.venv', 'env', '.idea', '.vscode', 'build', 'dist', ".*"}
MAX_DEPTH = 3  # 控制目录显示深度


def generate_tree(root='.', prefix='', depth=0):
    """递归生成目录树字符串"""
    if depth > MAX_DEPTH:
        return ''
    items = [i for i in os.listdir(root) if i not in IGNORE_DIRS]
    items.sort()
    lines = []
    for i, item in enumerate(items):
        full = os.path.join(root, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        lines.append(prefix + connector + item)
        if os.path.isdir(full):
            extension = "    " if i == len(items) - 1 else "│   "
            lines.extend(generate_tree(full, prefix + extension, depth + 1))
    return lines


def update_readme():
    """自动插入到 README.md"""
    if not os.path.exists('README.md'):
        print("❌ 未找到 README.md 文件")
        return

    # 生成目录树
    tree_lines = ['.', *generate_tree('.')]
    tree_text = '\n'.join(tree_lines)

    # 生成新的结构段
    new_section = f"## 📁 项目结构\n```bash\n{tree_text}\n```"

    # 读取原 README
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # 如果已经存在项目结构部分，就替换；否则追加到末尾
    if "## 📁 项目结构" in content:
        content = re.sub(r"## 📁 项目结构[\s\S]*?```", new_section, content)
    else:
        content += "\n\n" + new_section

    # 写回 README
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 已更新 README.md 中的项目结构！")


if __name__ == "__main__":
    update_readme()
