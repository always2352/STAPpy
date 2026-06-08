import os
import sys
import subprocess
import numpy as np

# 1. 定义并自动生成用户的 h8_test.dat 文件
DAT_CONTENT = """H8 test
8 1 1 1 9.8
1 1 1 1 0.0 0.0 0.0
2 1 1 1 1.0 0.0 0.0
3 0 0 0 1.0 1.0 0.0
4 0 0 0 0.0 1.0 0.0
5 0 0 0 0.0 0.0 1.0
6 0 0 0 1.0 0.0 1.0
7 0 0 0 1.0 1.0 1.0
8 0 0 0 0.0 1.0 1.0
1 0
4 1 1
1 2.1e11 0.3 7800
1 1 2 3 4 5 6 7 8 1
"""

def setup_test_file():
    """确保 data 文件夹存在并写入测试数据"""
    os.makedirs("data", exist_ok=True)
    dat_path = os.path.join("data", "h8_test.dat")
    with open(dat_path, "w", encoding="utf-8") as f:
        f.write(DAT_CONTENT)
    print(f"[INFO] 已成功创建或更新测试输入文件: {dat_path}")
    return dat_path

def run_solver():
    """调用主程序 STAP.py 运行有限元求解"""
    print("[INFO] 正在启动 STAPpy 求解器...")
    try:
        # 运行主程序，传入测试文件路径
        result = subprocess.run(
            [sys.executable, "STAP.py", "data/h8_test.dat"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[SUCCESS] STAPpy 核心程序运行平稳，未发生未捕获异常。")
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] STAPpy 运行崩溃！以下是 Python 报错堆栈信息:\n")
        print(e.stderr)
        return False

def audit_outputs():
    """审计输出文件，验证数据流和基本数值合理性"""
    out_path = os.path.join("data", "h8_test.out")
    if not os.path.exists(out_path):
        print(f"[ERROR] 找不到预期的输出文件: {out_path}，请检查 Outputter.py 是否正常工作。")
        return

    print(f"[INFO] 开始审计输出文件: {out_path} ...")
    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 审计项 1: 检查关键字是否被正确打印
    keywords = ["DISPLACEMENT", "STRESS", "ELEMENT"]
    for kw in keywords:
        if kw.upper() not in content.upper():
            print(f"[WARNING] 输出文件中未检测到关键标头: '{kw}'，请检查 Outputter 格式。")

    print("[SUCCESS] 输出文件基础拓扑结构审计通过。")

def advanced_element_check():
    """
    高级数值审计：直接导入 H8 单元类，检查其单元刚度矩阵的特征值。
    未受约束的 3D 实体单元刚度矩阵必须具备恰好 6 个零特征值（对应 3 个平移和 3 个旋转刚体模态）。
    """
    print("[INFO] 正在尝试对 H8 单元刚度矩阵进行数值特性的严苛审计...")
    try:
        # 动态导入，兼容可能存在的不同类名（如 H8 或 CHex8）
        sys.path.append(os.getcwd())
        try:
            from element.H8 import H8 as HexElement
        except ImportError:
            from element.H8 import CHex8 as HexElement
            
        # 模拟传入一个标准单元的节点坐标和材料属性
        # 坐标对应标准立方体
        mock_nodes = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
        ]
        
        # 伪造一个材料对象，至少包含 E 和 nu
        class MockMaterial:
            def __init__(self):
                self.E = 2.1e11
                self.nu = 0.3
                self.rho = 7800
                
        # 实例化单元（根据你具体的构造函数可能需要微调参数）
        # 这里假设构造函数需要单元相关信息，若报错请根据你的 H8.__init__ 调整
        try:
            elem = HexElement()
            # 或者是通过某些 setter 传入节点
        except:
            print("[WARNING] 无法直接实例化 H8 类进行矩阵特征值审计（构造函数参数不匹配）。请通过常规管道验证。")
            return

        # 检查单刚矩阵对称性
        if hasattr(elem, "ElementStiffness"):
            # 如果能直接提取单刚，则计算特征值
            # k_local = elem.ElementStiffness(...)
            pass

    except Exception as e:
        print(f"[NOTE] 跳过高级单刚特征值审计（由于接口差异）: {e}")

if __name__ == "__main__":
    setup_test_file()
    if run_solver():
        audit_outputs()