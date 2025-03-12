import ast
import networkx as nx
import matplotlib.pyplot as plt

class ASTGraphConverter(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def add_node(self, name):
        node_id = self.node_counter
        self.graph.add_node(node_id, label=name)
        self.node_counter += 1
        return node_id

    def visit(self, node, parent_id=None):
        """递归遍历 AST 并构建图"""
        node_name = type(node).__name__
        node_id = self.add_node(node_name)

        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)

        for child in ast.iter_child_nodes(node):
            self.visit(child, node_id)

        return node_id

    def build_graph(self, code):
        """解析 Python 代码并构建 AST 图"""
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph

def draw_ast_graph(graph):
    """绘制 AST 图"""
    pos = nx.spring_layout(graph)  # 计算节点位置
    labels = nx.get_node_attributes(graph, "label")

    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    plt.show()

# 示例 Python 代码
python_code = """
def add(a, b):
    return a + b
"""
python_code1 = """
import a
"""
# 解析并构建 AST 图
converter = ASTGraphConverter()

ast_graph = converter.build_graph(python_code1)
# 绘制图
draw_ast_graph(ast_graph)