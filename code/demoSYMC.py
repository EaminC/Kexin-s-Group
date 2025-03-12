import torch
import torch.nn as nn
import networkx as nx
import numpy as np

# ==== Step 1: Read Code from File ====
def read_code_from_file(filename):
    """ Reads Python code from a file """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

# ==== Step 2: Build the Program Dependence Graph (PDG) ====
def build_pdg_from_code(code):
    """ Constructs a PDG graph from Python code """
    pdg = nx.DiGraph()
    lines = code.strip().split("\n")
    
    for i, line in enumerate(lines):
        pdg.add_node(i)  # Node index starts from 0
        if i > 0:
            pdg.add_edge(i - 1, i)  # Simple sequential dependency
            
    return pdg

# ==== Step 3: Compute the Distance Matrix ====
def compute_distance_matrix(graph):
    """ Computes the distance matrix for the PDG """
    n = len(graph.nodes)
    d_matrix = np.zeros((n, n, 2))  # Each entry is (p_ij, n_ij)

    for i in graph.nodes:
        for j in graph.nodes:
            if i == j:
                continue
            try:
                path_length = nx.shortest_path_length(graph, source=i, target=j)
                d_matrix[i, j] = (path_length, -path_length)  # Negative for symmetry
            except nx.NetworkXNoPath:
                d_matrix[i, j] = (0, 0)  # No path case (避免 inf)

    return torch.tensor(d_matrix, dtype=torch.float32)

# ==== Step 4: Define the PDG Self-Attention Layer ====
class PDGSelfAttention(nn.Module):
    """ Self-attention layer with PDG equivariance """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5
        
        # Query, Key, Value projections
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        # Distance matrix embeddings
        self.dist_embedding = nn.Linear(2, embed_dim)  # Combine pos & neg distances

    def forward(self, x, dist_matrix):
        B, N, D = x.shape  # Batch, Num_Tokens, Embedding_Dim

        Q = self.w_q(x)  # (B, N, D)
        K = self.w_k(x)  # (B, N, D)
        V = self.w_v(x)  # (B, N, D)

        # Compute attention scores with distance matrix embeddings
        dist_embed = self.dist_embedding(dist_matrix)  # (B, N, N, D)

        # 确保数值稳定性
        attn = (torch.matmul(Q, K.transpose(-2, -1)) / self.scale) + dist_embed.mean(dim=-1)

        # 过滤 NaN / Inf
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=-1.0)

        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        
        return output

# ==== Step 5: Main Execution ====
if __name__ == "__main__":
    # Define parameters
    embed_dim = 8  # Fixed embedding dimension
    num_heads = 2   # Number of attention heads

    # File paths (modify if necessary)
    file1, file2 = "code1.py", "code2.py"

    # Read code
    code1 = read_code_from_file(file1)
    code2 = read_code_from_file(file2)

    # Build PDG
    pdg1 = build_pdg_from_code(code1)
    pdg2 = build_pdg_from_code(code2)

    # Compute distance matrices
    distance_matrix1 = compute_distance_matrix(pdg1)
    distance_matrix2 = compute_distance_matrix(pdg2)

    # 确保 num_tokens 一致
    num_tokens1 = len(pdg1.nodes)
    num_tokens2 = len(pdg2.nodes)

    # 生成随机 embeddings
    x1 = torch.randn(1, num_tokens1, embed_dim)
    x2 = torch.randn(1, num_tokens2, embed_dim)

    # 调整 distance_matrix 形状
    distance_matrix1 = distance_matrix1.unsqueeze(0).expand(1, num_tokens1, num_tokens1, 2)
    distance_matrix2 = distance_matrix2.unsqueeze(0).expand(1, num_tokens2, num_tokens2, 2)

    # 形状检查
    assert x1.shape[1] == distance_matrix1.shape[1], f"Mismatch in num_tokens1: {x1.shape[1]} vs {distance_matrix1.shape[1]}"
    assert x2.shape[1] == distance_matrix2.shape[1], f"Mismatch in num_tokens2: {x2.shape[1]} vs {distance_matrix2.shape[1]}"

    # 初始化 PDG Self-Attention 模型
    pdg_self_attention = PDGSelfAttention(embed_dim, num_heads)

    # 计算输出
    output1 = pdg_self_attention(x1, distance_matrix1)
    output2 = pdg_self_attention(x2, distance_matrix2)

    # 打印最终结果
    print("Self-Attention Output for Code1:\n", output1)
    print("Self-Attention Output for Code2:\n", output2)