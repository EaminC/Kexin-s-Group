# **SYMC: Exploiting Code Symmetries for Learning Program Semantics**

## **1. Background**

- Code exhibits **semantic-preserving symmetries**, such as variable renaming and token permutations.
- Existing Transformer models struggle to explicitly learn these symmetries, leading to inefficient representations.

## **2. Key Idea**

- SYMC enforces **Aut(IG)-equivariant self-attention** to capture program symmetries.
- Directly constructing IG is computationally impractical, so **Program Dependence Graph (PDG)** is used as an over-approximation.
- PDG preserves **control and data dependencies**, ensuring **Aut(PDG)-equivariance** also guarantees **Aut(IG)-equivariance**.

## **3. PDG Construction**

- PDG is built from **static analysis**, capturing:
  - **Data dependencies**: Read-after-write, write-after-read, write-after-write.
  - **Control dependencies**: Enforcing execution order constraints.
- **Over-approximation trade-off**: A conservative approach ensures correctness but may miss finer symmetries.
- Future work aims to incorporate **alias analysis** or **dynamic analysis** for improved precision.

## **4. Encoding Graph Structure in Self-Attention**

- Introduces a **distance matrix d** on PDG:
  - Each entry$d_{ij} = (p_{ij}, n_{ij})$ represents **longest paths** from the lowest common ancestor.
  - **Positive distance** (\(p*{ij}\)) and **negative distance** (\(n*{ij}\)) are incorporated into **Multi-Head Self-Attention (MHA)**.
  - The first half of attention heads use **positive distances**, and the second half use **negative distances**:
    1. $ MHA_i = W_V e \cdot s(W_K e^T \cdot W_Q e + d_p) $, for$ i \in [1, h/2]$
    2. $ MHA_i = W_V e \cdot s(W_K e^T \cdot W_Q e + d_n) $, for $i \in [h/2+1, h]$
- Ensures **Aut(PDG)-equivariance**, as **distance matrix d remains invariant under graph automorphisms**.

## **5. Overhead & Efficiency**

- PDG construction incurs computational overhead but is optimized for efficiency.
- Future work suggests **interleaving graph construction with training/inference** to minimize overhead.
- SYMC imposes **equivariance constraints regardless of training**, making it adaptable with minimal fine-tuning.

## **6. Extending to Other Code Symmetries**

- Current framework focuses on **permutation groups**, but can be extended to **other transformations forming a group**:
  - **Variable renaming** is a vocabulary-wide permutation.
  - **Token permutations** (e.g., `a = a + 1` ⇔ `a = 1 + a`).
- Some transformations (e.g., **insertion/deletion**) are **not invertible**, requiring **semigroup formalisms** instead of groups.
- Identifying symmetry structures for **arbitrary compiler optimizations** (e.g., `gcc -O3`) remains an open research direction.

## **7. Experimental Results**

- **Equivariant self-attention improves performance** in learning program semantics.
- **Pre-training helps adaptation to symmetry constraints** with minimal fine-tuning.
- **Comparison with baselines**:
  - **Aut(PDG)-equivariant layers outperform non-equivariant baselines**.
  - Fully permutation-equivariant models (with uniform distance matrices) underperform compared to SYMC.

## **8. Conclusion & Future Work**

- SYMC introduces a novel **Aut(PDG)-equivariant Transformer** for code analysis.
- Demonstrates strong results with efficient **PDG-based static analysis** and symmetry-aware self-attention.
- Future directions:

  - **More precise dependency analysis** (alias analysis, dynamic techniques).
  - **Applying symmetry constraints to broader transformations** beyond permutations.
  - **Optimizing PDG construction during training** to reduce computational overhead.

  # Code Generation Using GNN - ICLR 2020

## 1. Overview

The paper _"GLOBAL RELATIONAL MODELS OF SOURCE CODE"_ presents a method combining **Graph Neural Networks (GNNs) and Transformers** for code repair, specifically for **Variable Misuse Detection & Repair**.

## 2. Main Methods

### 2.1 Graph Neural Network (GNN)

- Represents code as a **Program Graph**, where nodes correspond to **variables, operators, and syntax elements**, and edges capture **syntax, data flow, and control flow** relations.
- Uses **Gated Graph Neural Networks (GGNNs)** for message passing to learn structured code representations.

### 2.2 Hybrid Models

To overcome GNN's locality limitations, the paper proposes **two hybrid models**:

1. **Graph Sandwiches**

   - Alternates **sequential message passing** (e.g., RNN or Transformer) with **graph-based message passing (GNN)**.
   - Enhances global information flow by combining GNN's structured insights with sequence models.

2. **Graph Relational Embedding Attention Transformer (GREAT)**
   - Extends **Transformer's self-attention** mechanism by incorporating **graph relational information**.
   - Captures both **global** and **structural** relationships for better code understanding.

## 3. Code Repair Task (Variable Misuse Repair)

- **Objective**: Identify incorrect variable usage in a function and suggest the correct replacement.
- **Approach**: Combining **GNN and Transformer** features improves both **bug localization** and **repair accuracy**.

## 4. Key Experimental Results

- **Hybrid models (GNN + Transformer) outperform standalone GNNs and RNNs**, achieving **faster convergence** and **higher accuracy**.
- **GREAT model** (Transformer + graph relational information) achieves **state-of-the-art results** on bug localization and repair tasks.

## 5. Code Implementation

GitHub Repository:  
🔗 [https://github.com/VHellendoorn/ICLR20-Great](https://github.com/VHellendoorn/ICLR20-Great)

# Code Transformation System with Knowledge Graph & Agent Reflection

## **🔹 Step 1: Code Parsing & Knowledge Graph Construction**

- **SYMC processes the code and generates labels**

  - Identifies function calls, variable dependencies, control flows, etc.
  - Example:
    ```python
    def add(a, b): return a + b
    ```
    → Label: `"Addition Function"`

- **Construct Knowledge Graph (KG)**

  - Store function names → semantic labels, code relationships (calls, inheritance, dependencies)
  - Example:
    ```
    Node: add_function
    ├── Type: Function
    ├── Semantic: Addition Operation
    ├── Related: Math Library
    ```
  - **Graph Representation**:
    ```mermaid
    graph TD
      A[add_function] -->|depends on| B[Math Library]
      A -->|semantic| C[Addition Operation]
    ```

- **Decomposing Code into Knowledge Graph Units**
  - The function is broken into components:
    - Function definition
    - Variable initialization
    - Control structures (loops, conditionals)
  - **SYMC labels each code unit and stores it in KG**
  - Example Knowledge Graph Representation:
    ```mermaid
    graph TD
      X[Function: add] -->|Contains| Y[Operation: Addition]
      X -->|Contains| Z[Return Statement]
    ```

## **🔹 Step 2: Code Decomposition & Knowledge Storage**

- **Each code block is identified and stored with its SYMC-generated labels**
- **Example: Loop Structure Decomposition**
  - **Given input code:**
    ```python
    def count_to_n(n):
        count = 0
        while count < n:
            print(count)
            count += 1
    ```
  - **SYMC generates semantic labels:**
    - `Loop Condition` → `count < n`
    - `Loop Body` → `print(count); count += 1`
    - `Loop Exit Logic` → `count reaches n`
  - **Knowledge Graph Representation:**
    ```mermaid
    graph TD
      A[Loop Condition] -->|Evaluates| B{count < n}
      B -->|True| C[Loop Body]
      B -->|False| D[Loop Exit Logic]
      C -->|Executes| E[print count]
      C -->|Updates| F[count += 1]
      C -->|Repeats| A
      D -->|Terminates Loop| G[End]
    ```

## **🔹 Step 3: Code Reconstruction & Generation**

- **Agent retrieves code units from KG and generates code block by block**
- **Ensures each generated unit aligns with stored knowledge**
- Example:
  ```python
  def count_to_n_v2(n):
      for i in range(n):
          print(i)
  ```
  - The transformation changes **while loop** to a **for loop** but retains the same semantic meaning.

## **🔹 Step 4: Agent Reflection & Code Optimization**

- **Agent reviews each generated unit and ensures correctness**
- **Reflection Mechanism:**
  - **Self-evaluation**: "Does this maintain original intent?"
  - **Error correction**: If inconsistencies arise, the agent modifies the structure.
- **Refinement Process:**
  ```mermaid
  sequenceDiagram
    participant Agent
    participant Model
    participant KG as Knowledge Graph
    Agent->>KG: Retrieve Code Units
    KG->>Agent: Return Structured Components
    Agent->>Model: Generate Transformed Code
    Model->>Agent: Returns Code
    Agent->>KG: Validate Semantics & Structure
    KG->>Agent: Feedback (Correct/Incorrect)
    alt Incorrect
        Agent->>Model: Adjust Components & Retry
    end
    Agent->>Model: Finalize Output
  ```

## **🔹 Step 5: Final Code Assembly & Output**

- **All generated components are assembled into the final output**
- **Ensures semantic consistency and optimized structure**
- Example Output:
  ```
  Code has been rewritten for efficiency.
  Loop structure has been optimized from `while` to `for` based on the KG.
  ```

### **Key Features of This Design:**

1. **SYMC-based Knowledge Graph (KG)** → Stores decomposed code units and semantic labels
2. **Agent-Controlled Generation** → Stepwise construction and validation of code
3. **Black-box Model (GNN/LLM)** → Ensures semantic equivalence while allowing stylistic changes
4. **Error Handling & Adaptation** → Iterative refinement with feedback loops

This design ensures **modular, explainable, and optimized code transformation**, combining **deep learning, structured knowledge storage, and agent-driven control.** 🚀
