Process by Apr4

[toc]

### **Code Editor Bench**

paper https://arxiv.org/pdf/2404.03543

code https://github.com/CodeEditorBench/CodeEditorBench

**baseline methods**

Test cases set $R_i$

Actual output set $a_c$

<img src="assets/image-20250404230035565.png" alt="image-20250404230035565" style="zoom:50%;" />

**correctness** 

defined as $\forall n,y_n =a_{c_i}(x_n)$    (All Correct,AC)

**subtask**

1. **Debug**

<img src="assets/image-20250404213853869.png" alt="image-20250404213853869" style="zoom:40%;" />

![image-20250404214057780](assets/image-20250404214057780.png)

For c: $\exist n,y_n \neq a_{c}(x_n)$    (not AC)

For $c*$ $\forall n,y_n =a_{c^*}(x_n)$   (AC)



2. **Translator**

<img src="assets/image-20250404214721418.png" alt="image-20250404214721418" style="zoom:50%;" />

![image-20250404214636869](assets/image-20250404214636869.png)

For c: $\forall n,y_n =a_{c}(x_n)$   (AC)

For $c*$ $\forall n,y_n =a_{c^*}(x_n)=a_{c}(x_n)$   (from AC to AC,but different language)

3. **polisher**

<img src="assets/image-20250404214839524.png" alt="image-20250404214839524" style="zoom:50%;" />

![image-20250404215002599](assets/image-20250404215002599.png)

from AC to AC

Additional:

$avg_{time}(c^*)\leq avg_{time}(c)$   or.   $avg_{memory}(c^*)\leq avg_{memory}(c)$



4. **Requirements Switch**

   <img src="assets/image-20250404215159968.png" alt="image-20250404215159968" style="zoom:50%;" />

   <img src="assets/image-20250404215333008.png" alt="image-20250404215333008" style="zoom:50%;" />

   ![image-20250404215413073](assets/image-20250404215413073.png)

AC on $R_i$ to AC on $R_i^*$





**Dataset**

https://huggingface.co/datasets/m-a-p/CodeEditorBench/tree/main

<img src="assets/image-20250404215941989.png" alt="image-20250404215941989" style="zoom:50%;" />

**Filter** : not too long 800 lines/1000tokens limit

**dataset topic**

 `data structures`— trees, stacks, queues,arrays, hash tables,and pointers.  

  `algorithms`--dp,sorting,D/BFS ,recursion



**Debug dataset:**

Insert error with LLM

Errtype:

<img src="assets/image-20250404221008612.png" alt="image-20250404221008612" style="zoom:30%;" />

prompt

```markdown
### Instruction: 
Given code: code
Please add error: error name to the above code
Please output the modified code directly
### Answer:
```

[CEBDEBUG](#datasample1)

**Translate &  Polish dataset** 

 stratify the dataset according to code complexity, 

$Easy:Medium:Hard=3:4:1$

sample

[CEBTran](#datasample2)

[CEBDPol](#datasample3)



**SwitchDataset**

Group A  `strong relation`

similar questions provided by Leetcode under a certain question(With clear human feedback)

Group B `weak relation`

collect labels

Cluster the questions based on the number of tags they possess

employ Bertto assess the semantic similarity between the descriptions of two questions

within each category. threshold  0.92.

[CEBSwitch](#datasample4)

Other:Primary-Plus :timestamp filter(exclude outdated data)

### Running `swe-agent`

**Local**

Successfully run through `ollama/llama3.1-8B-Instruct` 

takes a lot of time to setup but can provide some local model results for comparison

**API**

Using `Openai/Claude api` can be easier if possible(dont have key now, but can be setup quickly)

### datasample1



```json
{
  "idx": 1330,
  "title": "",
  "code_language": "java",
  "incorrect_solutions": "class Solution {\n    public void nextPermutation(int[] n) {\n        if (n == null || n.length <= 1) return;\n\n        int i = n.length - 2;\n        while (i >= 0 && n[i] >= n[i + 1]) i--;\n\n        int j = n.length - 1;\n        if (i >= 0) {\n            while (n[j] >= n[i]) j--;\n            swap(n, i, j);\n        }\n\n        reverse(n, i + 1, n.length - 1);\n\n        for (int p = 0; p < n.length; p++) {\n            System.out.println(n[p]);\n        }\n    }\n\n    public static void swap(int[] n, int i, int j) {\n        int temp = n[i];\n        n[i] = n[j];\n        n[j] = temp;\n    }\n\n    public static void reverse(int[] n, int i, int j) {\n        while (i < j) {\n            swap(n, i, j);\n            i++;\n            j--;\n        }\n    }\n}",
  "solutions": "class Solution {\n    public void nextPermutation(int[] n) {\n        if (n == null || n.length <= 1) return;\n\n        int i = n.length - 2;\n        while (i >= 0 && n[i] >= n[i + 1]) i--;\n\n        int j = n.length - 1;\n        if (i >= 0) {\n            while (n[j] <= n[i]) j--;\n            swap(n, i, j);\n        }\n\n        reverse(n, i + 1, n.length - 1);\n\n        for (int p = 0; p < n.length; p++) {\n            System.out.println(n[p]);\n        }\n    }\n\n    public static void swap(int[] n, int i, int j) {\n        int temp = n[i];\n        n[i] = n[j];\n        n[j] = temp;\n    }\n\n    public static void reverse(int[] n, int i, int j) {\n        while (i < j) {\n            swap(n, i, j);\n            i++;\n            j--;\n        }\n    }\n}",
  "type": "logic error:condition error",
  "difficulty": "medium",
  "public_tests_input": "nums = [1,2,3]",
  "public_tests_output": "[1,3,2]",
  "private_tests_input": [
    "[8, 7, 6, 5, 4, 3, 2, 1]",
    "[1,2,3,4]",
    "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
    "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
    "[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]",
    "[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]",
    "[1,1,5]",
    "[2, 3, 4, 5, 6, 7, 8, 9]",
    "[1, 2, 3, 4, 5]",
    "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
    "[1, 1, 1, 1, 1, 1, 1, 1]",
    "[5, 4, 3, 2, 1]",
    "[1, 2, 3, 4, 5, 6]",
    "[2, 3, 4, 5, 6, 7, 8, 9, 10]",
    "[3,2,1]",
    "[1, 2, 3, 4, 5, 6, 7, 8]",
    "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
    "[1,2,3]"
  ],
  "private_tests_output": [
    "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", 
    "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n", "null\n"
  ]
}
```

### Datasample2



```json
{
  "idx": 0,
  "num": 0,
  "title": "",
  "difficulty": "Easy",
  "source_code": "```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic int[] twoSum(int[] nums, int target) {\n    Map<Integer, Integer> map = new HashMap<>();\n    for (int i = 0; i < nums.length; i++) {\n        int complement = target - nums[i];\n        if (map.containsKey(complement)) {\n            return new int[]{map.get(complement), i};\n        }\n        map.put(nums[i], i);\n    }\n    throw new IllegalArgumentException(\"No two sum solution\");\n}\n```",
  "source_lang": "java",
  "target_lang": "c++",
  "public_tests_input": "nums = [2,7,11,15], target = 9",
  "public_tests_output": "[0,1]",
  "private_tests_input": [
    "[100,200,300,400,500]\n700",
    "[-1, -2, -3, -4, -5]\n8",
    "[10000000000000000,20000000000000000,30000000000000000,40000000000000000,50000000000000000]\n90000000000000000",
    "[1000000,2000000,3000000,4000000,5000000]\n8000000",
    "[1,2,3,4,5,6,7,8,9,10]\n26",
    "[100000000000000,200000000000000,300000000000000,400000000000000,500000000000000]\n800000000000000",
    "[10000000,20000000,30000000,40000000,50000000]\n90000000",
    "[3, 2, 4]\n6",
    "[100,200,300,400,500]\n600",
    "[3,2,4]\n6",
    "[1,2,3,4,5,6,7,8,9,10]\n31",
    "[1,2,3,4,5,6,7,8,9,10]\n16",
    "[3,3]\n6",
    "[1000,2000,3000,4000,5000]\n5000",
    "[1,2,3,4,5,6,7,8,9,10]\n21",
    "[0, 4, 3, 0]\n0",
    "[1,2,3,4,5,6,7,8,9,10]\n15",
    "[1,2,3,4,5,6,7,8,9,10]\n23",
    "[1, 2, 3, 4]\n5",
    "[1,3,5,7,9]\n8",
    "[1000,2000,3000,4000,5000]\n7000",
    "[5,10,15,20,25]\n30",
    "[1,3,5,7,9]\n14",
    "[1,2,3,4,5,6,7,8,9,10]\n25",
    "[1, 1, 1, 1, 1, 1, 1]\n2",
    "[1,2,3,4,5,6,7,8,9,10]\n29",
    "[1,2,3,4,5,6,7,8,9,10]\n19",
    "[2,4,6,8,10]\n20",
    "[200,300,400,500,600]\n900",
    "[1,2,3,4,5,6,7,8,9,10]\n20",
    "[2,7,11,15]\n9",
    "[10000,20000,30000,40000,50000]\n60000",
    "[1,2,3,4,5,6,7,8,9,10]\n18",
    "[1,2,3,4,5,6,7,8,9,10]\n28",
    "[10000000000,20000000000,30000000000,40000000000,50000000000]\n60000000000",
    "[2,4,6,8,10]\n16",
    "[10, 20, 30, 40]\n50",
    "[1,2,3,4,5,6,7,8,9,10]\n17",
    "[100000000,200000000,300000000,400000000,500000000]\n1000000000",
    "[1000000000000000000,2000000000000000000,3000000000000000000,4000000000000000000,5000000000000000000]\n10000000000000000000",
    "[1,2,3,4,5,6,7,8,9,10]\n30",
    "[1,2,3,4,5]\n7",
    "[2,4,6,8,10]\n10",
    "[1, 6, 3, 4, 3, 3]\n7",
    "[1,3,5,7,9]\n18",
    "[100000,200000,300000,400000,500000]\n700000",
    "[1,2,3,4,5,6,7,8,9,10]\n24",
    "[1,2,3,4,5,6,7,8,9,10]\n19",
    "[1, 2, 3, 4, 5, 6]\n7",
    "[1,2,3,4,5,6,7,8,9,10]\n27",
    "[1,2,3,4,5,6,7,8,9,10]\n22",
    "[1000000000000,2000000000000,3000000000000,4000000000000,5000000000000]\n7000000000000",
    "[100000000000000000000,200000000000000000000,300000000000000000000,400000000000000000000,500000000000000000000]\n1100000000000000000000",
    "[2,7,11,15]\n9",
    "[100,200,300,400,500]\n600",
    "[2000,3000,4000,5000,6000]\n9000",
    "[1000,2000,3000,4000,5000]\n8000"
  ],
  "private_tests_output": [
    "[2, 4]",
    "[1, 2]",
    "[2, 3]",
    "[1, 3]",
    "[0, 1]",
    "[]",
    "[2, 4]",
    "[2, 3]",
    "[2, 3]",
    "[6, 8]",
    "[]",
    "[8, 9]",
    "[]",
    "[1, 3]",
    "[2, 4]",
    "[0, 3]",
    "[7, 9]",
    "[]",
    "[1, 3]",
    "[1, 2]",
    "[]",
    "[6, 7]",
    "[]",
    "[3, 4]",
    "[1, 2]",
    "[2, 4]",
    "[0, 1]",
    "[]",
    "[]",
    "[]",
    "[7, 8]",
    "[2, 4]",
    "[]",
    "[2, 3]",
    "[1, 2]",
    "[]",
    "[2, 3]",
    "[1, 2]",
    "[2, 3]",
    "[]",
    "[0, 1]",
    "[1, 3]",
    "[2, 3]",
    "[1, 2]",
    "[]",
    "[1, 2]",
    "[0, 1]",
    "[]",
    "[0, 1]",
    "[1, 3]",
    "[]",
    "[]",
    "[3, 4]",
    "[8, 9]",
    "[]",
    "[]",
    "[2, 3]"
  ],
  "target_code": "```cpp\n#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) {\n    std::unordered_map<int, int> map;\n    for (int i = 0; i < nums.size(); i++) {\n        int complement = target - nums[i];\n        if (map.find(complement) != map.end()) {\n            return {map[complement], i};\n        }\n        map[nums[i]] = i;\n    }\n    return {};\n}\n```"
}
```

### Datasample3



```json
{
  "idx": 0,
  "num": 0,
  "title": "",
  "difficulty": "Easy",
  "source_code": "```cpp\n#include <vector>\n#include <unordered_map>\n\nstd::vector<int> twoSum(std::vector<int>& nums, int target) {\n    std::unordered_map<int, int> map;\n    for (int i = 0; i < nums.size(); i++) {\n        int complement = target - nums[i];\n        if (map.find(complement) != map.end()) {\n            return {map[complement], i};\n        }\n        map[nums[i]] = i;\n    }\n    return {};\n}\n```",
  "source_lang": "c++",
  "average_running_time": 0,
  "average_memory": 0,
  "public_tests_input": " nums = \\[2,7,11,15\\], target = 9\n",
  "public_tests_output": " \\[0,1\\]\n",
  "private_tests_input": [
    "[100,200,300,400,500]\n700",
    "[-1, -2, -3, -4, -5]\n8",
    "[10000000000000000,20000000000000000,30000000000000000,40000000000000000,50000000000000000]\n90000000000000000",
    "[1000000,2000000,3000000,4000000,5000000]\n8000000",
    "[1,2,3,4,5,6,7,8,9,10]\n26",
    "[100000000000000,200000000000000,300000000000000,400000000000000,500000000000000]\n800000000000000",
    "[10000000,20000000,30000000,40000000,50000000]\n90000000",
    "[3, 2, 4]\n6",
    "[100,200,300,400,500]\n600",
    "[3,2,4]\n6",
    "[1,2,3,4,5,6,7,8,9,10]\n31",
    "[1,2,3,4,5,6,7,8,9,10]\n16",
    "[3,3]\n6",
    "[1000,2000,3000,4000,5000]\n5000",
    "[1,2,3,4,5,6,7,8,9,10]\n21",
    "[0, 4, 3, 0]\n0",
    "[1,2,3,4,5,6,7,8,9,10]\n15",
    "[1,2,3,4,5,6,7,8,9,10]\n23",
    "[1, 2, 3, 4]\n5",
    "[1,3,5,7,9]\n8",
    "[1000,2000,3000,4000,5000]\n7000",
    "[5,10,15,20,25]\n30",
    "[1,3,5,7,9]\n14",
    "[1,2,3,4,5,6,7,8,9,10]\n25",
    "[1, 1, 1, 1, 1, 1, 1]\n2",
    "[1,2,3,4,5,6,7,8,9,10]\n29",
    "[1,2,3,4,5,6,7,8,9,10]\n19",
    "[2,4,6,8,10]\n20",
    "[200,300,400,500,600]\n900",
    "[1,2,3,4,5,6,7,8,9,10]\n20",
    "[2,7,11,15]\n9",
    "[10000,20000,30000,40000,50000]\n60000",
    "[1,2,3,4,5,6,7,8,9,10]\n18",
    "[1,2,3,4,5,6,7,8,9,10]\n28",
    "[10000000000,20000000000,30000000000,40000000000,50000000000]\n60000000000",
    "[2,4,6,8,10]\n16",
    "[10, 20, 30, 40]\n50",
    "[1,2,3,4,5,6,7,8,9,10]\n17",
    "[100000000,200000000,300000000,400000000,500000000]\n1000000000",
    "[1000000000000000000,2000000000000000000,3000000000000000000,4000000000000000000,5000000000000000000]\n10000000000000000000",
    "[1,2,3,4,5,6,7,8,9,10]\n30",
    "[1,2,3,4,5]\n7",
    "[2,4,6,8,10]\n10",
    "[1, 6, 3, 4, 3, 3]\n7",
    "[1,3,5,7,9]\n18",
    "[100000,200000,300000,400000,500000]\n700000",
    "[1,2,3,4,5,6,7,8,9,10]\n24",
    "[1,2,3,4,5,6,7,8,9,10]\n19",
    "[1, 2, 3, 4, 5, 6]\n7",
    "[1,2,3,4,5,6,7,8,9,10]\n27",
    "[1,2,3,4,5,6,7,8,9,10]\n22",
    "[1000000000000,2000000000000,3000000000000,4000000000000,5000000000000]\n7000000000000",
    "[100000000000000000000,200000000000000000000,300000000000000000000,400000000000000000000,500000000000000000000]\n1100000000000000000000",
    "[2,7,11,15]\n9",
    "[100,200,300,400,500]\n600",
    "[2000,3000,4000,5000,6000]\n9000",
    "[1000,2000,3000,4000,5000]\n8000"
  ],
  "private_tests_output": [
    "[2, 4]\n",
    "[1, 2]\n",
    "[2, 3]\n",
    "[1, 3]\n",
    "[0, 1]\n",
    "[]\n",
    "[2, 4]\n",
    "[2, 3]\n",
    "[2, 3]\n",
    "[6, 8]\n",
    "[]\n",
    "[8, 9]\n",
    "[]\n",
    "[1, 3]\n",
    "[2, 4]\n",
    "[0, 3]\n",
    "[7, 9]\n",
    "[]\n",
    "[1, 3]\n",
    "[1, 2]\n",
    "[]\n",
    "[6, 7]\n",
    "[]\n",
    "[3, 4]\n",
    "[1, 2]\n",
    "[2, 4]\n",
    "[0, 1]\n",
    "[]\n",
    "[]\n",
    "[]\n",
    "[7, 8]\n",
    "[2, 4]\n",
    "[]\n",
    "[2, 3]\n",
    "[1, 2]\n",
    "[]\n",
    "[2, 3]\n",
    "[1, 2]\n",
    "[2, 3]\n",
    "[]\n",
    "[0, 1]\n",
    "[1, 3]\n",
    "[2, 3]\n",
    "[1, 2]\n",
    "[]\n",
    "[1, 2]\n",
    "[0, 1]\n",
    "[]\n",
    "[0, 1]\n",
    "[1, 3]\n",
    "[]\n",
    "[]\n",
    "[3, 4]\n",
    "[8, 9]\n",
    "[]\n",
    "[]\n",
    "[2, 3]\n"
  ]
}
```



### Datasample4



```json
{
  "num": 0,
  "language": "python",
  "similar_source_code": "```python\ndef minEatingSpeed(piles, h):\n    left, right = 1, max(piles)\n    while left < right:\n        mid = left + (right - left) // 2\n        totalHours = sum((pile + mid - 1) // mid for pile in piles)\n        if totalHours > h:\n            left = mid + 1\n        else:\n            right = mid\n    return left\n```",
  "similar_id": 907,
  "target_id": 1504,
  "pair_id": ["907", "1504"],
  "pair_title": [
    "Sum of Subarray Minimums",
    "Count Submatrices With All Ones"
  ],
  "similar_content": "Given an array of integers arr, find the sum of `min(b)`, where `b` ranges over every (contiguous) subarray of `arr`. Since the answer may be large, return the answer **modulo** `10^9 + 7`.\n\n**Example 1:**\n**Input:** arr = [3,1,2,4]\n**Output:** 17\n**Explanation:** \nSubarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. \nMinimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1. Sum is 17.\n\n**Example 2:**\n**Input:** arr = [11,81,94,43,3]\n**Output:** 444\n\n**Constraints:**\n* 1 <= arr.length <= 3 * 10^4\n* 1 <= arr[i] <= 3 * 10^4",
  "target_content": "Given an `m x n` binary matrix `mat`, return the number of **submatrices** that have all ones.\n\n**Example 1:**\n**Input:** mat = [[1,0,1],[1,1,0],[1,1,0]]\n**Output:** 13\n**Explanation:**\nThere are 6 rectangles of size 1x1.\nThere are 2 rectangles of size 1x2.\nThere are 3 rectangles of size 2x1.\nThere is 1 rectangle of size 2x2.\nThere is 1 rectangle of size 3x1.\nTotal = 6 + 2 + 3 + 1 + 1 = 13.\n\n**Example 2:**\n**Input:** mat = [[0,1,1,0],[0,1,1,1],[1,1,1,0]]\n**Output:** 24\n\n**Constraints:**\n* 1 <= m, n <= 150\n* mat[i][j] is either 0 or 1",
  "public_similar_tests_input": "arr = [3,1,2,4]",
  "public_similar_tests_output": "17",
  "public_target_tests_input": "mat = [[1,0,1],[1,1,0],[1,1,0]]",
  "public_target_tests_output": "13",
  "private_target_tests_input": [
    "[[1,1,1,1,0],[1,1,1,0,1],[1,0,1,1,1]]",
    "[[1,1,0],[0,1,1],[0,0,1]]",
    "[[1,1,1],[1,1,1],[1,1,1]]",
    "[[1,1,1,1,0],[0,0,0,1,1],[1,1,1,0,0]]",
    "[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]",
    "[[0,0,0,0,1],[0,0,0,1,0],[0,1,1,0,0]]",
    "[[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]]",
    "[[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]]",
    "[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]]",
    "[[1,0,1],[0,1,0],[1,0,1]]",
    "[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]",
    "[[0,0,0],[0,0,0],[0,0,0]]",
    "[[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]]",
    "[[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]]",
    "[[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]]",
    "[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]]",
    "[[1,1,1,1,1],[1,1,1,1,0],[1,1,1,0,0],[1,1,0,0,0],[1,0,0,0,0]]",
    "[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]]",
    "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]",
    "[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[0,0,0,0,1],[1,1,1,1,0],[0,0,0,1,1]]",
    "[[1,1,1,1,1],[1,0,1,0,1],[1,1,1,1,1]]",
    "[[1,1,1,1,0],[0,0,0,1,1],[1,1,1,0,0]]",
    "[[0,0,0,0,1],[1,1,1,1,0],[0,0,0,1,1]]",
    "[[1,1,1,1,0],[0,0,0,1,1],[1,1,1,0,0]]",
    "[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]]",
    "[[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[1,1,1,0],[0,1,1,1],[1,0,1,1]]",
    "[[0,0,0,0,1],[1,1,1,1,0],[0,0,0,1,1]]",
    "[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[0,0,0,0,1],[1,1,1,1,0],[0,0,0,1,1]]",
    "[[1,0],[0,1]]",
    "[[0,0,0,0,1],[1,1,1,1,0],[0,0,0,1,1]]",
    "[[1,0,1],[1,1,0],[1,1,0]]",
    "[[1,1,1,1,0],[1,0,0,1,1],[1,1,1,0,0]]",
    "[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]",
    "[[1,1,1,1,0],[1,1,1,0,1],[1,0,1,1,1]]"
  ],
  "private_target_tests_output": [
    "15\n", "36\n", "30\n", "20\n", "70\n", "15\n", "45\n", "30\n", "5\n", "15\n", "24\n", "35\n", "5\n", "15\n", "20\n", "30\n", "9\n", "0\n", "0\n", "20\n", "13\n", "15\n", "15\n", "30\n", "30\n", "30\n", "0\n", "35\n", "23\n", "7\n", "8\n", "7\n", "8\n", "42\n", "90\n", "15\n", "15\n", "15\n", "2\n"
  ],
  "target_source_code": "```python\ndef numSubmat(mat: List[List[int]]) -> int:\n    m, n = len(mat), len(mat[0])\n    dp = [[0] * n for _ in range(m)]\n    ans = 0\n\n    for i in range(m):\n        for j in range(n):\n            if mat[i][j]:\n                dp[i][j] = 1 if j == 0 else dp[i][j - 1] + 1\n                width = dp[i][j]\n                for k in range(i, -1, -1):\n                    width = min(width, dp[k][j])\n                    ans += width\n\n    return ans\n```",
  "idx": 0
}
```

