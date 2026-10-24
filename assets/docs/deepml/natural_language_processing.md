# Deep-ML Knowledge Base: Natural Language Processing

Total verified problems: 7

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 51: Optimal String Alignment Distance
- **Difficulty**: `medium`
- **Category**: NLP

### Description & Spec
In this problem, you need to implement a function that calculates the [Optimal String Alignment (OSA)](https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance) distance between two given strings. The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:

- Insert a character
- Delete a character
- Substitute a character
- Transpose two adjacent characters

Each of these operations costs 1 unit.

Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).

For example, the OSA distance between the strings `caper` and `acer` is 2: one deletion (removing "p") and one transposition (swapping "a" and "c").

### Mathematical Derivation & Intuition
## Optimal String Alignment Distance

Given two strings \( s_1 \) and \( s_2 \), find the Optimal String Alignment (OSA) distance between them.

The OSA distance gives the minimum number of edits needed to transform string \( s_1 \) into \( s_2 \). Here are the allowed edit operations:

1. **Insert a character**
2. **Delete a character**
3. **Substitute a character**
4. **Transpose two adjacent characters**

Each operation has a cost of 1 unit.

For example, the OSA distance between the strings "caper" and "acer" is 2:
- One deletion (removing the letter 'p')
- One transposition (swapping the letters 'a' and 'c')

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def OSA(source: str, target: str) -> int:
    source_len, target_len = len(source), len(target)

    # Initialize matrix with zeros
    osa_matrix = [[0] * (target_len + 1) for _ in range(source_len + 1)]

    # Fill the first row and first column with index values
    for j in range(1, target_len + 1):
        osa_matrix[0][j] = j
    for i in range(1, source_len + 1):
        osa_matrix[i][0] = i

    # Compute the OSA distance
    for i in range(1, source_len + 1):
        for j in range(1, target_len + 1):
            osa_matrix[i][j] = min(
                osa_matrix[i - 1][j] + 1,  # Deletion
                osa_matrix[i][j - 1] + 1,  # Insertion
                osa_matrix[i - 1][j - 1] + (1 if source[i - 1] != target[j - 1] else 0)  # Substitution
            )
            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                osa_matrix[i][j] = min(osa_matrix[i][j], osa_matrix[i - 2][j - 2] + 1)  # Transposition

    return osa_matrix[-1][-1]
```

### Verified Sample Trace Data
```json
{
  "test": "source = \"butterfly\"\ntarget = \"dragonfly\"\noutput = OSA(source, target)\nprint(output)",
  "expected_output": "6"
}
```

---

## Problem 60: Implement TF-IDF (Term Frequency-Inverse Document Frequency)
- **Difficulty**: `medium`
- **Category**: NLP

### Description & Spec
## Task: Implement TF-IDF (Term Frequency-Inverse Document Frequency)

Your task is to implement a function that computes the TF-IDF scores for a query against a given corpus of documents.

### Function Signature

Write a function `compute_tf_idf(corpus, query)` that takes the following inputs:

- `corpus`: A list of documents, where each document is a list of words.
- `query`: A list of words for which you want to compute the TF-IDF scores.

### Output

The function should return a list of lists containing the TF-IDF scores for the query words in each document, rounded to five decimal places.

### Important Considerations

1. **Handling Division by Zero:**  
   When implementing the Inverse Document Frequency (IDF) calculation, you must account for cases where a term does not appear in any document (`df = 0`). This can lead to division by zero in the standard IDF formula. Add smoothing (e.g., adding 1 to both numerator and denominator) to avoid such errors.

2. **Empty Corpus:**  
   Ensure your implementation gracefully handles the case of an empty corpus. If no documents are provided, your function should either raise an appropriate error or return an empty result. This will ensure the program remains robust and predictable.

3. **Edge Cases:**  
   - Query terms not present in the corpus.  
   - Documents with no words.  
   - Extremely large or small values for term frequencies or document frequencies.

By addressing these considerations, your implementation will be robust and handle real-world scenarios effectively.

### Mathematical Derivation & Intuition
# Understanding TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a numerical statistic that reflects how important a word is in a document relative to a collection (or corpus). It is widely used in information retrieval, text mining, and natural language processing tasks.

## Mathematical Formulation

TF-IDF is the product of two key statistics: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

### 1. Term Frequency (TF)

The term frequency is defined as:

$TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$

- $t$: A specific term (word).
- $d$: A specific document in the corpus.

### 2. Inverse Document Frequency (IDF)

To account for how common or rare a term is across all documents in the corpus, we calculate:

$IDF(t) = \log\Bigl(\frac{N + 1}{\text{df}(t) + 1}\Bigr) + 1$

Where:

- $N$: Total number of documents in the corpus.
- $\text{df}(t)$: Number of documents containing the term $t$.
- Adding $+1$ inside the fraction prevents division by zero if a term never appears.
- Adding $+1$ outside the log ensures IDF remains nonzero.

### 3. TF-IDF

Combining TF and IDF:

$TFIDF(t, d) = TF(t, d) \times IDF(t)$

## Implementation Steps

1. **Compute TF**  
   For each document, count how often each term appears and divide by the document’s total word count.

2. **Compute IDF**  
   For each term, calculate its document frequency across all documents and apply the IDF formula.

3. **Calculate TF-IDF**  
   For every term in every document, multiply the term’s TF by its IDF.

4. **Normalization (Optional)**  
   Normalize TF-IDF vectors (e.g., using $L2$ norm) if comparing documents in a vector space model.

## Example Calculation

Suppose we have a small corpus of 3 documents:

- **Doc1**: "The cat sat on the mat"  
- **Doc2**: "The dog chased the cat"  
- **Doc3**: "The bird flew over the mat"

We want to calculate the TF-IDF for the word **"cat"** in **Doc1**.

### Step 1: Compute $TF("cat", \text{Doc1})$

$TF("cat", \text{Doc1}) = \frac{1}{6} \approx 0.1667$

- "cat" appears once.
- Total words in Doc1 (counting each occurrence of “the”) = 6.

### Step 2: Compute $IDF("cat")$

- "cat" appears in 2 out of 3 documents, so $\text{df}("cat") = 2$.
- $N = 3$.

Using the formula with smoothing and an added constant:

$IDF("cat") = \log\Bigl(\frac{N + 1}{\text{df}("cat") + 1}\Bigr) + 1 = \log\Bigl(\frac{3 + 1}{2 + 1}\Bigr) + 1 = \log\Bigl(\frac{4}{3}\Bigr) + 1 \approx 0.2877 + 1 = 1.2877$

### Step 3: Calculate $TFIDF("cat", \text{Doc1})$

$TFIDF("cat", \text{Doc1}) = TF("cat", \text{Doc1}) \times IDF("cat") = 0.1667 \times 1.2877 \approx 0.2147$

## Applications of TF-IDF

1. **Information Retrieval**  
   TF-IDF is often used in search engines to rank how relevant a document is to a given query.
2. **Text Mining**  
   Helps identify key terms and topics in large volumes of text.
3. **Document Classification**  
   Useful for weighting important words in classification tasks.
4. **Search Engines**  
   Refines document ranking by emphasizing distinctive terms.
5. **Recommendation Systems**  
   Evaluates text-based similarity (e.g., for content-based filtering).

TF-IDF remains a foundational technique in natural language processing, widely used for feature extraction and analysis across numerous text-based applications.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents using only NumPy.
    The output TF-IDF scores retain five decimal places.
    """
    vocab = sorted(set(word for document in corpus for word in document).union(query))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    tf = np.zeros((len(corpus), len(vocab)))

    for doc_idx, document in enumerate(corpus):
        for word in document:
            word_idx = word_to_index[word]
            tf[doc_idx, word_idx] += 1
        tf[doc_idx, :] /= len(document)

    df = np.count_nonzero(tf > 0, axis=0)

    num_docs = len(corpus)
    idf = np.log((num_docs + 1) / (df + 1)) + 1

    tf_idf = tf * idf

    query_indices = [word_to_index[word] for word in query]
    tf_idf_scores = tf_idf[:, query_indices]

    tf_idf_scores = np.round(tf_idf_scores, 5)

    return tf_idf_scores.tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\ncorpus = [\n    [\"the\", \"cat\", \"sat\", \"on\", \"the\", \"mat\"],\n    [\"the\", \"dog\", \"chased\", \"the\", \"cat\"],\n    [\"the\", \"bird\", \"flew\", \"over\", \"the\", \"mat\"]\n]\nquery = [\"cat\"]\n\nprint(compute_tf_idf(corpus, query))",
  "expected_output": "[[0.21461], [0.25754], [0.0]]"
}
```

---

## Problem 88: GPT-2 Text Generation
- **Difficulty**: `hard`
- **Category**: NLP

### Description & Spec
### Implement a Simplified GPT-2-like Text Generation Function

You are tasked with implementing a simplified GPT-2-like text generation function in Python. This function will incorporate the following components of a minimal GPT-2 architecture:

- **Token Embeddings**: Map input tokens to dense vector representations.
- **Positional Embeddings**: Add positional information to token embeddings.
- **Multi-head Attention**: Attend to various parts of the sequence.
- **Feed-Forward Network**: Process attention outputs through a dense layer.
- **Layer Normalization**: Stabilize the training process.

The function must take in the following parameters:

1. **Prompt**: The initial text to guide the generation process.
2. **Number of Tokens to Generate**: Specify how many tokens to output.

Your function should output the generated text.

Additionally, utilize the helper function `load_encoder_hparams_and_params` to retrieve:

- A dummy encoder.
- Model hyperparameters.
- Model parameters.

Build your text generation logic around these components. This exercise is designed to help you understand the core concepts behind GPT-2's autoregressive text generation.

### Mathematical Derivation & Intuition
# Understanding Transformer Architecture and Text Generation

Transformers have revolutionized the field of Natural Language Processing (NLP) with their efficient and scalable architecture. This guide provides an in-depth look into the core components of transformers and how they facilitate advanced text generation.

## 1. Introduction to Transformers

Transformers are a groundbreaking neural network architecture that has significantly advanced NLP. Introduced in the seminal paper *"Attention is All You Need"* by Vaswani et al. (2017), transformers have outperformed traditional models like Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) in various NLP tasks.

### Key Advantages of Transformers

- **Parallel Processing**:  
  Unlike RNNs, which process input sequences sequentially, transformers handle entire sequences simultaneously. This parallelism leads to substantial improvements in training speed and efficiency.

- **Scalability**:  
  Transformers can effectively scale to handle large datasets and complex tasks, making them ideal for applications like language translation, text generation, and summarization.

- **Self-Attention Mechanism**:  
  The core innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence relative to each other. This capability enables the model to capture long-range dependencies and contextual relationships within the text.

### Applications of Transformers

- **Text Generation**: Creating coherent and contextually relevant text based on a given prompt.
- **Machine Translation**: Translating text from one language to another with high accuracy.
- **Text Summarization**: Condensing long documents into concise summaries while retaining key information.
- **Question Answering**: Providing accurate answers to user queries based on contextual understanding.

---

## 2. Core Concepts

To fully grasp the transformer architecture, it's essential to understand its foundational components. Below are the core concepts that constitute the building blocks of transformers:

### 2.1 GELU Activation Function

The Gaussian Error Linear Unit (GELU) is an advanced activation function that enhances the performance of deep neural networks.

**Mathematical Expression**:  
$$
\text{GELU}(x) = 0.5 \cdot x \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)))
$$

**Purpose**:  
GELU introduces non-linearity in the network while maintaining smooth gradient flow. Unlike the Rectified Linear Unit (ReLU) or Sigmoid functions, GELU provides a probabilistic approach to activation, allowing for better handling of uncertainty and improving model performance in deep architectures.

**Benefits**:
- **Smooth Activation**: Reduces the likelihood of "dead neurons" that can occur with ReLU.
- **Improved Gradient Flow**: Facilitates more stable and efficient training by preventing gradient vanishing or exploding.

### 2.2 Softmax for Attention

Softmax is a fundamental function used to convert raw attention scores into a probability distribution, ensuring that the weights sum to one.

**Mathematical Expression**:  
$$
\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)}
$$

**Purpose**:  
In the context of attention mechanisms, Softmax normalizes the attention scores, enabling the model to focus on relevant parts of the input sequence by assigning higher weights to more important tokens.

**Example**:  
If the attention scores for a sentence are `[2, 1, 0.1]`, applying Softmax will convert these to probabilities like `[0.659, 0.242, 0.099]`, indicating the relative importance of each token.

### 2.3 Layer Normalization

Layer normalization stabilizes and accelerates the training process by standardizing the inputs across the features.

**Mathematical Expression**:  
$$
\text{LayerNorm}(x) = g \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b
$$

Where:
- \( \mu \): Mean of input \( x \) along the last axis.
- \( \sigma^2 \): Variance of \( x \).
- \( g, b \): Learnable scaling and bias parameters.
- \( \epsilon \): A small constant to prevent division by zero.

**Purpose**:  
By normalizing the inputs, layer normalization ensures that each layer receives inputs with a consistent distribution, which enhances training stability and convergence speed.

---

### 2.4 Multi-Head Attention

Multi-head attention is an extension of the attention mechanism that allows the model to focus on different representation subspaces simultaneously.

**Components**:
- **Query (Q), Key (K), Value (V) Matrices**: Each attention head computes its own set of Q, K, and V matrices by projecting the input embeddings into different subspaces.
- **Scaled Dot-Product Attention**:
  $$
  \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  $$

**Benefits**:
- **Diversity of Attention**: Allows the model to focus on different parts of the input simultaneously.
- **Enhanced Representation**: Captures richer features by aggregating multiple attention heads.

---

### 2.5 Feedforward Network (FFN)

The Feedforward Network is a simple yet powerful component applied to each position independently within the transformer.

**Mathematical Expression**:  
$$
\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))
$$

**Structure**:
1. First Linear Layer: Projects the input to a higher-dimensional space.
2. GELU Activation: Introduces non-linearity to the model.
3. Second Linear Layer: Projects the data back to the original dimensionality.

**Purpose**:  
The FFN enhances the model's capacity to learn intricate patterns.

---

### 2.6 Transformer Block

A transformer block is the fundamental building unit of the transformer architecture, combining multi-head attention and the feedforward network with residual connections and layer normalization.

**Structure**:
- **Multi-Head Attention Layer**:  
  $$ x_1 = \text{LayerNorm}(x + \text{MHA}(x)) $$
- **Feedforward Network**:  
  $$ x_2 = \text{LayerNorm}(x_1 + \text{FFN}(x_1)) $$

**Advantages**:
- **Deep Architecture Support**: Facilitates the construction of deep networks without significant performance degradation.
- **Modularity**: Each transformer block can be stacked multiple times, allowing for scalable model depth.

---

### 2.7 GPT-2 Text Generation

GPT-2 (Generative Pre-trained Transformer 2) leverages the transformer architecture for generating human-like text. Developed by OpenAI, GPT-2 has demonstrated remarkable capabilities in various NLP tasks.

**Key Components**:
- **Word and Positional Embeddings**: Captures semantic meaning and token position in a sequence.
- **Causal Attention**: Ensures left-to-right text generation by masking future tokens.
- **Stacked Transformer Blocks**: Refines input representations iteratively.

**Text Generation Process**:
1. Provide a prompt to initiate the process.
2. Tokenize the input into embeddings.
3. Process embeddings through transformer blocks.
4. Generate the next token based on probabilities.
5. Repeat steps 3-4 to produce coherent text.

---

### Conclusion

Transformers have fundamentally transformed NLP by introducing efficient and scalable architectures capable of handling complex language tasks. Understanding their core components such as GELU activation, Softmax attention, layer normalization, multi-head attention, feedforward networks, and the transformer block provides a foundation for leveraging these models in various applications. GPT-2 exemplifies the transformative power of these architectures while highlighting ethical considerations for their use.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    np.random.seed(42)  # Set the random seed for reproducibility
    encoder, hparams, params = load_encoder_hparams_and_params()
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nprint(gen_text(\"hello\", n_tokens_to_generate=5))",
  "expected_output": "hello hello hello <UNK> <UNK>"
}
```

---

## Problem 90: BM25 Ranking 
- **Difficulty**: `medium`
- **Category**: NLP

### Description & Spec
Implement the BM25 ranking function to calculate document scores for a query in an information retrieval context. BM25 is an advanced variation of TF-IDF that incorporates term frequency saturation, document length normalization, and a configurable penalty for document length effects.

### Mathematical Derivation & Intuition
## **BM25**

BM25 (Best Match 25) is used in information retrieval for search relevance. Similar to TF-IDF, it reflects the importance of a word in a document within a collection or corpus. However, BM25 improves upon TF-IDF by addressing key limitations.

### Limitations of TF-IDF Addressed by BM25

1. **Saturation**: In TF-IDF, having a term multiple times in a document skews the term frequency, making the document overly relevant. BM25 mitigates this by using:  
   $$ \text{TF-adjusted} = \frac{\text{TF}}{\text{TF} + k_1} $$  

2. **Document Length Normalization**: BM25 accounts for document length by normalizing term frequencies using:  
   $$ \text{Normalized Length} = 1 - b + b \times \frac{\text{Doc Len}}{\text{Average Doc Len}} $$  

3. **Amplifying Parameter**: The $b$ parameter controls the influence of document length normalization. Higher $b$ values amplify the effect.

### Final BM25 Formula

The BM25 score for a term is given by:  
$$ \text{BM25} = \text{IDF} \times \frac{\text{TF} \times (k_1 + 1)} {\text{TF} + k_1 \times (1 - b + b \times \frac{\text{dl}}{\text{adl}})} $$  

Where:  
- $ \text{TF} $: Term frequency in the document.  
- $ \text{IDF} $: Inverse document frequency, calculated as $ \log(\frac{N + 1}{\text{df} + 1}) $.  
- $ N $: Total number of documents.  
- $ \text{df} $: Number of documents containing the term.  
- $ \text{dl} $: Document length.  
- $ \text{adl} $: Average document length.  
- $ k_1 $: Saturation parameter.  
- $ b $: Normalization parameter.

### Implementation Steps

1. Compute document length ($ dl $) and average document length ($ adl $).  
2. Calculate term frequencies ($ TF $) using the BM25 formula.  
3. Compute inverse document frequencies ($ IDF $) for each term.  
4. Calculate BM25 scores for each document.

### Applications

BM25 is widely used in:  
- Search Engines  
- Recommendation Systems  
- Natural Language Processing (NLP)  

Understanding BM25 enables the creation of robust systems for search and ranking tasks.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query cannot be empty")

    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    doc_freqs = Counter()
    for doc in corpus:
        doc_freqs.update(set(doc))

    scores = np.zeros(len(corpus))
    N = len(corpus)

    for term in query:
        df = doc_freqs.get(term, 0) + 1
        idf = np.log((N + 1) / df)

        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue

            tf = term_counts[term]
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            scores[idx] += idf * term_score

    return np.round(scores, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_bm25_scores([['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']], ['the', 'cat']))",
  "expected_output": "[0.693, 0., 0. ]"
}
```

---

## Problem 110: Evaluate Translation Quality with METEOR Score
- **Difficulty**: `medium`
- **Category**: NLP

### Description & Spec
Develop a function to compute the METEOR score for evaluating machine translation quality. Given a reference translation and a candidate translation, calculate the score based on unigram matches, precision, recall, F-mean, and a penalty for word order fragmentation.

### Mathematical Derivation & Intuition
METEOR(Metric for Evaluation of Translation with Explicit ORdering) is a metric generally used for 
machine translation and evaluating the text output of generative AI models. METEOR build was introduced to address 
the limitations in earlier metrics like BLEU.

## Key Characteristics
- Considers semantic similarity beyond exact word matching
- Accounts for word order and translation variations
- Provides more human-aligned translation assessment

# Implementation 
1. **Tokenization**

2. **Frequency of matching words** : Matching needs to be exact

3. **Calculate Precision, Recall and F-mean**
```
   F_mean = (Precision * Recall) / 
   (alpha * Precision + (1 - alpha) * Recall)
```
   - alpha typically set to 0.9
   - Balances precision and recall

4. **Fragmentation Penalty**
   ```
   Chunks = Count of contiguous matched word sequences
   Penalty = gamma * (Chunks / Matches)^β
   ```
   - beta controls penalty weight (typically 3)
   - gamma limits maximum penalty (typically 0.5)

5. **Final METEOR Score**
   ```
   METEOR = F_mean * (1 - Penalty)
   ```
   - Ranges from 0 (no match) to 1 (perfect match)

**__Note__** : The [paper](https://aclanthology.org/W05-0909/) that introduced the metric doesn't have the parameters (alpha,β, and gamma) as tunable parameters, but implementation in other libraries like NLTK offers this flexibility.

# Example 

- Reference: "The quick brown fox jumps over the lazy dog"
- Candidate: "A quick brown fox jumps over a lazy dog"

### 1. Tokenization
- Reference Tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
- Candidate Tokens: ['a', 'quick', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'dog']

### 2. Unigram Matching
- Matching tokens: ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
- Matches: 7

### 3. Unigram Precision and Recall Calculation
- Precision = Matches / Candidate Length = 7 / 9 ~ 0.778

- Recall = Matches / Reference Length = 7 / 9 ~ 0.778

### 4. F-mean Calculation (alpha = 0.9)
```
F_mean = (Precision * Recall) / 
(alpha * Precision + (1 - alpha) * Recall)
       = (0.778 * 0.778) / (0.9 * 0.778 + (1 - 0.9) * 0.778)
       = 0.606 / (0.7 + 0.078)
       = 0.606 / 0.778
       ≈ 0.779
```

### 5. Chunk Calculation
- Contiguous matched sequences:
  1. ['quick', 'brown', 'fox']
  2. ['jumps', 'over']
  3. ['lazy', 'dog']
- Number of Chunks: 3
- Total Number of Unigram Matches: 7

### 6. Penalty Calculation (betta = 3, gamma = 0.5)
```
Penalty = gamma * 
(Number of Chunks / Total Number of Unigram Matches)^betta
        = 0.5 * (3 / 7)^3
        = 0.5 * (0.429)^3
        ≈ 0.039
```

### 7. Final METEOR Score
```
METEOR = F_mean * (1 - Penalty)
       = 0.779 * (1 - 0.039)
       = 0.779 * 0.961
       ≈ 0.749
```

### Verified Pure NumPy Solution (From Scratch)
```python
"import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    if not reference or not candidate:
        raise ValueError("Reference and candidate cannot be empty")
    
    # Tokenize and count
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Counter for unigram for reference and candidate 
    ref_counts = Counter(ref_tokens) 
    cand_counts = Counter(cand_tokens)
    
    # Calculate matches
    num_matches = sum((ref_counts & cand_counts).values()) # Number of matching words in candidate and reference 
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)  

    # Unigram Precision and Recall 
    precision = num_matches / cand_len if cand_len > 0 else 0 # Avoiding Division by zero
    recall = num_matches / ref_len if ref_len > 0 else 0 # Avoiding Division by zero 
    
    if num_matches == 0:
        return 0.0
    
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Chunk calculation 
    matched_positions = []
    ref_positions = {}  # Store positions of words in reference
    used_positions = set()  # Track already used indices

    # Populate reference positions for word alignment tracking
    for i, word in enumerate(ref_tokens):
        ref_positions.setdefault(word, []).append(i)

    # Determine the sequence of matched positions in reference
    for word in cand_tokens:
        if word in ref_positions:
            for pos in ref_positions[word]:
                if pos not in used_positions:
                    matched_positions.append(pos)
                    used_positions.add(pos)
                    break  # Ensure each match is used only once

    # Count chunks by detecting breaks in position sequence
    num_chunks = 1 if matched_positions else 0
    for i in range(1, len(matched_positions)):
        if matched_positions[i] != matched_positions[i - 1] + 1:
            num_chunks += 1  # Break in sequence → new chunk

    # Fragmentation penalty
    penalty = gamma * ((num_chunks / num_matches) ** beta) if num_matches > 0 else 0
    
    # Final score
    return round(fmean * (1 - penalty), 3) # Rounding to 3 Decimal places
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(meteor_score('The dog barks at the moon', 'The dog barks at the moon'),3))",
  "expected_output": "0.998"
}
```

---

## Problem 111: Compute Pointwise Mutual Information
- **Difficulty**: `medium`
- **Category**: NLP

### Description & Spec
Implement a function to compute the Pointwise Mutual Information (PMI) given the joint occurrence count of two events, their individual counts, and the total number of samples. PMI measures how much the actual joint occurrence of events differs from what we would expect by chance.

### Mathematical Derivation & Intuition
# Pointwise Mutual Information (PMI)

Pointwise Mutual Information (PMI) is a statistical measure used in information theory and Natural Language Processing (NLP) to quantify the association between two events. It measures how much the actual joint occurrence of two events differs from what would be expected if they were independent. PMI is commonly used for identifying word associations, feature selection in text classification, and calculating document similarity.

## Implementation 

1. **Collect Count Data** for events $x$, $y$, and their joint occurrence $(x,y)$.

2. **Calculate Individual Probabilities**:
   - $P(x) = \frac{\text{Count}(x)}{\text{Total Count}}$
   - $P(y) = \frac{\text{Count}(y)}{\text{Total Count}}$

3. **Calculate Joint Probability**:
   - $P(x,y) = \frac{\text{Count}(x,y)}{\text{Total Count}}$

4. **Calculate PMI**:
   - $$\text{PMI}(x,y) = \log_2\left(\frac{P(x,y)}{P(x) \cdot P(y)}\right)$$

## Interpretation of PMI Values

- **Positive PMI**: Events co-occur more frequently than expected by chance.
- **Zero PMI**: Events are statistically independent.
- **Negative PMI**: Events co-occur less frequently than expected by chance.
- **Undefined PMI**: Occurs when $P(x,y) = 0$ (the events never co-occur).

## Variants of PMI

### 1. Normalized PMI (NPMI)

NPMI scales PMI to a range of [-1, 1] to account for dataset size variations:

$$
\text{NPMI}(x,y) = \frac{\text{PMI}(x,y)}{-\log_2 P(x,y)}
$$

### 2. Positive PMI (PPMI)

PPMI sets negative PMI scores to zero, often used in word embeddings:

$$
\text{PPMI}(x,y) = \max(\text{PMI}(x,y),\,0)
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):

    if not all(isinstance(x, int) and x >= 0 for x in [joint_counts, total_counts_x, total_counts_y, total_samples]):
        raise ValueError("All inputs must be non-negative integers.")

    if total_samples == 0:
        raise ValueError("Total samples cannot be zero.")

    if joint_counts > min(total_counts_x, total_counts_y):
        raise ValueError("Joint counts cannot exceed individual counts.")

    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    p_xy = joint_counts / total_samples

    if p_xy == 0:
        return float('-inf')

    pmi = np.log2(p_xy / (p_x * p_y))

    return round(pmi, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "print(compute_pmi(10, 50, 50, 200))",
  "expected_output": "-0.322"
}
```

---

## Problem 129: Calculate Unigram Probability from Corpus
- **Difficulty**: `easy`
- **Category**: NLP

### Description & Spec
Implement a function that calculates the unigram probability of a given word in a corpus of sentences. Include start `<s>` and end `</s>` tokens in the calculation. The probability should be rounded to 4 decimal places.

### Mathematical Derivation & Intuition
# Unigram Probability Calculation

- In Natural Language Processing (NLP), a  unigram model is the simplest form of a language model. 
- It assumes each word in a sentence is generated independently.  


- The probability of a word w under the unigram model is:

$P(w) = \frac{\text{Count}(w)}{\sum_{w' \in V} \text{Count}(w')}$

Where:

- $\text{Count}(w)$ = Number of times the word w appears in the corpus.

- $V$ = Vocabulary (all word tokens in the corpus).

- $\sum_{w' \in V} \text{Count}(w')$ = Total number of word tokens.
- Round upto the 4th decimal point.


---

### Sample Corpus

```text
<s> I am Jack </s>
<s> Jack I am </s>
<s> Jack I like </s>
<s> Jack I do like </s>
<s> do I like Jack </s>
```

Notes : 
- \<s> : Start of a sentence
- \</s> : End of a sentence
- Need to count both the start and enod of sentence tokens while calculating probability.
- Zero probability issues are not addressed here and will be covered separately under smoothing techniques in  later problems.

### Verified Pure NumPy Solution (From Scratch)
```python
def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split()
    total_word_count = len(tokens)
    word_count = tokens.count(word)
    return round(word_count / total_word_count, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "corpus = \"\"\"<s> I am Jack </s> <s> Jack I am </s> <s> Jack I like </s> <s> Jack I do like </s> <s> do I like Jack </s>\"\"\"\nprint(round(unigram_probability(corpus, \"Jack\"),4))",
  "expected_output": "0.1852"
}
```

---

