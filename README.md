# NanoSciKGQA: A Question Answering Dataset for Benchmarking Nanoscience Knowledge of Large Language Models
## 🔬 Introduction
The NanoSciKGQA dataset is a dedicated resource designed to advance Question-Answering (QA) capabilities within the rapidly evolving domain of nanoscience and nanotechnology. We introduce a two-phase Knowledge Graph (KG) construction methodology to manage knowledge depth and ensure factual rigor. This project includes a comprehensive benchmark evaluating the intrinsic domain knowledge of modern Large Language Models (LLMs).

## ✨ Key Features & Contributions
- Specialized Domain Focus: The first large-scale KGQA dataset dedicated exclusively to nanoscience and nanotechnology research.
- Dual Ground-Truth Answers: Each question is accompanied by two curated answers, offering varied perspectives crucial for nuanced LLM evaluation:
- Phase 1 Answers: Concise and high-precision.
- Phase 2 Answers: Comprehensive and contextually rich.

## 📊 NanoSciKGQA Dataset Statistics
| Metric | Value | Description | 
| ----- | ----- | ----- | 
| **Domain** | Nanoscience & Nanotechnology | Focuses on advanced research from academic abstracts. | 
| **Size** | 6,214 | Total number of open-ended questions. | 
| **Answer Sets** | 2 per question | Phase 1 (Concise) and Phase 2 (Comprehensive). | 
| **Question Types** | 8 Distinct Types | Questions categorized to ensure comprehensive coverage of functional, structural, and procedural nanoscience aspects. |

## 🛠️ Methodology: GraphRAG and Answer Generation
1. NanoSciKG: constructed from article topics and abstracts through a structured entity extraction process using LLM. Entities were categorized into:
   - Nanoscience Concepts: (e.g., nanotechnology, nanodevice, TEM, AFM)
   - Properties and Attributes: (e.g., functional mechanism, material components).
2. Synergized GraphRAG Retrieval: Answers were generated using a GraphRAG method featuring two parallel retrievers:
   
| Retriever | Function | Key Mechanism | 
| ----- | ----- | ----- | 
| **Vector Index Similarity** | Semantic Matching | Embeds all KG nodes/relationships using mxbai-embed-large and performs vector similarity search against the embedded question. | 
| **Subgraph Retriever** | Relational Context | Extracts subject entities using LLM, maps them to the KG, and retrieves all two-hop neighbors to form a contextual subgraph. | 

## 📈 LLM Evaluation Benchmark
We benchmarked seven open-source LLMs using a mixed evaluation suite (lexical metrics like Rouge-1/RoBERTa F1 and a qualitative LLM-as-a-Judge metric).

Key findings include:
- Lexical Bias: Models like Qwen-3:4B: scored highly in lexical metrics, often due to general, ambiguous answers that achieve partial keyword overlap, despite lacking deep domain knowledge.
- Qualitative Superiority: GPT-oos:20B obtained lower lexical scores but was rated highest by the LLM-as-a-Judge. This suggests GPT-oos:20B excels at abstractive synthesis and superior knowledge integration, paraphrasing accurate information rather than recalling literal keywords.

## ⬇️ Download 
[NanoSciKGQA dataset](https://huggingface.co/datasets/borwoeihuang/NanoSciKGQA/resolve/main/NanoSciKGQA.csv)
