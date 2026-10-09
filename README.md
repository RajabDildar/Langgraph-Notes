# LangGraph Learning Notes

This repository contains a collection of learning materials, experiments, and implementation examples focused on LangGraph and related LLM application patterns. It acts as a personal knowledge base and reference repo for exploring concepts such as stateful workflows, multi-step reasoning, tool use, RAG, memory, and agent-based systems.

The goal of this repo is simple: make it easy to understand how different LangGraph workflows work by studying practical code and notebooks in a structured way.

## Project Purpose

This repo is not a single app or production system. Instead, it is a learning repository where different LangGraph concepts are explored in small, focused examples. Some files are interactive notebooks used during experimentation, while others are Python scripts that demonstrate how the same ideas can be implemented in a more reusable way.

In short, this repository is a collection of notes, experiments, and examples that document the learning journey around LangGraph.

## Repository Structure

The repo is organized by workflow category and topic area.

### 1. sequential-workflow/
This folder covers workflows where tasks occur in a fixed order, one after another.

Files include:
- `prompt_chaining.ipynb` - demonstrates chaining one LLM prompt output into the next step
- `sequential_workflow1.ipynb` - basic sequential LangGraph workflow example
- `sequential_workflow2.ipynb` - additional sequential workflow patterns

Use cases:
- Prompt chaining
- Multi-step reasoning pipelines
- Linear task workflows

### 2. conditional-workflow/
This folder contains examples where the workflow branches based on conditions or dynamic decisions.

Files include:
- `feedback-workflow.ipynb` - conditional processing of feedback
- `quadratic_equation_workflow.ipynb` - workflow that routes based on mathematical condition checks

Use cases:
- Decision-making workflows
- Routing logic
- State-based branching

### 3. parallel-workflow/
This folder contains patterns where multiple tasks can run at the same time and their results are later combined.

Files include:
- `cricket_workflow.ipynb` - parallel workflow example around cricket-related analysis
- `essay_eval_workflow.ipynb` - parallel evaluation workflow across multiple dimensions

Use cases:
- Parallel LLM evaluation
- Simultaneous analysis
- Aggregating independent outputs

### 4. looping-workflow/
This folder covers iterative workflows where the system keeps refining results until a condition is satisfied.

Files include:
- `linkedin_post_generation_workflow.ipynb` - an iterative workflow for generating and improving a LinkedIn post

Use cases:
- Self-refinement loops
- Iterative content generation
- Retry and improvement cycles

### 5. Chatbot/
This is the most extensive section of the repo and contains chatbot-related examples across multiple versions and features.

It includes both Jupyter notebooks and Python scripts for different stages of chatbot development.

Key files:
- `chatbot.ipynb` - basic notebook-based chatbot experiments
- `chatbot_short_term_memory.ipynb` - chatbot with short-term memory
- `chatbot_long_term_memory.ipynb` - chatbot with long-term memory
- `chatbot_hitl.ipynb` - Human-in-the-loop chatbot example
- `chatbot_backend_v1.py` - first chatbot backend version
- `chatbot_backend_v2_with_db.py` - backend with database storage
- `chatbot_backend_v4_tools.py` - chatbot with tool invocation
- `chatbot_backend_v5_mcp.py` - MCP-based backend
- `chatbot_backend_v6_rag.py` - RAG-enabled chatbot backend
- `chatbot_backend_async.py` - asynchronous chatbot backend
- `chatbot_backend_hitl.py` - human-in-the-loop backend
- `chatbot_backend_rag.ipynb` - notebook version of RAG chatbot
- `chatbot_backend_long_term_memory.ipynb` - long-term memory notebook
- `chatbot_frontend_v1.py` through `chatbot_frontend_v6_rag.py` - progressively advanced frontend implementations
- `tools.ipynb` - examples for tool use and tool definitions
- `joke_generator.ipynb` - playful chatbot example for joke generation
- `.env.example` - example environment variables
- `requirements.txt` - dependencies for the chatbot section

This folder shows how a chatbot can evolve from a simple LLM wrapper into a more advanced system with memory, tools, search, RAG, and human oversight.

### 6. langsmith101/
This folder focuses on LangSmith usage for tracing, observability, and debugging LLM applications.

Files include:
- `1_simple_llm_call.py` - basic LLM call example
- `2_sequential_chain.py` - sequential trace example
- `3_rag_v1.py` - basic RAG implementation
- `3_rag_v2.py` - enhanced RAG implementation
- `3_rag_v3.py` - more advanced version
- `3_rag_v4.py` - refined RAG version
- `4_agent.py` - agent example
- `5_langgraph.py` - complete LangGraph application example
- `.env.example` - example environment config
- `requirements.txt` - dependencies for this section

This section is especially useful for understanding how to monitor and debug LLM workflows in production-like settings.

## Core Concepts Covered

This repo explores several concepts commonly used in modern LLM application development:

- Sequential workflows
- Conditional branching
- Parallel execution
- Looping / iterative refinement
- Prompt chaining
- Tool calling
- Retrieval-Augmented Generation (RAG)
- Memory management
- Human-in-the-loop workflows
- Model Context Protocol (MCP)
- LangSmith tracing and debugging
- Chatbot architecture evolution

## Recommended Learning Order

If you are starting from scratch, this is the simplest order to follow:

1. `sequential-workflow/` - understand linear workflows
2. `conditional-workflow/` - learn decision logic and branching
3. `parallel-workflow/` - see concurrent processing patterns
4. `looping-workflow/` - understand iterative refinement
5. `langsmith101/` - learn observability and debugging
6. `Chatbot/` - study practical chatbot architectures and advanced examples

## Setup Instructions

### Prerequisites
- Python 3.9+
- Jupyter Notebook / JupyterLab
- API keys for LLM provider(s) used in the examples
- Optional: LangSmith access for tracing examples

### Install dependencies

For the chatbot examples:

```bash
cd Chatbot
pip install -r requirements.txt
```

For the LangSmith examples:

```bash
cd langsmith101
pip install -r requirements.txt
```

### Environment variables
Many examples require environment variables such as API keys. Look for `.env.example` files and copy them into `.env` before running:

```bash
cp .env.example .env
```

Then update the file with your own keys and settings.

### Run notebooks

```bash
jupyter notebook
```

Then open the notebook you want to explore.

## Why This Repo Exists

This repo is meant to be a practical reference for learning LangGraph. It is especially useful for:

- understanding different LangGraph flow patterns
- learning through working examples rather than only theory
- experimenting with chatbot patterns, memory, tools, and RAG
- debugging and monitoring LLM workflows with LangSmith
- revisiting past experiments when building new systems

## Notes

This repository is intentionally hands-on. It is designed as a personal knowledge base and learning workspace, not a polished product. Many files are examples and experiments, which is exactly what makes them valuable for learning.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Summary

`Langgraph-Notes` is a personal learning repository for exploring LangGraph workflows and LLM application architecture. It includes notebooks and scripts for sequential, conditional, parallel, looping, chatbot, and LangSmith-based workflows, making it a useful reference for anyone learning LangGraph in a practical way.
