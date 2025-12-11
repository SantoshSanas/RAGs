# RAG Systems with LangGraph 🤖📚

Advanced Retrieval-Augmented Generation (RAG) implementations using LangGraph, demonstrating intelligent document retrieval and reasoning patterns.

---

## 📚 Project Overview

This folder contains three progressive implementations of RAG systems using **LangGraph** and **LangChain**:

### **Progression Level**
```
1. Agentic RAG      (Intermediate) - Agent-driven retrieval
    ↓
2. Corrective RAG   (Advanced) - Relevance checking & rewriting
    ↓
3. Adaptive RAG     (Expert) - Multi-step routing & adaptive logic
```

---

## 🔍 Notebooks Detailed Breakdown

### **1. Agentic RAG** (`1-AgenticRAG.ipynb`)

**What it does:**
- Uses an LLM agent to decide when and how to retrieve documents
- Agent has access to multiple retriever tools
- Makes intelligent decisions about document relevance
- Combines multiple knowledge bases (LangGraph docs + LangChain docs)

**Key Concepts:**
- Agent-based decision making
- Multiple retriever tools
- Tool binding with LLM
- State management with messages

**Workflow:**
```
User Query
    ↓
Agent (LLM with tools)
    ├→ Decides: Retrieve or answer directly?
    ├→ If retrieve: Uses retriever tools
    └→ Process results and respond
```

**Architecture:**
- **2 Separate Vector DBs:**
  - LangGraph documentation (FAISS)
  - LangChain documentation (FAISS)
- **Retriever Tools:** Agent chooses which knowledge base to query
- **LLM:** Groq (qwen/qwen3-32b)
- **Embeddings:** Ollama (mxbai-embed-large)

**Key Features:**
- ✅ Multiple knowledge sources
- ✅ Agent-driven retrieval decisions
- ✅ Tool selection based on query intent
- ✅ Message-based state management

---

### **2. Corrective RAG** (`2-CorrectiveRAG.ipynb`)

**What it does:**
- Retrieves documents AND evaluates their relevance
- If documents aren't relevant, rewrites the query
- Continues retrieving until relevant documents found
- Generates answer only from relevant docs

**Key Concepts:**
- Retrieval + Relevance Grading
- Query rewriting for improved retrieval
- Adaptive looping based on relevance scores
- Quality control in RAG pipeline

**Workflow:**
```
User Query
    ↓
Retrieve Documents
    ↓
Grade Relevance (LLM evaluates)
    ├→ Relevant? → Generate Answer
    └→ Not relevant? → Rewrite Query → Retrieve again
```

**Architecture:**
- **Retrieval:** Web-based document loading
- **Grading:** Pydantic-based relevance classifier
- **Rewriting:** LLM-powered query transformation
- **Generation:** RAG answer synthesis

**Key Features:**
- ✅ Relevance scoring (binary yes/no)
- ✅ Query rewriting on low relevance
- ✅ Looping retrieval until success
- ✅ Quality-checked answers

---

### **3. Adaptive RAG** (`3-AdaptiveRAG.ipynb`)

**What it does:**
- Routes queries based on complexity
- Uses different retrieval strategies for different query types
- Adapts approach based on query analysis
- Most sophisticated RAG implementation

**Key Concepts:**
- Query routing and classification
- Adaptive strategy selection
- Multi-route RAG pipeline
- Dynamic decision trees

**Workflow:**
```
User Query
    ↓
Classify Query Type
    ├→ Web Search → Use web results directly
    ├→ Local Retrieval → Query vector DB
    ├→ Fallback → No retrieval needed
    └→ Generate answer based on strategy
```

**Architecture:**
- **Query Classification:** Route based on intent
- **Multiple Strategies:**
  - Web search integration
  - Vector DB retrieval
  - Direct LLM response
  - Hybrid approaches
- **Adaptive Selection:** Choose best strategy per query

**Key Features:**
- ✅ Intelligent query routing
- ✅ Multiple retrieval strategies
- ✅ Adaptive decision making
- ✅ Optimized for different query types

---

## 📊 Comparison Table

| Feature | Agentic RAG | Corrective RAG | Adaptive RAG |
|---------|-------------|----------------|--------------|
| **Complexity** | Intermediate | Advanced | Expert |
| **Agent Decision** | ✅ Tool selection | ❌ Relevance grading | ✅ Query routing |
| **Query Rewriting** | ❌ | ✅ Dynamic | ✅ Conditional |
| **Multi-tools** | ✅ 2+ retrievers | ✅ 1 retriever | ✅ Multiple strategies |
| **Relevance Check** | Implicit (agent) | Explicit (grader) | Implicit (routing) |
| **Looping** | Single pass | Loop until good | Conditional |
| **Use Case** | Multi-knowledge | Quality assurance | Complex queries |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- API Keys:
  - **Groq** (free): https://console.groq.com
  - **OpenAI** (optional): https://platform.openai.com/api-keys
- **Ollama** (for embeddings): https://ollama.ai
- Internet (for document loading)

### Installation

1. Navigate to the RAGS folder:
```bash
cd c:\Users\Admin\Documents\AgenticAI\Free_courses\Langgraph\7-RAGS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install langchain langgraph langchain-core langchain-community langchain-groq python-dotenv pydantic
```

3. Set up environment variables:

Create a `.env` file in the `7-RAGS/` folder:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key (optional)
LANGCHAIN_API_KEY=your_langchain_api_key (optional)
```

4. Ensure Ollama is running (for embeddings):
```bash
ollama run mxbai-embed-large
```

---

## 📖 Learning Path

### Recommended Order:

**1. Start with Agentic RAG** (30-45 minutes)
   - Understand basic RAG with agents
   - Learn tool binding and selection
   - See multi-source retrieval in action
   - Run: `jupyter notebook 1-AgenticRAG.ipynb`

**2. Then Corrective RAG** (45-60 minutes)
   - Learn quality control in RAG
   - Implement relevance grading
   - Understand query rewriting
   - Run: `jupyter notebook 2-CorrectiveRAG.ipynb`

**3. Finally Adaptive RAG** (60+ minutes)
   - Master routing and adaptation
   - Complex decision trees
   - Optimize for different query types
   - Run: `jupyter notebook 3-AdaptiveRAG.ipynb`

---

## 🔧 How to Run Each Notebook

### Run Agentic RAG
```bash
jupyter notebook 1-AgenticRAG.ipynb
```
**Key cells to execute:**
1. Imports and setup
2. Load LangGraph docs
3. Load LangChain docs
4. Create retriever tools
5. Build graph with agent
6. Test with queries

**Example queries:**
- "What is LangGraph?"
- "How do I use LangChain?"

### Run Corrective RAG
```bash
jupyter notebook 2-CorrectiveRAG.ipynb
```
**Key cells to execute:**
1. Setup and imports
2. Load documents
3. Create retriever
4. Define grade_documents function
5. Build graph with grading
6. Invoke with test queries

**Example queries:**
- "What are the best practices for RAG?"
- "How do I optimize retrieval?"

### Run Adaptive RAG
```bash
jupyter notebook 3-AdaptiveRAG.ipynb
```
**Key cells to execute:**
1. Imports and configuration
2. Query classifier setup
3. Multiple strategy definitions
4. Build adaptive graph
5. Route and invoke

**Example queries:**
- "Latest news on AI" (web search)
- "What is RAG?" (local retrieval)

---

## 🏗️ Architecture Patterns

### Pattern 1: Agentic Decision (Notebook 1)
```
LLM Agent with Tools
    ↓
Agent decides which tool to use
    ↓
Execute selected tool
    ↓
Process result
```

### Pattern 2: Quality Loop (Notebook 2)
```
Retrieve Documents
    ↓
Grade Relevance
    ├→ Good? → Answer
    └→ Bad? → Rewrite & Retry
```

### Pattern 3: Adaptive Routing (Notebook 3)
```
Classify Query
    ↓
Select Strategy
    ├→ Web search
    ├→ Vector retrieval
    ├→ Direct LLM
    └→ Hybrid
    ↓
Execute & Answer
```

---

## 🛠️ Key Components Across Notebooks

**Common Tools & Libraries:**
- **LangGraph** - Workflow orchestration
- **LangChain** - LLM integration and chains
- **FAISS** - Vector database (Notebooks 1-2)
- **WebBaseLoader** - Document loading
- **ChatGroq** - LLM provider
- **OllamaEmbeddings** - Local embeddings
- **Pydantic** - Data validation

**Common Patterns:**
- State management with TypedDict
- Message reducers with `add_messages`
- Tool nodes and conditional routing
- LLM chains with output parsing

---

## 📝 Key Concepts to Understand

### RAG (Retrieval-Augmented Generation)
- Retrieve relevant documents first
- Use documents as context for LLM
- Improve answer quality and accuracy

### Agentic RAG
- Let agent decide when/what to retrieve
- Agent has access to tools
- More flexible than traditional RAG

### Corrective RAG
- Check if retrieved docs are relevant
- Improve query if needed
- Ensure quality results

### Adaptive RAG
- Route queries to best strategy
- Different approaches for different needs
- Optimized performance

---

## 🔗 Dependencies & Versions

```txt
pydantic          - Data validation
langchain         - LLM framework
langgraph         - Workflow graphs
langchain-core    - Core abstractions
langchain-community - Integrations
langchain-groq    - Groq integration
python-dotenv     - Environment variables
arxiv             - Arxiv search (optional)
wikipedia         - Wikipedia search (optional)
```

---

## 🎯 Common Tasks

### Load Documents
```python
from langchain_community.document_loaders import WebBaseLoader
docs = WebBaseLoader(url).load()
```

### Create Vector Store
```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
```

### Create Retriever Tool
```python
from langchain_core.tools import create_retriever_tool
tool = create_retriever_tool(retriever, name, description)
```

### Build State Graph
```python
from langgraph.graph import StateGraph, START, END
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_func)
workflow.add_conditional_edges("agent", tools_condition)
graph = workflow.compile()
```

---

## ⚠️ Troubleshooting

### Issue: Import errors (hub, tools, etc.)
**Solution:** Ensure all packages are installed and up-to-date
```bash
pip install --upgrade langchain langgraph langchain-core
```

### Issue: Ollama embeddings not found
**Solution:** Make sure Ollama is running
```bash
ollama run mxbai-embed-large
```

### Issue: API key errors
**Solution:** Verify `.env` file exists with correct keys in RAGS folder
```bash
echo GROQ_API_KEY=your_key > .env
```

### Issue: WebBaseLoader fails
**Solution:** Check internet connection and URL accessibility

### Issue: Graph invocation errors
**Solution:** Check function signatures match graph expectations (state only, no extra params)

---

## 🚀 Next Steps / Enhancements

- [ ] Compare performance of all 3 approaches
- [ ] Add caching to improve speed
- [ ] Implement custom retrievers
- [ ] Add multi-turn conversation support
- [ ] Integrate with more data sources
- [ ] Add persistence/memory
- [ ] Deploy to production
- [ ] Add monitoring and logging
- [ ] Benchmark different strategies

---

## 📚 Resources

- **LangGraph Docs**: https://python.langchain.com/docs/langgraph/
- **LangChain Docs**: https://python.langchain.com/docs/
- **Groq Console**: https://console.groq.com
- **Ollama**: https://ollama.ai
- **Pydantic Docs**: https://docs.pydantic.dev/

---

## 📄 Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run specific notebook
jupyter notebook 1-AgenticRAG.ipynb
jupyter notebook 2-CorrectiveRAG.ipynb
jupyter notebook 3-AdaptiveRAG.ipynb

# Start Ollama embedding model
ollama run mxbai-embed-large

# Check if Ollama is running
curl http://localhost:11434/api/tags
```

---

## 💡 Key Takeaways

1. **Agentic RAG** - Agents can make intelligent decisions about retrieval
2. **Corrective RAG** - Quality control improves RAG reliability
3. **Adaptive RAG** - Routing strategies optimize for different query types
4. **Progressive Learning** - Each notebook builds on previous concepts
5. **LangGraph Power** - Flexible workflows for complex AI pipelines

---

**Last Updated:** December 11, 2025  
**Status:** ✅ Ready to Learn  
**Difficulty:** Intermediate → Advanced → Expert  
**Estimated Time:** 2-3 hours total
