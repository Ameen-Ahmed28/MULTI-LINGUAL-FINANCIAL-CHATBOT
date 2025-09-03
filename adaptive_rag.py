# Enhanced Adaptive RAG with LangGraph (2025 Version)
# Integrates with your existing Flask Financial Chatbot

import os
from typing import List, Literal, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain imports (2025 compatible)
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

# LangGraph imports (latest version)
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

# ======================= ENHANCED STATE DEFINITION =======================
class AdaptiveRAGState(TypedDict):
    """Enhanced state for Adaptive RAG with comprehensive tracking"""
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    language: str
    query_classification: Literal["simple", "moderate", "complex", "current_events", "non_finance"]
    
    # Retrieval tracking
    vectorstore_docs: List[str]
    vectorstore_confidence: float
    web_search_docs: List[str]
    
    # Generation tracking
    generated_answer: str
    is_grounded: bool
    addresses_question: bool
    
    # Adaptive routing
    retrieval_attempts: int
    max_retrieval_attempts: int
    final_answer: str

# ======================= PYDANTIC MODELS FOR STRUCTURED OUTPUT =======================
class QueryClassification(BaseModel):
    """Classification of user query for routing"""
    classification: Literal["simple", "moderate", "complex", "current_events", "non_finance"] = Field(
        description="Classify the query complexity and type"
    )
    confidence: float = Field(description="Confidence in classification (0-1)")
    reasoning: str = Field(description="Brief reasoning for classification")

class DocumentRelevance(BaseModel):
    """Assessment of document relevance"""
    is_relevant: bool = Field(description="Whether document is relevant to query")
    confidence: float = Field(description="Confidence in relevance (0-1)")

class GroundingCheck(BaseModel):
    """Check if generation is grounded in documents"""
    is_grounded: bool = Field(description="Whether answer is grounded in provided documents")

class AnswerQuality(BaseModel):
    """Assessment of answer quality"""
    addresses_question: bool = Field(description="Whether answer addresses the original question")
    quality_score: float = Field(description="Quality score (0-1)")

# ======================= ENHANCED COMPONENTS =======================
class AdaptiveRAGSystem:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0
        )
        
        # Initialize embeddings and vectorstore (use your existing)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        # Web search tool
        self.web_search = DuckDuckGoSearchRun()
        
        # Trusted financial sites
        self.trusted_sites = [
            'investopedia.com', 'bloomberg.com', 'reuters.com', 'cnbc.com',
            'economictimes.indiatimes.com', 'marketwatch.com', 
            'finance.yahoo.com', 'moneycontrol.com'
        ]
        
        # Load your existing vectorstore
        self.load_vectorstore()
        
        # Create structured LLM instances
        self.classifier_llm = self.llm.with_structured_output(QueryClassification)
        self.relevance_llm = self.llm.with_structured_output(DocumentRelevance)
        self.grounding_llm = self.llm.with_structured_output(GroundingCheck)
        self.quality_llm = self.llm.with_structured_output(AnswerQuality)
    
    def load_vectorstore(self):
        """Load your existing vectorstore"""
        vectorstore_path = "financial_vectorstore"
        if os.path.exists(vectorstore_path):
            try:
                self.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("✅ Loaded existing vectorstore")
            except Exception as e:
                print(f"❌ Failed to load vectorstore: {e}")
                self.vectorstore = None
        else:
            self.vectorstore = None
    
    def is_finance_related(self, query: str) -> bool:
        """Enhanced finance detection"""
        finance_keywords = [
            'stock', 'bond', 'investment', 'finance', 'financial', 'market',
            'banking', 'loan', 'credit', 'insurance', 'portfolio', 'trading',
            'economics', 'economy', 'inflation', 'interest', 'dividend',
            'mutual fund', 'etf', 'cryptocurrency', 'forex', 'derivatives',
            'options', 'futures', 'black-scholes', 'valuation', 'roi',

            # Additional terms from comprehensive financial glossaries
            'accounts payable', 'accounts receivable', 'amortisation', 'assets', 'audit',
            'balance sheet', 'benchmark', 'beta', 'bookkeeping', 'capital gain',
            'capital market', 'cash flow', 'collateral', 'compound interest',
            'cost of capital', 'credit risk', 'debt', 'depreciation', 'diversification',
            'earnings', 'equity', 'expense', 'financial statements', 'foreclosure',
            'fund manager', 'gross income', 'hedging', 'income statement',
            'inflation rate', 'initial public offering', 'interest rate',
            'internal rate of return', 'investment grade', 'issuer', 'leverage',
            'liability', 'liquidity', 'loan-to-value', 'margin', 'maturity',
            'monetary policy', 'mutual fund', 'net income', 'net present value',
            'operating expenses', 'pension', 'portfolio management', 'preferred stock',
            'price to earnings ratio', 'principal', 'profit margin', 'rate of return',
            'risk premium', 'shareholder equity', 'short selling', 'stock market index',
            'stockholder', 'takeover', 'taxation', 'treasury bills', 'underwriter',
            'valuation method', 'volatility', 'wacc', 'working capital'
        ]
        return any(keyword in query.lower() for keyword in finance_keywords)

# ======================= NODE FUNCTIONS =======================
def query_classifier(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Classify query for adaptive routing"""
    print("🔍 CLASSIFYING QUERY")
    query = state["query"]
    
    # Check if finance-related first
    if not rag_system.is_finance_related(query):
        return {"query_classification": "non_finance"}
    
    # Classify complexity and type
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial query classifier. Classify queries as:
        - simple: Basic definitions, straightforward concepts
        - moderate: Calculations, comparisons, analysis
        - complex: Multi-step reasoning, advanced concepts
        - current_events: Recent market news, today's prices, current trends
        
        Consider query complexity, required reasoning steps, and time sensitivity."""),
        ("user", "Classify this financial query: {query}")
    ])
    
    chain = classification_prompt | rag_system.classifier_llm
    result = chain.invoke({"query": query})
    
    print(f"📊 Classification: {result.classification} (confidence: {result.confidence:.2f})")
    print(f"💭 Reasoning: {result.reasoning}")
    
    return {"query_classification": result.classification}

def vectorstore_retrieval(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> Command:
    """Enhanced vectorstore retrieval with confidence scoring"""
    print("📚 VECTORSTORE RETRIEVAL")
    query = state["query"]
    
    if not rag_system.vectorstore:
        print("❌ No vectorstore available")
        return Command(
            update={"vectorstore_docs": [], "vectorstore_confidence": 0.0},
            goto="web_search_retrieval"
        )
    
    try:
        # Get documents with scores (implement your cosine similarity logic here)
        docs_with_scores = rag_system.vectorstore.similarity_search_with_score(query, k=3)
        
        relevant_docs = []
        total_confidence = 0
        
        for doc, score in docs_with_scores:
            # Convert to confidence (use your cosine similarity conversion)
            confidence = max(0.0, 1.0 - (score / 2.0))  # Adjust based on your distance metric
            
            if confidence >= 0.65:  # Your threshold
                content = doc.metadata.get("answer", doc.page_content)
                relevant_docs.append(content)
                total_confidence += confidence
                print(f"✅ Retrieved doc (confidence: {confidence:.3f})")
            else:
                print(f"❌ Rejected doc (confidence: {confidence:.3f})")
        
        avg_confidence = total_confidence / len(relevant_docs) if relevant_docs else 0.0
        
        if relevant_docs and avg_confidence >= 0.7:
            return Command(
                update={
                    "vectorstore_docs": relevant_docs,
                    "vectorstore_confidence": avg_confidence
                },
                goto="generate_answer"
            )
        else:
            print("🔄 Vectorstore confidence too low, trying web search")
            return Command(
                update={"vectorstore_docs": [], "vectorstore_confidence": avg_confidence},
                goto="web_search_retrieval"
            )
            
    except Exception as e:
        print(f"❌ Vectorstore error: {e}")
        return Command(
            update={"vectorstore_docs": [], "vectorstore_confidence": 0.0},
            goto="web_search_retrieval"
        )

def web_search_retrieval(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Enhanced web search with site restrictions"""
    print("🌐 WEB SEARCH RETRIEVAL")
    query = state["query"]
    
    # Try trusted financial sites
    for site in rag_system.trusted_sites[:3]:  # Try first 3 sites
        site_query = f"site:{site} {query}"
        try:
            result = rag_system.web_search.run(site_query)
            if result and len(result) > 100:  # Reasonable content length
                print(f"✅ Found content from {site}")
                return {
                    "web_search_docs": [result],
                    "vectorstore_docs": state.get("vectorstore_docs", [])
                }
        except Exception as e:
            print(f"❌ Error searching {site}: {e}")
    
    # Fallback: general search
    try:
        sites_or = " OR ".join([f"site:{site}" for site in rag_system.trusted_sites])
        full_query = f"{query} ({sites_or})"
        result = rag_system.web_search.run(full_query)
        if result:
            print("✅ Found content from general search")
            return {"web_search_docs": [result]}
    except Exception as e:
        print(f"❌ Web search error: {e}")
    
    print("❌ No web search results")
    return {"web_search_docs": []}

def generate_answer(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Generate answer with available context"""
    print("🤖 GENERATING ANSWER")
    
    query = state["query"]
    language = state.get("language", "english")
    vectorstore_docs = state.get("vectorstore_docs", [])
    web_docs = state.get("web_search_docs", [])
    
    # Combine all available context
    all_context = vectorstore_docs + web_docs
    
    if all_context:
        # Generate with context
        context_text = "\n\n".join(all_context)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a financial expert. Answer the query in {language} using the provided context.
            
            Format your answer professionally with:
            - **Bold** headings and key terms
            - Clear structure and numbered points
            - Relevant examples when appropriate
            - Proper financial terminology
            
            Only use information from the provided context. If context is insufficient, clearly state limitations."""),
            ("user", "Context:\n{context}\n\nQuery: {query}")
        ])
        
        chain = prompt | rag_system.llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "query": query})
        print("✅ Generated answer WITH context")
        
    else:
        # Generate without context using LLM knowledge
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a financial expert. Answer the query in {language} using your knowledge.
            
            Format your answer professionally with:
            - **Bold** headings and key terms  
            - Clear structure and numbered points
            - Relevant examples when appropriate
            - Acknowledge when you don't have current/specific information"""),
            ("user", "{query}")
        ])
        
        chain = prompt | rag_system.llm | StrOutputParser()
        answer = chain.invoke({"query": query})
        print("🧠 Generated answer WITHOUT context (LLM knowledge)")
    
    return {"generated_answer": answer}

def quality_check(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> Command:
    """Check answer quality and decide next action"""
    print("🔍 QUALITY CHECK")
    
    query = state["query"]
    answer = state["generated_answer"]
    all_docs = state.get("vectorstore_docs", []) + state.get("web_search_docs", [])
    
    # Check if grounded (if we have docs)
    is_grounded = True
    if all_docs:
        grounding_prompt = ChatPromptTemplate.from_messages([
            ("system", "Determine if the answer is grounded in the provided documents. Answer 'true' only if the answer's claims are supported by the documents."),
            ("user", "Documents:\n{docs}\n\nAnswer:\n{answer}")
        ])
        
        grounding_chain = grounding_prompt | rag_system.grounding_llm
        grounding_result = grounding_chain.invoke({
            "docs": "\n\n".join(all_docs),
            "answer": answer
        })
        is_grounded = grounding_result.is_grounded
        print(f"📋 Grounding check: {'✅' if is_grounded else '❌'}")
    
    # Check if addresses question
    quality_prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine if the answer adequately addresses the original question."),
        ("user", "Question: {query}\n\nAnswer: {answer}")
    ])
    
    quality_chain = quality_prompt | rag_system.quality_llm
    quality_result = quality_chain.invoke({"query": query, "answer": answer})
    addresses_question = quality_result.addresses_question
    
    print(f"❓ Addresses question: {'✅' if addresses_question else '❌'}")
    
    # Decision logic
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_retrieval_attempts", 2)
    
    if is_grounded and addresses_question:
        return Command(
            update={
                "is_grounded": is_grounded,
                "addresses_question": addresses_question,
                "final_answer": answer
            },
            goto="finalize"
        )
    elif attempts < max_attempts:
        print(f"🔄 Retrying retrieval (attempt {attempts + 1}/{max_attempts})")
        return Command(
            update={
                "is_grounded": is_grounded,
                "addresses_question": addresses_question,
                "retrieval_attempts": attempts + 1
            },
            goto="rewrite_query"
        )
    else:
        print("⚠️ Max attempts reached, using current answer")
        return Command(
            update={
                "is_grounded": is_grounded,
                "addresses_question": addresses_question,
                "final_answer": answer
            },
            goto="finalize"
        )

def rewrite_query(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Rewrite query for better retrieval"""
    print("✏️ REWRITING QUERY")
    
    original_query = state["query"]
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query optimizer for financial information retrieval.
        Rewrite the query to be more specific and better suited for semantic search.
        Focus on key financial concepts and terminology."""),
        ("user", "Original query: {query}\n\nProvide an improved version:")
    ])
    
    chain = rewrite_prompt | rag_system.llm | StrOutputParser()
    rewritten_query = chain.invoke({"query": original_query})
    
    print(f"📝 Rewritten: '{original_query}' → '{rewritten_query}'")
    
    return {"query": rewritten_query}

def reject_non_finance(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Handle non-finance queries"""
    print("🚫 REJECTING NON-FINANCE QUERY")
    
    language = state.get("language", "english")
    
    decline_messages = {
        'english': "I'm a specialized Financial Assistant focused on finance, investments, banking, and economics. Please ask me a finance-related question.",
        'hindi': "मैं वित्त, निवेश, बैंकिंग और अर्थशास्त्र पर केंद्रित एक विशेष वित्तीय सहायक हूं। कृपया मुझसे वित्त संबंधी प्रश्न पूछें।"
    }
    
    return {"final_answer": decline_messages.get(language, decline_messages['english'])}

def finalize_answer(state: AdaptiveRAGState, rag_system: AdaptiveRAGSystem) -> AdaptiveRAGState:
    """Finalize the answer"""
    print("✅ FINALIZING ANSWER")
    return {"final_answer": state["generated_answer"]}

# ======================= GRAPH CONSTRUCTION =======================
def create_adaptive_rag_graph():
    """Create the Adaptive RAG workflow graph"""
    
    rag_system = AdaptiveRAGSystem()
    
    # Create the graph
    workflow = StateGraph(AdaptiveRAGState)
    
    # Add nodes with system binding
    workflow.add_node("classify_query", lambda state: query_classifier(state, rag_system))
    workflow.add_node("vectorstore_retrieval", lambda state: vectorstore_retrieval(state, rag_system))
    workflow.add_node("web_search_retrieval", lambda state: web_search_retrieval(state, rag_system))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, rag_system))
    workflow.add_node("quality_check", lambda state: quality_check(state, rag_system))
    workflow.add_node("rewrite_query", lambda state: rewrite_query(state, rag_system))
    workflow.add_node("reject_non_finance", lambda state: reject_non_finance(state, rag_system))
    workflow.add_node("finalize", lambda state: finalize_answer(state, rag_system))
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Add routing logic
    def route_classification(state: AdaptiveRAGState) -> Literal["reject_non_finance", "vectorstore_retrieval", "web_search_retrieval"]:
        classification = state["query_classification"]
        if classification == "non_finance":
            return "reject_non_finance"
        elif classification == "current_events":
            return "web_search_retrieval"
        else:
            return "vectorstore_retrieval"
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify_query",
        route_classification,
        {
            "reject_non_finance": "reject_non_finance",
            "vectorstore_retrieval": "vectorstore_retrieval", 
            "web_search_retrieval": "web_search_retrieval"
        }
    )
    
    # vectorstore_retrieval uses Command for routing
    # web_search_retrieval → generate_answer
    workflow.add_edge("web_search_retrieval", "generate_answer")
    
    # generate_answer → quality_check  
    workflow.add_edge("generate_answer", "quality_check")
    
    # quality_check uses Command for routing
    # rewrite_query → vectorstore_retrieval
    workflow.add_edge("rewrite_query", "vectorstore_retrieval")
    
    # End nodes
    workflow.add_edge("reject_non_finance", END)
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# ======================= INTEGRATION WITH FLASK APP =======================
def integrate_adaptive_rag():
    """Integration function for your Flask app"""
    
    # Create the adaptive RAG graph
    adaptive_rag = create_adaptive_rag_graph()
    
    def process_query(query: str, language: str = "english") -> str:
        """Process a query through the Adaptive RAG system"""
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "language": language,
            "query_classification": "simple",
            "vectorstore_docs": [],
            "vectorstore_confidence": 0.0,
            "web_search_docs": [],
            "generated_answer": "",
            "is_grounded": True,
            "addresses_question": True,
            "retrieval_attempts": 0,
            "max_retrieval_attempts": 2,
            "final_answer": ""
        }
        
        try:
            # Run the adaptive RAG workflow
            result = adaptive_rag.invoke(initial_state)
            return result.get("final_answer", "Sorry, I couldn't process your query.")
            
        except Exception as e:
            print(f"❌ Adaptive RAG error: {e}")
            return "I encountered an error processing your query. Please try again."
    
    return process_query

# ======================= USAGE EXAMPLE =======================
if __name__ == "__main__":
    # Example usage
    process_query = integrate_adaptive_rag()
    
    # Test queries
    test_queries = [
        "What is the Black-Scholes formula?",
        "Tesla stock price today",
        "How to calculate compound interest?",
        "What's the weather like?",  # Non-finance
        "Latest news about cryptocurrency market"
    ]
    
    for query in test_queries:
        print(f"\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
        
        result = process_query(query)
        print(f"Answer: {result}")