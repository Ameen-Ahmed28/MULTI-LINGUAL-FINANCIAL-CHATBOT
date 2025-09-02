from flask import Flask, render_template, request, jsonify, session
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
import pickle
import torch
from typing import Dict, List, Optional, TypedDict
from dotenv import load_dotenv
import uuid
import requests
from bs4 import BeautifulSoup
load_dotenv()
import re
# Hugging Face datasets
from datasets import load_dataset

# LangChain imports - Updated for 0.3.x versions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated

app = Flask(__name__)
app.secret_key = os.urandom(16)

# Force CPU usage and disable warnings
torch.set_default_dtype(torch.float32)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================= STATE DEFINITION =======================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    language: str
    query_type: str
    retrieved_docs: List[str]
    generated_answer: str
    is_relevant: bool
    has_hallucination: bool
    final_answer: str

# ======================= AUDIO PROCESSOR =======================
class AudioProcessor:
    def __init__(self):
        pass
        
    def get_language_code_tts(self, language: str) -> str:
        """Get TTS language code for all Indian languages"""
        lang_map = {
            'english': 'en', 'hindi': 'hi', 'marathi': 'mr', 'tamil': 'ta', 
            'bengali': 'bn', 'gujarati': 'gu', 'kannada': 'kn', 'malayalam': 'ml', 
            'punjabi': 'pa', 'telugu': 'te', 'urdu': 'ur', 'odia': 'or',
            'assamese': 'as', 'nepali': 'ne', 'sindhi': 'sd', 'kashmiri': 'ks',
            'sanskrit': 'sa', 'maithili': 'mai', 'dogri': 'doi', 'manipuri': 'mni',
            'bodo': 'brx', 'santhali': 'sat', 'konkani': 'gom'
        }
        return lang_map.get(language.lower(), 'en')
    
    def text_to_speech(self, text: str, language: str = 'english') -> Optional[str]:
        try:
            lang_code = self.get_language_code_tts(language)
            text = re.sub(r"\*", "", text)
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Create unique filename
            filename = f"audio_{uuid.uuid4().hex}.mp3"
            filepath = os.path.join('static', 'audio', filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            tts.save(filepath)
            return filename
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

# ======================= ENHANCED FINANCE DETECTION =======================
def is_strictly_finance_related(query: str) -> bool:
    """ENHANCED check if query is finance-related with comprehensive keywords"""
    
    # COMPREHENSIVE finance keywords including advanced concepts
    finance_keywords = [
        # Core Finance
        'stock', 'stocks', 'share', 'shares', 'equity', 'equities', 'bond', 'bonds', 
        'investment', 'investments', 'invest', 'investing', 'investor', 'finance', 
        'financial', 'money', 'cash', 'currency', 'dollar', 'rupee', 'price', 'cost',
        
        # Banking & Credit
        'bank', 'banking', 'account', 'savings', 'checking', 'loan', 'loans', 
        'credit', 'debit', 'mortgage', 'emi', 'interest', 'rate', 'rates', 'apr',
        
        # Trading & Markets  
        'trading', 'trade', 'trader', 'market', 'markets', 'exchange', 'nse', 'bse',
        'nasdaq', 'dow', 's&p', 'portfolio', 'asset', 'assets', 'security', 'securities',
        
        # Investment Products
        'mutual fund', 'mutual funds', 'etf', 'etfs', 'sip', 'fd', 'fixed deposit', 
        'recurring deposit', 'dividend', 'dividends', 'capital gains', 'roi', 'returns', 'yield',
        
        # Advanced Finance & Derivatives
        'futures', 'options', 'derivatives', 'commodity', 'commodities', 'forex', 
        'currency', 'crypto', 'cryptocurrency', 'bitcoin', 'blockchain', 'derivatives',
        'black scholes', 'black-scholes', 'options pricing', 'put option', 'call option',
        'strike price', 'expiry', 'volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
        
        # Financial Planning & Analysis
        'budget', 'budgeting', 'planning', 'retirement', 'pension', 'insurance',
        'tax', 'taxation', 'deduction', 'exemption', 'wealth', 'income', 'expense',
        'cash flow', 'dcf', 'npv', 'irr', 'wacc', 'capm',
        
        # Economic Terms
        'inflation', 'deflation', 'gdp', 'economy', 'economic', 'recession', 'growth',
        'bull market', 'bear market', 'volatility', 'risk', 'diversification',
        'correlation', 'beta', 'alpha', 'sharpe ratio',
        
        # Business Finance  
        'revenue', 'profit', 'loss', 'earnings', 'valuation', 'ipo', 'lic', 'pe ratio',
        'balance sheet', 'cash flow', 'debt', 'liability', 'capital', 'working capital',
        'financial statements', 'income statement', 'ebitda', 'eps', 'book value',
        
        # Quantitative Finance & Statistics
        'formula', 'calculation', 'model', 'financial model', 'pricing model',
        'risk model', 'monte carlo', 'binomial', 'trinomial', 'stochastic',
        'mean reversion', 'arbitrage', 'hedge', 'hedging', 'statistics', 'statistical',
        'regression', 'correlation', 'variance', 'standard deviation', 'median', 'mean',
        'prediction', 'forecasting', 'analysis', 'data analysis', 'market prediction'
    ]
    
    # Financial phrases and patterns
    finance_phrases = [
        'how much', 'what is the price', 'cost of', 'value of', 'worth of',
        'should i invest', 'how to invest', 'best investment', 'financial advice',
        'money management', 'portfolio management', 'risk management',
        'financial planning', 'investment strategy', 'market analysis',
        'stock analysis', 'company valuation', 'financial ratio', 'predicting market',
        'market performance', 'statistical analysis', 'market statistics'
    ]
    
    query_lower = query.lower().strip()
    
    # Remove special characters for better matching
    query_clean = query_lower.replace('-', ' ').replace('_', ' ')
    
    # Check for finance keywords (including multi-word terms)
    has_finance_keyword = any(keyword.lower() in query_clean for keyword in finance_keywords)
    
    # Check for finance phrases
    has_finance_phrase = any(phrase.lower() in query_clean for phrase in finance_phrases)
    
    # Special checks for specific financial concepts
    special_finance_terms = [
        'black scholes', 'black-scholes', 'bs model', 'options pricing',
        'put call parity', 'binomial model', 'monte carlo simulation',
        'value at risk', 'var', 'credit risk', 'market risk',
        'financial engineering', 'quantitative finance', 'derivatives pricing',
        'statistics used in predicting market', 'statistical analysis market'
    ]
    
    has_special_term = any(term.lower() in query_clean for term in special_finance_terms)
    
    return has_finance_keyword or has_finance_phrase or has_special_term

def get_strict_decline_message(language: str) -> str:
    """Get strict decline message"""
    messages = {
        'english': """Hey, I am a specialized Financial Assistant and can ONLY answer questions related to finance, investments, banking, and economics.

I can help with:
💰 Stocks, bonds, mutual funds, ETFs
🏦 Banking, loans, savings accounts  
📊 Investment strategies and portfolio management
📈 Market analysis and trading
💳 Credit cards, EMIs, and financial products
🪙 Cryptocurrency and digital assets
📋 Tax planning and financial planning
💼 Insurance and retirement planning
⚖️ Options pricing models (Black-Scholes, Binomial)
📐 Financial formulas and calculations
📊 Statistical analysis for market prediction

Please ask me a finance-related question.""",

        'hindi': """मैं एक विशेष वित्तीय सहायक हूं और केवल वित्त, निवेश, बैंकिंग और अर्थशास्त्र से संबंधित प्रश्नों का उत्तर दे सकता हूं।

मैं इसमें मदद कर सकता हूं:
💰 शेयर, बॉन्ड, म्यूचुअल फंड, ईटीएफ
🏦 बैंकिंग, ऋण, बचत खाते
📊 निवेश रणनीति और पोर्टफोलियो प्रबंधन
📈 बाजार विश्लेषण और ट्रेडिंग
💳 क्रेडिट कार्ड, ईएमआई और वित्तीय उत्पाद
🪙 क्रिप्टोकरेंसी और डिजिटल संपत्ति
📋 टैक्स प्लानिंग और वित्तीय योजना
💼 बीमा और सेवानिवृत्ति योजना
⚖️ विकल्प मूल्य निर्धारण मॉडल (ब्लैक-स्कोल्स)
📐 वित्तीय सूत्र और गणना
📊 बाजार पूर्वानुमान के लिए सांख्यिकीय विश्लेषण

कृपया मुझसे वित्त संबंधी प्रश्न पूछें।"""
    }
    
    return messages.get(language, messages['english'])

# ======================= RAG SYSTEM =======================
def initialize_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"HuggingFace embeddings failed: {e}, using fallback")
        return SimpleEmbeddings()

class SimpleEmbeddings:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
            print("Using fallback SimpleEmbeddings")
        except Exception as e:
            print(f"Fallback embeddings failed: {e}")
            self.model = None
    
    def embed_documents(self, texts):
        if self.model is None:
            return [[0.0] * 384 for _ in texts]
        try:
            return self.model.encode(texts).tolist()
        except:
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text):
        if self.model is None:
            return [0.0] * 384
        try:
            return self.model.encode([text])[0].tolist()
        except:
            return [0.0] * 384

class PersistentFinancialRAG:
    def __init__(self):
        self.vectorstore_path = "financial_vectorstore"
        self.docs_path = "financial_docs.pkl"
        self.confidence_threshold = 0.7  # Similarity threshold
        print("Initializing embeddings...")
        self.embedding_model = initialize_embeddings()
        
        if not self.load_existing_db():
            print("Loading dataset...")
            self.docs = self.load_hf_dataset()
            print("Setting up vectorstore...")
            self.setup_vectorstore()
            print("Saving database...")
            self.save_db()
        print("RAG system ready!")
    
    # ... (keep existing: load_existing_db, save_db, load_hf_dataset, setup_vectorstore)
    def load_existing_db(self) -> bool:
        try:
            if os.path.exists(self.vectorstore_path) and os.path.exists(self.docs_path):
                print("Loading existing vectorstore...")
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                
                with open(self.docs_path, 'rb') as f:
                    self.docs = pickle.load(f)
                
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
                print(f"Loaded {len(self.docs)} documents from cache")
                return True
        except Exception as e:
            print(f"Failed to load existing DB: {e}")
            return False
        return False
    
    def save_db(self):
        try:
            if hasattr(self, 'vectorstore'):
                self.vectorstore.save_local(self.vectorstore_path)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.docs, f)
            print("Database saved successfully")
        except Exception as e:
            print(f"Failed to save DB: {e}")
    
    def load_hf_dataset(self) -> List[Dict]:
        try:
            dataset = load_dataset("Adityaaaa468/Financial_Questions")
            docs = []
            for item in dataset['train'][:5000]:
                docs.append({
                    'question': item['instruction'],
                    'answer': item['output']
                })
            print(f"Loaded {len(docs)} documents from Hugging Face")
            
            # Add WELL-FORMATTED Financial content
            additional_docs = [
                {
                    'question': 'What is the Black-Scholes formula?',
                    'answer': '''**Black-Scholes Formula**

The **Black-Scholes formula** is a mathematical model used to estimate the value of a call option or a put option, which are types of derivatives. It was first introduced by Fischer Black and Myron Scholes in 1973.

**Key Inputs:**

**1. S:** Stock Price – The current market price of the underlying stock.

**2. K:** Strike Price – The price at which the option can be exercised.

**3. T:** Time to Expiration – The time remaining until the option expires.

**4. r:** Risk-Free Rate – The interest rate on a risk-free investment, such as a U.S. Treasury bond.

**5. σ:** Volatility – The standard deviation of the stock's returns, measuring the uncertainty of the stock's price movements.

**6. d1 and d2:** Two critical values calculated using the inputs above.

**Formula for a call option:**

**C = S · N(d1) − K · e^(-rT) · N(d2)**

**Where:**
• **C** = value of the call option
• **N(d1)** and **N(d2)** are the cumulative distribution functions of the standard normal distribution

**Formula for a put option:**

**P = K · e^(-rT) · N(−d2) − S · N(−d1)**

**Where:**
• **P** = value of the put option

**Calculation of d1 and d2:**

**d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)**

**d2 = d1 - σ√T**

The Black-Scholes formula is widely used in finance to value options and derivatives. However, it assumes **constant volatility, constant interest rate, no dividends, and no complexities** like transaction costs.'''
                },
                {
                    'question': 'How statistics used in predicting market?',
                    'answer': '''**Predicting Market Performance using Statistical Analysis**

**Introduction**

Predicting market performance is a crucial aspect of investment decision-making. Statistical analysis plays a vital role in identifying trends, patterns, and relationships between various market variables.

**Key Statistics**

The following statistics are commonly used in predicting market performance:

**1. Mean and Median**
• **Mean**: The average value of a dataset, calculated as the sum of all values divided by the number of values.
• **Median**: The middle value of a dataset, separating the higher half from the lower half.

**2. Standard Deviation**
• A measure of the spread or dispersion of a dataset, calculated as the square root of the variance.
• **Variance**: The average of the squared differences from the mean.

**3. Correlation Coefficient**
• A measure of the linear relationship between two variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

**4. Regression Analysis**
• A statistical method used to establish a relationship between a dependent variable and one or more independent variables.

**Statistical Formulas**

The following formulas are used in predicting market performance:

**1. Mean and Median**
• **Mean**: **μ = (Σx) / n**
• **Where**: μ = Mean, x = Individual data points, n = Number of data points
• **Median**: **M = (x(n/2) + x((n+1)/2)) / 2**
• **Where**: M = Median, x = Individual data points, n = Number of data points

**2. Standard Deviation**
• **σ = √(Σ(x - μ)² / (n - 1))**
• **Where**: σ = Standard Deviation, x = Individual data points, μ = Mean, n = Number of data points

**3. Correlation Coefficient**
• **r = Σ[(xi - μx)(yi - μy)] / (√Σ(xi - μx)² × √Σ(yi - μy)²)**
• **Where**: r = Correlation Coefficient, xi = Individual data points of variable x, yi = Individual data points of variable y, μx = Mean of variable x, μy = Mean of variable y

**4. Regression Analysis**
• **y = β0 + β1x + ε**
• **Where**: y = Dependent variable, β0 = Intercept, β1 = Slope, x = Independent variable, ε = Error term

**Example**

Suppose we want to predict the stock price of a company using historical data. We can use regression analysis to establish a relationship between the stock price and other variables such as earnings per share (EPS) and dividend yield.

**Stock Price = β0 + β1(EPS) + β2(Dividend Yield) + ε**

**Where:**
• **Stock Price** = Dependent variable
• **EPS** = Earnings per share (independent variable)
• **Dividend Yield** = Dividend yield (independent variable)
• **β0** = Intercept
• **β1** = Slope of EPS
• **β2** = Slope of Dividend Yield
• **ε** = Error term

By using statistical analysis, we can identify the relationships between various market variables and make informed investment decisions.'''
                },
                {
                    'question': 'How do you calculate option Greeks?',
                    'answer': '''**Option Greeks**

Option Greeks measure the sensitivity of option prices to various factors:

**1. Delta (Δ)** - Price sensitivity to underlying asset price changes
• Call Delta = N(d₁)
• Put Delta = N(d₁) - 1
• **Range:** 0 to 1 for calls, -1 to 0 for puts

**2. Gamma (Γ)** - Rate of change of Delta
• Γ = φ(d₁) / (S₀σ√T)
• **Measures:** How much Delta changes for $1 move in stock

**3. Theta (Θ)** - Time decay sensitivity
• Call Theta = -[S₀φ(d₁)σ/(2√T) + rXe^(-rT)N(d₂)]
• **Measures:** Option value loss per day

**4. Vega (ν)** - Volatility sensitivity
• ν = S₀φ(d₁)√T
• **Measures:** Price change per 1% volatility change

**5. Rho (ρ)** - Interest rate sensitivity
• Call Rho = XTe^(-rT)N(d₂)
• **Measures:** Price change per 1% interest rate change

**Where:**
• φ(x) = standard normal probability density function
• All other variables same as Black-Scholes formula'''
                }
            ]
            
            docs.extend(additional_docs)
            return docs
            
        except Exception as e:
            print(f"Failed to load HF dataset: {e}")
            return [
                {
                    'question': 'What is the Black-Scholes formula?',
                    'answer': '''**Black-Scholes Formula**

The **Black-Scholes formula** calculates theoretical option prices:

**Call Option Formula:**
**C = S₀N(d₁) - Xe^(-rT)N(d₂)**

**Variables:**
• **C** = Call option price
• **S₀** = Current stock price
• **X** = Strike price  
• **r** = Risk-free rate
• **T** = Time to expiry
• **σ** = Volatility

This model revolutionized options trading and derivatives pricing.'''
                },
                {
                    'question': 'How statistics used in predicting market?',
                    'answer': '''**Statistical Analysis in Market Prediction**

**Key Statistics:**

**1. Mean and Median:** Average and middle values of datasets
**2. Standard Deviation:** Measure of data spread and volatility
**3. Correlation:** Relationship strength between variables
**4. Regression Analysis:** Predictive modeling technique

**Applications:**
• Price forecasting
• Risk assessment  
• Trend identification
• Portfolio optimization'''
                },
                {
                    'question': 'What is a stock?',
                    'answer': 'A **stock** represents ownership in a company and constitutes a claim on part of the company\'s assets and earnings.'
                }
            ]
    
    def setup_vectorstore(self):
        documents = []
        for i, doc in enumerate(self.docs):
            content = f"Question: {doc['question']}\nAnswer: {doc['answer']}"
            documents.append(Document(
                page_content=content,
                metadata={
                    'question': doc['question'], 
                    'answer': doc['answer'],
                    'doc_id': i
                }
            ))
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            print("Vectorstore created successfully")
        except Exception as e:
            print(f"Failed to setup vectorstore: {e}")
            self.retriever = None

    def retrieve_with_confidence(self, query: str, k: int = 3) -> tuple[List[str], float]:
        """SIMPLIFIED vectorstore retrieval with proper confidence scoring"""
        print(f"🔍 Vectorstore retrieval for: {query}")
        
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            print("❌ Vectorstore not available")
            return [], 0.0
        
        try:
            # Get documents with similarity scores from vectorstore
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not docs_with_scores:
                print("❌ No documents retrieved from vectorstore")
                return [], 0.0
            
            retrieved_docs = []
            total_confidence = 0
            
            for doc, score in docs_with_scores:
                if hasattr(doc, 'metadata') and 'answer' in doc.metadata:
                    retrieved_docs.append(doc.metadata['answer'])
                else:
                    retrieved_docs.append(doc.page_content)
                
                # SIMPLIFIED confidence calculation for FAISS L2 distance
                # FAISS L2 distance: lower = better, typically 0.0-2.0 for normalized embeddings
                # Convert to confidence: 0.0 distance = 1.0 confidence, 2.0 distance = 0.0 confidence
                confidence = max(0.0, 1.0 - (score / 2.0))
                total_confidence += confidence
                
                print(f"  📄 Score: {score:.4f} -> Confidence: {confidence:.4f}")
            
            avg_confidence = total_confidence / len(docs_with_scores) if docs_with_scores else 0.0
            print(f"✅ Average confidence: {avg_confidence:.4f}")
            
            return retrieved_docs, avg_confidence
            
        except Exception as e:
            print(f"❌ Vectorstore error: {e}")
            return [], 0.0

# ======================= WEB SEARCH & LLM =======================
# ======================= ENHANCED WEB SEARCH WITH GOOGLE CUSTOM SEARCH API =======================

from langchain_community.tools import DuckDuckGoSearchRun
import time

class FinancialWebSearch:
    def __init__(self):
        self.TRUSTED_SITES = [
            'investopedia.com',
            'bloomberg.com', 
            'reuters.com',
            'cnbc.com',
            'economictimes.indiatimes.com',
            'marketwatch.com',
            'finance.yahoo.com',
            'moneycontrol.com'
        ]
        self.search_tool = DuckDuckGoSearchRun()
    
    def search_financial_sources(self, query: str) -> str:
        """
        Uses DuckDuckGo (via LangChain tool) to search on trusted financial sites only.
        """
        print(f"🦆 DuckDuckGo Search for: {query}")
        # 1. Query each trusted site separately for better precision
        for site in self.TRUSTED_SITES:
            site_query = f"site:{site} {query}"
            print(f"  Searching {site}...")
            try:
                result = self.search_tool.run(site_query)
                if result and "http" in result:
                    print(f"✅ Found on {site}: {result[:120]}...")
                    return result
            except Exception as e:
                print(f"  DuckDuckGo error on {site}: {e}")
            time.sleep(0.5)  # be respectful, avoid rate limit
        
        # 2. If nothing is found, try a general search (all trusted as an OR query)
        sites_or = " OR ".join([f"site:{site}" for site in self.TRUSTED_SITES])
        full_query = f"{query} ({sites_or})"
        print(f"  General search: {full_query}")
        try:
            result = self.search_tool.run(full_query)
            if result and "http" in result:
                print(f"✅ Found via general search: {result[:120]}...")
                return result
        except Exception as e:
            print(f"  DuckDuckGo error in general search: {e}")

        print("❌ No usable DuckDuckGo search result.")
        return ""


class LangChainGroqLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key and api_key != "your-groq-api-key-here":
            try:
                self.llm = ChatGroq(
                    api_key=api_key,
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    max_tokens=500
                )
                self.available = True
                print("Groq LLM initialized successfully")
            except Exception as e:
                print(f"Groq LLM initialization failed: {e}")
                self.available = False
                self.llm = None
        else:
            print("No GROQ_API_KEY found, using fallback responses")
            self.available = False
            self.llm = None
    
    def generate(self, prompt: str, system_message: str = None) -> str:
        if not self.available or not self.llm:
            return self.enhanced_fallback_response(prompt)
        
        try:
            if system_message:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm.invoke(messages)
            else:
                response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self.enhanced_fallback_response(prompt)
    
    def enhanced_fallback_response(self, prompt: str) -> str:
        """Enhanced fallback with WELL-FORMATTED responses"""
        prompt_lower = prompt.lower()
        
        if 'statistics' in prompt_lower and ('market' in prompt_lower or 'predict' in prompt_lower):
            return '''**Statistical Analysis in Market Prediction**

**Introduction**
Predicting market performance is a crucial aspect of investment decision-making. Statistical analysis plays a vital role in identifying trends, patterns, and relationships between various market variables.

**Key Statistics:**

**1. Mean and Median**
• **Mean**: The average value of a dataset
• **Median**: The middle value separating higher and lower halves

**2. Standard Deviation**  
• Measures spread or dispersion of data
• **Formula:** **σ = √(Σ(x - μ)² / (n - 1))**

**3. Correlation Coefficient**
• Measures linear relationship between variables
• **Range:** -1 (negative) to +1 (positive correlation)

**4. Regression Analysis**
• Establishes relationships between dependent and independent variables
• **Formula:** **y = β₀ + β₁x + ε**

**Applications:**
• Price forecasting
• Risk assessment
• Trend identification  
• Portfolio optimization

Statistical models help investors make informed decisions by analyzing historical patterns and predicting future market movements.'''

        elif 'black' in prompt_lower and 'scholes' in prompt_lower:
            return '''**Black-Scholes Formula**

The **Black-Scholes formula** is a mathematical model used to estimate the value of a call option or a put option, which are types of derivatives.

**Key Inputs:**

**1. S:** Stock Price – Current market price of underlying stock
**2. K:** Strike Price – Exercise price of the option  
**3. T:** Time to Expiration – Time remaining until expiry
**4. r:** Risk-Free Rate – Interest rate on risk-free investment
**5. σ:** Volatility – Standard deviation of stock returns

**Formula for Call Option:**
**C = S · N(d₁) − K · e^(-rT) · N(d₂)**

**Formula for Put Option:**  
**P = K · e^(-rT) · N(−d₂) − S · N(−d₁)**

**Where:**
• **C** = Call option value
• **P** = Put option value
• **N()** = Cumulative normal distribution

**Calculation of d₁ and d₂:**
**d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)**
**d₂ = d₁ - σ√T**

The Black-Scholes model revolutionized options trading and derivatives pricing.'''

        elif any(word in prompt_lower for word in ['stock', 'share']):
            return '''**Stocks**

A **stock** represents ownership in a company and constitutes a claim on part of the company's assets and earnings.

**Key Aspects:**

**1. Ownership:** Partial ownership of the company
**2. Returns:** Through capital appreciation and dividends  
**3. Risk:** Price fluctuations based on market conditions
**4. Types:** Common stock (voting rights) vs Preferred stock (fixed dividends)

**Factors Affecting Stock Prices:**
• Company performance
• Market conditions
• Economic factors
• Investor sentiment'''

        elif 'option' in prompt_lower:
            return '''**Options**

**Options** are financial contracts that give the holder the right, but not the obligation, to buy or sell an underlying asset at a specific price within a certain time period.

**Types:**
**1. Call Options:** Right to BUY the asset
**2. Put Options:** Right to SELL the asset

**Key Components:**
• **Strike Price:** Exercise price
• **Expiration Date:** When option expires
• **Premium:** Cost to buy the option

Options are priced using models like **Black-Scholes** for theoretical value calculation.'''

        else:
            return "I can help with **finance-related questions** including options pricing, Black-Scholes model, statistical analysis for market prediction, stocks, bonds, and other financial concepts. Please ask me a specific finance question."

# ======================= LANGGRAPH PIPELINE =======================
def query_analysis_node(state: AgentState) -> AgentState:
    query = state["query"]
    
    if not is_strictly_finance_related(query):
        state["query_type"] = "REJECTED_NON_FINANCE"
        return state
    
    # Finance subcategorization
    current_event_keywords = ['news', 'today', 'current', 'latest', 'recent', 'trend', 'update']
    query_lower = query.lower()
    
    has_current = any(keyword in query_lower for keyword in current_event_keywords)
    
    if has_current:
        state["query_type"] = "current_events"
    else:
        state["query_type"] = "finance_query"
    
    return state

def reject_non_finance_node(state: AgentState) -> AgentState:
    language = state["language"]
    state["final_answer"] = get_strict_decline_message(language)
    return state

def vectorstore_node(state: AgentState) -> AgentState:
    """Try ONLY vectorstore retrieval"""
    query = state["query"]
    
    # Get documents with confidence score from vectorstore ONLY
    retrieved_docs, confidence = rag_system.retrieve_with_confidence(query, k=3)
    
    state["retrieved_docs"] = retrieved_docs
    state["is_relevant"] = confidence >= rag_system.confidence_threshold and len(retrieved_docs) > 0
    
    print(f"📊 RAG Result: {len(retrieved_docs)} docs, confidence: {confidence:.3f}, relevant: {state['is_relevant']}")
    
    return state

def web_search_node(state: AgentState) -> AgentState:
    """Web search when vectorstore fails"""
    query = state["query"]
    print("🔄 Vectorstore failed, trying web search...")
    
    web_content = web_search.search_financial_sources(query)
    
    if web_content:
        state["retrieved_docs"] = [web_content]
        state["is_relevant"] = True
        print("✅ Web search found content")
    else:
        state["retrieved_docs"] = []
        state["is_relevant"] = False
        print("❌ Web search failed, will use LLM only")
    
    return state

def web_search_node(state: AgentState) -> AgentState:
    search_result = web_search.search_financial_sources(state["query"])
    state["retrieved_docs"] = [search_result]
    return state

def generate_node(state: AgentState) -> AgentState:
    """Generate response - with context or without"""
    query = state["query"]
    language = state["language"]
    docs = state.get("retrieved_docs", [])
    is_relevant = state.get("is_relevant", False)
    
    if not is_strictly_finance_related(query):
        state["generated_answer"] = get_strict_decline_message(language)
        return state
    
    system_message = f"""You are a financial assistant expert. Answer ONLY finance questions in {language} with EXCELLENT FORMATTING.

FORMATTING RULES:
- Use **bold** for important terms and headings
- Use numbered lists for key points (1., 2., 3., etc.)
- Use bullet points (•) for sub-items  
- Put formulas on separate lines with **bold formatting**
- Clearly define all variables used in formulas
- Use clear section headers
- Add proper spacing between sections

Specialties: stocks, bonds, options, Black-Scholes, derivatives, banking, investments, statistical analysis.
"""
    
    if docs and is_relevant:
        # Use retrieved context (from vectorstore OR web search)
        context = "\n\n".join(docs)
        prompt = f"""Context from trusted sources: {context}

Financial Question: {query}

Based on the provided context, give a comprehensive financial answer with proper formatting:"""
        print("🤖 Generating WITH retrieved context")
    else:
        # Generate using LLM knowledge only (no external context)
        prompt = f"""Financial Question: {query}

Using your financial knowledge, provide a comprehensive answer with proper formatting:"""
        print("🧠 Generating WITHOUT context (LLM knowledge only)")
    
    answer = llm.generate(prompt, system_message)
    state["generated_answer"] = answer
    return state

def final_answer_node(state: AgentState) -> AgentState:
    """Final step - set the final answer"""
    state["final_answer"] = state["generated_answer"]
    return state

def create_agent():
    """Create clean adaptive RAG workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("reject_non_finance", reject_non_finance_node)
    workflow.add_node("vectorstore", vectorstore_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("final_answer", final_answer_node)
    
    workflow.set_entry_point("query_analysis")
    
    # Route from query analysis
    workflow.add_conditional_edges(
        "query_analysis",
        lambda state: state["query_type"],
        {
            "REJECTED_NON_FINANCE": "reject_non_finance",
            "current_events": "vectorstore",    # Try vectorstore first
            "finance_query": "vectorstore"      # Try vectorstore first
        }
    )
    
    # CLEAN ADAPTIVE ROUTING: vectorstore → web_search OR generate
    workflow.add_conditional_edges(
        "vectorstore",
        lambda state: "generate" if state.get("is_relevant", False) else "web_search",
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    # Simple edges
    workflow.add_edge("web_search", "generate")      # Web search always goes to generate
    workflow.add_edge("reject_non_finance", END)
    workflow.add_edge("generate", "final_answer") 
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()

# Initialize components
print("Starting Financial AI Assistant...")
rag_system = PersistentFinancialRAG()
web_search = FinancialWebSearch()
llm = LangChainGroqLLM()
audio_processor = AudioProcessor()
agent = create_agent()
print("All systems ready!")

# ======================= FLASK ROUTES =======================
@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message', '').strip()
    language = data.get('language', 'english')
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({"role": "user", "content": query})
        
        initial_state = {
            "messages": [],
            "query": query,
            "language": language,
            "query_type": "",
            "retrieved_docs": [],
            "generated_answer": "",
            "is_relevant": True,
            "has_hallucination": False,
            "final_answer": ""
        }
        
        result = agent.invoke(initial_state)
        answer = result["final_answer"]
        
        session['chat_history'].append({"role": "assistant", "content": answer})
        session.modified = True
        
        return jsonify({
            'message': answer,
            'success': True
        })
        
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({
            'message': get_strict_decline_message(language),
            'success': True
        })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'english')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        audio_filename = audio_processor.text_to_speech(text, language)
        if audio_filename:
            return jsonify({
                'audio_url': f'/static/audio/{audio_filename}',
                'success': True
            })
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({'error': 'TTS processing failed'}), 500

@app.route('/history')
def get_history():
    return jsonify(session.get('chat_history', []))

@app.route('/clear')
def clear_history():
    session['chat_history'] = []
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
