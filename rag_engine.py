# rag_engine.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List, Any
import threading

load_dotenv(".env")

# Global variables for caching
_rag_chain_cache = None
_rag_lock = threading.Lock()
VECTOR_STORE_PATH = "data/faiss_index"  # Persistent vector store location

# Custom LangChain wrapper for Google Generative AI
class GoogleGenerativeAILLM(BaseLLM):
    """LangChain-compatible wrapper for Google Generative AI"""
    model_name: str = "gemini-flash-latest"
    temperature: float = 0.2
    api_key: Optional[str] = None
    
    def __init__(self, model_name: str = "gemini-flash-latest", temperature: float = 0.2, api_key: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=self.api_key)
        # Use object.__setattr__ to bypass Pydantic v2 validation
        object.__setattr__(self, '_model', genai.GenerativeModel(self.model_name))
    
    @property
    def _llm_type(self) -> str:
        return "google_generative_ai"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        generations = []
        for prompt in prompts:
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
            )
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config
            )
            generations.append([Generation(text=response.text)])
        
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with a single prompt."""
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
        )
        response = self._model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

def initialize_rag(use_cache=True, force_rebuild=False):
    """
    Initialize RAG system with persistent vector store.
    
    Args:
        use_cache: If True, load from cache if available
        force_rebuild: If True, rebuild vectors even if cache exists
    """
    print("🔄 [RAG] Initializing RAG system...")
    
    # Try to load cached vector store first
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = None
    
    if use_cache and not force_rebuild and os.path.exists(VECTOR_STORE_PATH):
        try:
            print("📦 [RAG] Loading persistent vector store from cache...")
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"✅ [RAG] Loaded cached vector store from: {os.path.abspath(VECTOR_STORE_PATH)}")
        except Exception as e:
            print(f"⚠️  [RAG] Could not load cache: {e}. Creating new vector store...")
            vectorstore = None
    
    # Create new vector store if cache doesn't exist or force_rebuild is True
    if vectorstore is None:
        # Load your document
        print("📄 [RAG] Loading PDF document...")
        loader = PyPDFLoader("data/Electric_Car_Overview.pdf")
        documents = loader.load()
        print(f"✅ [RAG] Loaded {len(documents)} pages from PDF")

        # Split into smaller chunks for embedding
        print("✂️  [RAG] Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        print(f"✅ [RAG] Created {len(chunks)} text chunks")

        # Create embeddings & store in FAISS
        print("🔢 [RAG] Creating embeddings and vector store (this may take a moment)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save to cache for future use
        if use_cache:
            try:
                os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
                vectorstore.save_local(VECTOR_STORE_PATH)
                print("💾 [RAG] Vector store cached for faster future loads")
            except Exception as e:
                print(f"⚠️  [RAG] Could not save cache: {e}")
        
        print("✅ [RAG] Vector store created successfully")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Connect to Gemini for retrieval-based answering
    print("🤖 [RAG] Initializing LLM...")
    # Use gemini-flash-latest (or gemini-2.5-flash for stable version)
    model_name = "gemini-flash-latest"
    llm = GoogleGenerativeAILLM(
        model_name=model_name,
        temperature=0.2,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    print(f"✅ [RAG] LLM initialized with model: {model_name}")

    # Create prompt template - STRICT: Only answer from knowledge base
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant that ONLY answers questions based on the provided knowledge base context.

CRITICAL RULES:
1. You MUST ONLY use information from the provided context below to answer the question.
2. If the answer is not in the context, you MUST say: "I don't have that information in my knowledge base."
3. DO NOT use any external knowledge or make assumptions beyond what's in the context.
4. DO NOT add information that is not explicitly stated in the context.
5. If the context is empty or irrelevant, say: "I don't have that information in my knowledge base."

Context from knowledge base:
{context}

Question: {question}

Answer (ONLY from the context above): """
    )

    # Create RAG pipeline using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Create a wrapper to maintain compatibility with .run() method
    class RAGChainWrapper:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever
        
        def run(self, question):
            print(f"\n🔍 [RAG] Processing question: {question}")
            
            # Retrieve relevant documents using invoke() (newer LangChain API)
            try:
                docs = self.retriever.invoke(question)
                print(f"📚 [RAG] Retrieved {len(docs)} relevant document chunks")
                
                # Show first chunk preview (for debugging)
                if docs:
                    preview = docs[0].page_content[:200] + "..." if len(docs[0].page_content) > 200 else docs[0].page_content
                    print(f"📖 [RAG] First chunk preview: {preview}")
            except Exception as e:
                print(f"⚠️  [RAG] Could not preview retrieved docs: {e}")
                docs = []
            
            # Get answer from chain
            answer = self.chain.invoke(question)
            print(f"💬 [RAG] Answer: {answer[:100]}..." if len(answer) > 100 else f"💬 [RAG] Answer: {answer}")
            
            return answer

    print("✅ [RAG] RAG system initialized successfully!\n")
    return RAGChainWrapper(rag_chain, retriever)

def get_rag_chain(force_rebuild=False):
    """
    Thread-safe lazy initialization of RAG chain with persistent vectors.
    This function ensures RAG is only initialized once per process.
    
    Args:
        force_rebuild: If True, rebuild vectors even if cache exists (default: False)
    
    Returns:
        RAG chain that uses persistent vector store
    """
    global _rag_chain_cache
    
    # Double-checked locking pattern for thread safety
    if _rag_chain_cache is None or force_rebuild:
        with _rag_lock:
            if _rag_chain_cache is None or force_rebuild:
                # Always use cache if available (unless force_rebuild)
                _rag_chain_cache = initialize_rag(use_cache=True, force_rebuild=force_rebuild)
    
    return _rag_chain_cache
