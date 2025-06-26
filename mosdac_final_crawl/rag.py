import json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
import requests
from typing import List, Dict

class EnhancedMOSDACRAG:
    def __init__(self, data_file="rag_prepared_data/mosdac_html_text_data.json"):
        self.data_file = data_file
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load and process data
        self.load_data()
        self.create_enhanced_chunks()
        self.build_index()
        
        # Knowledge base for common questions
        self.knowledge_base = self.build_knowledge_base()
    
    def load_data(self):
        """Load the prepared data"""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)
            print(f"✅ Loaded {len(self.raw_data)} data entries")
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_file}")
            self.raw_data = []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.raw_data = []
    
    def clean_content(self, text):
        """Clean and improve content quality"""
        if not text:
            return ""
        
        # Remove CSS and HTML artifacts
        text = re.sub(r'\.[\w-]+\s*{[^}]*}', '', text)  # Remove CSS rules
        text = re.sub(r'-webkit-[^;]+;', '', text)  # Remove webkit properties
        text = re.sub(r'transform[^;]+;', '', text)  # Remove transform properties
        text = re.sub(r'position:\s*absolute[^;]*;', '', text)  # Remove positioning
        text = re.sub(r'box-shadow[^;]+;', '', text)  # Remove box-shadow
        text = re.sub(r'border-radius[^;]+;', '', text)  # Remove border-radius
        
        # Remove excessive whitespace and clean up
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        text = text.strip()
        
        # Filter out very short or meaningless content
        if len(text) < 50 or 'sticky' in text.lower() or 'webkit' in text.lower():
            return ""
        
        return text
    
    def create_enhanced_chunks(self):
        """Create high-quality chunks from the loaded data"""
        chunk_size = 500  # Smaller chunks for better retrieval
        
        for entry in self.raw_data:
            text = entry.get("text", "")
            cleaned_text = self.clean_content(text)
            
            if len(cleaned_text.strip()) < 100:  # Skip poor quality content
                continue
            
            # Create meaningful chunks
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk.strip():
                        self.chunks.append({
                            "text": current_chunk.strip(),
                            "source": entry.get("file", "unknown"),
                            "title": entry.get("title", ""),
                            "source_type": entry.get("source_type", "unknown")
                        })
                    current_chunk = sentence
            
            # Add final chunk
            if current_chunk.strip():
                self.chunks.append({
                    "text": current_chunk.strip(),
                    "source": entry.get("file", "unknown"),
                    "title": entry.get("title", ""),
                    "source_type": entry.get("source_type", "unknown")
                })
        
        print(f"📝 Created {len(self.chunks)} high-quality chunks")
    
    def build_knowledge_base(self):
        """Build structured knowledge base for common MOSDAC topics"""
        return {
            "insat-3dr": {
                "description": "INSAT-3DR is an advanced meteorological satellite launched by ISRO in 2016. It provides improved weather forecasting and disaster warning capabilities.",
                "capabilities": [
                    "Advanced atmospheric sounding",
                    "High-resolution imaging",
                    "Data relay services",
                    "Search and rescue operations"
                ],
                "instruments": ["Imager", "Sounder", "Data Relay Transponder"],
                "applications": ["Weather forecasting", "Cyclone tracking", "Monsoon monitoring", "Climate studies"]
            },
            "mosdac": {
                "description": "MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre) is ISRO's facility for satellite data archival and dissemination.",
                "services": [
                    "Satellite data archive and retrieval",
                    "Real-time data access",
                    "Data processing and analysis tools",
                    "User training and support"
                ],
                "data_types": ["Meteorological data", "Oceanographic data", "Land surface data", "Atmospheric data"]
            },
            "oceansat": {
                "description": "OCEANSAT series satellites are designed for ocean color monitoring and oceanographic studies.",
                "applications": [
                    "Ocean color monitoring",
                    "Sea surface temperature measurement",
                    "Coastal zone management",
                    "Marine ecosystem studies"
                ]
            },
            "weather_forecasting": {
                "description": "Satellite-based weather forecasting uses data from meteorological satellites to predict weather patterns.",
                "process": [
                    "Data collection from satellites",
                    "Atmospheric parameter analysis",
                    "Numerical weather prediction models",
                    "Forecast generation and dissemination"
                ]
            }
        }
    
    def build_index(self):
        """Build FAISS index"""
        if not self.chunks:
            print("❌ No chunks available for indexing")
            return
        
        print("🔄 Building embeddings and FAISS index...")
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def search(self, query, top_k=5):
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)
        
        results = []
        for i in I[0]:
            if i < len(self.chunks):
                chunk = self.chunks[i]
                results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "source_type": chunk["source_type"],
                    "score": float(D[0][len(results)])
                })
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate a comprehensive answer using retrieved context"""
        
        # Check for direct knowledge base matches
        query_lower = query.lower()
        kb_answer = self.get_knowledge_base_answer(query_lower)
        
        if not retrieved_chunks:
            if kb_answer:
                return kb_answer
            return "I couldn't find specific information about that in the MOSDAC database. Please try rephrasing your question or ask about ISRO satellites, weather forecasting, or ocean monitoring."
        
        # Extract relevant information from chunks
        relevant_info = []
        for chunk in retrieved_chunks[:3]:  # Use top 3 chunks
            if chunk['score'] < 1.5:  # Only use high-quality matches
                relevant_info.append(chunk['text'])
        
        if not relevant_info:
            if kb_answer:
                return kb_answer
            return "I found some related information but it may not directly answer your question. Please try asking more specifically about MOSDAC services, satellite missions, or data products."
        
        # Generate comprehensive answer
        context = " ".join(relevant_info)
        
        # Template-based answer generation
        if any(keyword in query_lower for keyword in ['what is', 'define', 'explain']):
            answer = self.generate_definition_answer(query, context, kb_answer)
        elif any(keyword in query_lower for keyword in ['how', 'process', 'work']):
            answer = self.generate_process_answer(query, context, kb_answer)
        elif any(keyword in query_lower for keyword in ['capabilities', 'features', 'instruments']):
            answer = self.generate_capability_answer(query, context, kb_answer)
        else:
            answer = self.generate_general_answer(query, context, kb_answer)
        
        # Add sources
        sources = list(set([chunk['source'] for chunk in retrieved_chunks[:3] if chunk['score'] < 1.5]))
        if sources:
            answer += f"\n\n**Sources:** {', '.join(sources[:2])}"
        
        return answer
    
    def get_knowledge_base_answer(self, query_lower: str) -> str:
        """Get answer from structured knowledge base"""
        if 'insat-3dr' in query_lower or 'insat 3dr' in query_lower:
            kb = self.knowledge_base['insat-3dr']
            return f"**INSAT-3DR:** {kb['description']}\n\n**Key Capabilities:** {', '.join(kb['capabilities'])}\n\n**Applications:** {', '.join(kb['applications'])}"
        
        elif 'mosdac' in query_lower and ('what is' in query_lower or 'about' in query_lower):
            kb = self.knowledge_base['mosdac']
            return f"**MOSDAC:** {kb['description']}\n\n**Services:** {', '.join(kb['services'])}\n\n**Data Types:** {', '.join(kb['data_types'])}"
        
        elif 'oceansat' in query_lower:
            kb = self.knowledge_base['oceansat']
            return f"**OCEANSAT:** {kb['description']}\n\n**Applications:** {', '.join(kb['applications'])}"
        
        elif 'weather forecasting' in query_lower or 'weather prediction' in query_lower:
            kb = self.knowledge_base['weather_forecasting']
            return f"**Weather Forecasting:** {kb['description']}\n\n**Process:** {', '.join(kb['process'])}"
        
        return ""
    
    def generate_definition_answer(self, query: str, context: str, kb_answer: str) -> str:
        """Generate definition-style answers"""
        if kb_answer:
            return kb_answer
        
        # Extract key information from context
        sentences = context.split('.')[:3]  # First 3 sentences
        definition = '. '.join(sentences).strip()
        
        if definition:
            return f"Based on MOSDAC data: {definition}."
        return "I found some information but couldn't extract a clear definition. Please try asking more specifically."
    
    def generate_process_answer(self, query: str, context: str, kb_answer: str) -> str:
        """Generate process-oriented answers"""
        if kb_answer:
            return kb_answer
        
        # Look for process-related keywords
        process_info = []
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['process', 'step', 'method', 'procedure', 'how']):
                process_info.append(sentence.strip())
        
        if process_info:
            return f"**Process:** {'. '.join(process_info[:2])}."
        
        return f"Based on available information: {context[:300]}..."
    
    def generate_capability_answer(self, query: str, context: str, kb_answer: str) -> str:
        """Generate capability-focused answers"""
        if kb_answer:
            return kb_answer
        
        # Extract capability information
        capabilities = []
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['capability', 'feature', 'instrument', 'can', 'able']):
                capabilities.append(sentence.strip())
        
        if capabilities:
            return f"**Capabilities:** {'. '.join(capabilities[:3])}."
        
        return f"Based on available data: {context[:300]}..."
    
    def generate_general_answer(self, query: str, context: str, kb_answer: str) -> str:
        """Generate general answers"""
        if kb_answer:
            additional_context = context[:200] if context else ""
            if additional_context and additional_context not in kb_answer:
                return f"{kb_answer}\n\n**Additional Information:** {additional_context}..."
            return kb_answer
        
        # Summarize context
        sentences = context.split('.')[:4]  # First 4 sentences
        summary = '. '.join(sentences).strip()
        
        if summary:
            return f"Based on MOSDAC data: {summary}."
        
        return "I found some related information but couldn't generate a specific answer. Please try asking about specific satellites, services, or data products."
    
    def query_rag(self, user_input, top_k=5):
        """Main query function with enhanced answer generation"""
        if not user_input.strip():
            return "Please enter a question about MOSDAC, ISRO satellites, weather forecasting, or ocean monitoring."
        
        # Search for relevant chunks
        results = self.search(user_input, top_k)
        
        # Generate comprehensive answer
        answer = self.generate_answer(user_input, results)
        
        return answer

# Initialize the enhanced system
print("🚀 Initializing Enhanced MOSDAC RAG System...")

# Check if data file exists
data_file = "rag_prepared_data/mosdac_html_text_data.json"
if not Path(data_file).exists():
    print("❌ Prepared data file not found!")
    exit(1)

rag_system = EnhancedMOSDACRAG(data_file)

# Create enhanced Gradio interface
interface = gr.Interface(
    fn=rag_system.query_rag,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Ask about ISRO satellites, MOSDAC services, weather forecasting, ocean monitoring...",
        label="Your Question"
    ),
    outputs=gr.Markdown(label="Answer"),
    title="🛰️ Enhanced MOSDAC RAG Assistant",
    description=f"""
    **Status:** ✅ Loaded {len(rag_system.chunks)} high-quality chunks from MOSDAC data
    
    **I can help you with:**
    • ISRO satellites (INSAT-3DR, OCEANSAT series, etc.)
    • Weather forecasting and monsoon prediction
    • Ocean monitoring and satellite data
    • MOSDAC services and data access
    • Satellite instruments and capabilities
    """,
    examples=[
        "What is INSAT-3DR and what are its capabilities?",
        "How does MOSDAC help in weather forecasting?",
        "What satellite data is available for ocean monitoring?",
        "Tell me about monsoon prediction using satellite data",
        "What services does MOSDAC provide?",
        "How do meteorological satellites work?"
    ],
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """
)

if __name__ == "__main__":
    interface.launch(share=False, inbrowser=True)
