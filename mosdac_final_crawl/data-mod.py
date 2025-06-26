import json
import os
from pathlib import Path
import re

class FixedMOSDACDataPreparator:
    def __init__(self):
        # Get the correct path to your crawled data
        current_dir = Path.cwd()
        print(f"Current directory: {current_dir}")
        
        # Look for data in current directory and parent directories
        possible_data_dirs = [
            current_dir,  # Current directory
            current_dir.parent,  # Parent directory
            current_dir.parent / "mosdac_final_crawl",
            current_dir.parent / "mosdac_complete_collection",
            current_dir.parent / "mosdac_enhanced_collection"
        ]
        
        self.crawl_data_dir = None
        for data_dir in possible_data_dirs:
            if self.check_data_directory(data_dir):
                self.crawl_data_dir = data_dir
                break
        
        if not self.crawl_data_dir:
            print("❌ Could not find crawled data directory!")
            print("Looking for directories with 'pages', 'pdfs', or summary files...")
            self.find_data_directories()
        
        self.output_dir = Path("rag_prepared_data")
        self.output_dir.mkdir(exist_ok=True)
    
    def check_data_directory(self, data_dir):
        """Check if directory contains crawled data"""
        if not data_dir.exists():
            return False
        
        # Check for expected subdirectories or files
        has_pages = (data_dir / "pages").exists()
        has_pdfs = (data_dir / "pdfs").exists()
        has_summary = any([
            (data_dir / "latest_summary.json").exists(),
            (data_dir / "crawl_summary.json").exists(),
            (data_dir / "comprehensive_summary.json").exists()
        ])
        
        if has_pages or has_pdfs or has_summary:
            print(f"✅ Found data directory: {data_dir}")
            return True
        return False
    
    def find_data_directories(self):
        """Find all possible data directories"""
        current = Path.cwd()
        parent = current.parent
        
        print("\n🔍 Searching for data directories...")
        for directory in [current, parent]:
            for item in directory.iterdir():
                if item.is_dir():
                    if self.check_data_directory(item):
                        print(f"Found: {item}")
    
    def prepare_all_data(self):
        """Prepare all available data"""
        if not self.crawl_data_dir:
            # Create sample data if no crawled data found
            return self.create_sample_data()
        
        print(f"🔄 Preparing data from: {self.crawl_data_dir}")
        
        all_data = []
        
        # Try to load from summary files first
        summary_data = self.load_from_summary()
        if summary_data:
            all_data.extend(summary_data)
        
        # Load HTML pages
        html_data = self.prepare_html_data()
        all_data.extend(html_data)
        
        # Load PDFs
        pdf_data = self.prepare_pdf_data()
        all_data.extend(pdf_data)
        
        # Load images
        image_data = self.prepare_image_data()
        all_data.extend(image_data)
        
        if not all_data:
            print("⚠️ No data found, creating sample data...")
            all_data = self.create_sample_data()
        
        self.save_formatted_data(all_data)
        return all_data
    
    def load_from_summary(self):
        """Load data from summary files"""
        summary_files = [
            "latest_summary.json",
            "crawl_summary.json",
            "comprehensive_summary.json"
        ]
        
        for filename in summary_files:
            summary_path = self.crawl_data_dir / filename
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    data = []
                    # Extract page data from summary
                    if 'pages' in summary:
                        for page in summary['pages'][:50]:  # Limit to 50 pages
                            if 'title' in page and 'url' in page:
                                # Create text from page metadata
                                text = f"""
                                Title: {page.get('title', 'Unknown')}
                                URL: {page.get('url', '')}
                                Page Type: {page.get('page_type', 'webpage')}
                                Content: This page from MOSDAC contains information about {page.get('title', 'satellite and weather data')}.
                                """
                                
                                data.append({
                                    "text": self.clean_text(text),
                                    "file": page.get('url', 'unknown'),
                                    "title": page.get('title', ''),
                                    "source_type": "webpage"
                                })
                    
                    print(f"📊 Loaded {len(data)} entries from {filename}")
                    return data
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        return []
    
    def prepare_html_data(self):
        """Load HTML data if available"""
        html_data = []
        pages_dir = self.crawl_data_dir / "pages"
        
        if not pages_dir.exists():
            return html_data
        
        count = 0
        for page_type_dir in pages_dir.iterdir():
            if page_type_dir.is_dir() and count < 20:  # Limit to 20 pages
                for depth_dir in page_type_dir.iterdir():
                    if depth_dir.is_dir() and count < 20:
                        for page_dir in depth_dir.iterdir():
                            if page_dir.is_dir() and count < 20:
                                page_data = self.extract_page_data(page_dir)
                                if page_data:
                                    html_data.append(page_data)
                                    count += 1
        
        print(f"📄 Processed {len(html_data)} HTML pages")
        return html_data
    
    def extract_page_data(self, page_dir):
        """Extract data from page directory"""
        data_file = page_dir / "data.json"
        content_file = page_dir / "content.md"
        
        if not data_file.exists():
            return None
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                page_metadata = json.load(f)
            
            content = ""
            if content_file.exists():
                with open(content_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            cleaned_content = self.clean_text(content)
            
            if len(cleaned_content.strip()) < 100:
                return None
            
            return {
                "text": cleaned_content,
                "file": page_metadata.get('url', 'unknown'),
                "title": page_metadata.get('title', ''),
                "source_type": "webpage"
            }
        except Exception as e:
            return None
    
    def prepare_pdf_data(self):
        """Prepare PDF data"""
        pdf_data = []
        pdfs_dir = self.crawl_data_dir / "pdfs"
        
        if pdfs_dir.exists():
            pdf_files = list(pdfs_dir.glob("*.pdf"))
            for pdf_file in pdf_files[:5]:  # Limit to 5 PDFs
                # Create metadata entry for PDF
                pdf_data.append({
                    "text": f"PDF Document: {pdf_file.name}. This document contains technical information about satellite data, weather forecasting, or MOSDAC services.",
                    "file": pdf_file.name,
                    "title": pdf_file.stem,
                    "source_type": "pdf"
                })
        
        print(f"📚 Processed {len(pdf_data)} PDF documents")
        return pdf_data
    
    def prepare_image_data(self):
        """Prepare image data"""
        image_data = []
        images_dir = self.crawl_data_dir / "images"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            for img_file in image_files[:10]:  # Limit to 10 images
                image_data.append({
                    "text": f"Image: {img_file.name}. This satellite or weather image from MOSDAC shows meteorological or oceanographic data visualization.",
                    "file": img_file.name,
                    "title": f"Image: {img_file.stem}",
                    "source_type": "image"
                })
        
        print(f"🖼️ Processed {len(image_data)} images")
        return image_data
    
    def create_sample_data(self):
        """Create sample MOSDAC data if no crawled data available"""
        sample_data = [
            {
                "text": "INSAT-3DR is an advanced meteorological satellite launched by ISRO for weather forecasting and disaster warning. It carries improved imaging and sounding instruments for better atmospheric observations.",
                "file": "sample_insat3dr.html",
                "title": "INSAT-3DR Satellite",
                "source_type": "webpage"
            },
            {
                "text": "MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre) provides satellite data for weather forecasting, climate monitoring, and oceanographic studies. It serves data from various ISRO satellites.",
                "file": "sample_mosdac.html", 
                "title": "MOSDAC Services",
                "source_type": "webpage"
            },
            {
                "text": "OCEANSAT-3 is designed for ocean color monitoring and sea surface temperature measurements. It helps in studying marine ecosystems and coastal zone management.",
                "file": "sample_oceansat3.html",
                "title": "OCEANSAT-3 Mission",
                "source_type": "webpage"
            },
            {
                "text": "Weather forecasting using satellite data involves analysis of atmospheric parameters like temperature, humidity, and wind patterns from instruments aboard meteorological satellites.",
                "file": "sample_weather.html",
                "title": "Satellite Weather Forecasting",
                "source_type": "webpage"
            },
            {
                "text": "Monsoon prediction relies on satellite observations of sea surface temperature, atmospheric moisture, and wind patterns. INSAT series satellites provide crucial data for monsoon forecasting.",
                "file": "sample_monsoon.html",
                "title": "Monsoon Prediction",
                "source_type": "webpage"
            }
        ]
        
        print("📝 Created sample MOSDAC data")
        return sample_data
    
    def clean_text(self, text):
        """Clean text content"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_formatted_data(self, all_data):
        """Save formatted data"""
        output_file = self.output_dir / "mosdac_html_text_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(all_data)} chunks to: {output_file}")
        
        summary = {
            "total_chunks": len(all_data),
            "source_breakdown": {},
            "preparation_timestamp": "2025-06-26"
        }
        
        for item in all_data:
            source_type = item.get('source_type', 'unknown')
            summary['source_breakdown'][source_type] = summary['source_breakdown'].get(source_type, 0) + 1
        
        print(f"📊 Data summary: {summary}")

if __name__ == "__main__":
    preparator = FixedMOSDACDataPreparator()
    prepared_data = preparator.prepare_all_data()
