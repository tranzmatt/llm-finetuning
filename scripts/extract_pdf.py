#!/usr/bin/env python3
"""
PDF Extraction Script
Extracts text/content from PDFs in data/raw/ and saves to data/extracted/
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

def extract_with_pypdf2(pdf_path: Path) -> List[Dict]:
    """Extract text using PyPDF2."""
    pages = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            pages.append({
                'page': page_num,
                'text': text.strip(),
                'method': 'pypdf2'
            })
    
    return pages

def extract_with_pdfplumber(pdf_path: Path) -> List[Dict]:
    """Extract text using pdfplumber (better for tables/structure)."""
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            pages.append({
                'page': page_num,
                'text': text.strip() if text else "",
                'method': 'pdfplumber'
            })
    
    return pages

def extract_pdf(pdf_path: Path, output_dir: Path, method='auto'):
    """Extract content from PDF."""
    
    print(f"\n📄 Processing: {pdf_path.name}")
    
    # Choose extraction method
    if method == 'auto':
        if HAS_PDFPLUMBER:
            method = 'pdfplumber'
        elif HAS_PYPDF2:
            method = 'pypdf2'
        else:
            print("❌ No PDF library available!")
            return None
    
    # Extract
    try:
        if method == 'pdfplumber':
            pages = extract_with_pdfplumber(pdf_path)
        elif method == 'pypdf2':
            pages = extract_with_pypdf2(pdf_path)
        else:
            print(f"❌ Unknown method: {method}")
            return None
        
        # Save extracted content
        output_file = output_dir / f"{pdf_path.stem}_extracted.json"
        
        data = {
            'source_file': pdf_path.name,
            'extraction_method': method,
            'total_pages': len(pages),
            'pages': pages
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Extracted {len(pages)} pages")
        print(f"  ✓ Saved to: {output_file.name}")
        
        return output_file
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract content from PDFs")
    parser.add_argument('--input', type=str, help='Specific PDF file')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'pypdf2', 'pdfplumber'],
                        help='Extraction method')
    
    args = parser.parse_args()
    
    raw_dir = PROJECT_ROOT / "data" / "raw"
    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    extracted_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("PDF EXTRACTION")
    print("="*80)
    
    # Find PDFs
    if args.input:
        pdf_files = [Path(args.input)]
    else:
        pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("\n❌ No PDF files found!")
        print(f"   Place PDFs in: {raw_dir}/")
        sys.exit(1)
    
    print(f"\nFound {len(pdf_files)} PDF(s)")
    
    # Extract each PDF
    results = []
    for pdf_path in pdf_files:
        result = extract_pdf(pdf_path, extracted_dir, args.method)
        results.append((pdf_path.name, result is not None))
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    
    for filename, success in results:
        status = "✓" if success else "❌"
        print(f"{status} {filename}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\n{successful}/{len(results)} files extracted successfully")
    print(f"\nExtracted files in: {extracted_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
