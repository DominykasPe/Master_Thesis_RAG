#!/usr/bin/env python3
"""
NASA Data Downloader & Processor
================================
A unified script to download NASA Lessons Learned and Technical Documents,
with optional chunking for RAG (Retrieval Augmented Generation) systems.

Usage:
    python download_nasa_data.py

Author: Dominykas Petniunas
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nasa_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Default output directories
DEFAULT_LESSONS_OUTPUT = SCRIPT_DIR / "Document Collection" / "NASA_Lessons_Learned"
DEFAULT_PDFS_OUTPUT = SCRIPT_DIR / "Document Collection" / "NTRS_PDFS"
DEFAULT_CHUNKS_OUTPUT = SCRIPT_DIR.parent / "Chunks" / "section_chunks"


def print_banner():
    """Print a welcome banner."""
    print("\n" + "=" * 60)
    print("       NASA Data Downloader & Processor")
    print("       Supporting RAG-based Knowledge Retrieval")
    print("=" * 60 + "\n")


def print_menu():
    """Print the main menu options."""
    print("What would you like to download?\n")
    print("  [1] NASA Lessons Learned (CSV)")
    print("      - Scrapes lessons from NASA LLIS database (2000-present)")
    print("      - Output: CSV file with lessons, recommendations, etc.\n")
    
    print("  [2] NASA Conference Papers (PDFs)")
    print("      - Downloads Conference Papers from NASA NTRS")
    print("      - Output: PDF files in specified directory\n")
    
    print("  [3] Both (Lessons Learned + Conference Papers)\n")
    
    print("  [4] Process existing PDFs (Chunking only)")
    print("      - Converts PDFs to searchable chunks for RAG\n")
    
    print("  [5] Exit\n")


def get_user_choice(prompt: str, valid_choices: list) -> str:
    """Get validated user input."""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def get_directory_input(prompt: str, default: Path) -> Path:
    """Get directory path from user with default option."""
    print(f"\n{prompt}")
    print(f"  Default: {default}")
    user_input = input("  Press Enter for default, or enter custom path: ").strip()
    
    if not user_input:
        return default
    return Path(user_input)


def get_numeric_input(prompt: str, default: int, min_val: int = 1, max_val: int = 10000) -> int:
    """Get numeric input from user with validation."""
    while True:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")


def download_lessons_learned(output_dir: Path, start_year: int = 2000, end_year: int = None, max_workers: int = 4):
    """Download NASA Lessons Learned."""
    print("\n" + "-" * 40)
    print("Starting NASA Lessons Learned Download...")
    print(f"Date range: {start_year} to {end_year if end_year else 'present'}")
    print("-" * 40)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and run the lessons learned scraper
    try:
        # Add the NASA_Web_Scraping directory to path
        web_scraping_dir = SCRIPT_DIR / "Document Collection" / "NASA_Web_Scraping"
        sys.path.insert(0, str(web_scraping_dir))
        
        from lessons_learned import NASALessonsLearned
        
        # Create scraper with date range
        scraper = NASALessonsLearned(
            max_workers=max_workers,
            start_year=start_year,
            end_year=end_year
        )
        
        # Update the scraper to use our output directory
        scraper.csv_path = str(output_dir / f"nasa_lessons_learned_{start_year}_{end_year if end_year else 'present'}.csv")
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(scraper.csv_path):
            import pandas as pd
            pd.DataFrame(columns=[
                'url', 'subject', 'abstract', 'driving_event', 
                'lessons_learned', 'recommendations', 'evidence',
                'program_relation', 'program_phase', 
                'mission_directorate', 'topics', 'date_range'
            ]).to_csv(scraper.csv_path, index=False)
        
        # Run the collection
        df = scraper.collect_all_lessons()
        
        print(f"\n✓ Lessons Learned saved to: {scraper.csv_path}")
        print(f"✓ Total lessons collected: {len(df)}")
        return True
        
    except ImportError as e:
        logger.error(f"Could not import lessons_learned module: {e}")
        print("\n✗ Error: Could not find the lessons_learned.py script.")
        print("  Make sure the file exists in: Document Collection/NASA_Web_Scraping/")
        return False
    except Exception as e:
        logger.error(f"Error downloading lessons learned: {e}")
        print(f"\n✗ Error during download: {e}")
        return False


def download_technical_documents(output_dir: Path, max_docs: int = 100, doc_type: str = "conference"):
    """Download NASA Conference Papers (PDFs) from NTRS."""
    print("\n" + "-" * 40)
    print("Starting NASA Conference Papers Download...")
    print("-" * 40)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Add the NASA_Web_Scraping directory to path
        web_scraping_dir = SCRIPT_DIR / "Document Collection" / "NASA_Web_Scraping"
        sys.path.insert(0, str(web_scraping_dir))
        
        from technical_documents import PDFDownloader
        
        # Conference Papers URL from NASA Technical Reports Server
        nasa_url = "https://ntrs.nasa.gov/search?stiTypeDetails=Conference%20Paper"
        
        # Create downloader and run
        downloader = PDFDownloader(
            base_url=nasa_url,
            output_dir=str(output_dir),
            max_docs=max_docs
        )
        
        num_downloaded = downloader.download_pdfs()
        
        print(f"\n✓ Conference Papers saved to: {output_dir}")
        print(f"✓ Total PDFs downloaded: {num_downloaded}")
        return True
        
    except ImportError as e:
        logger.error(f"Could not import technical_documents module: {e}")
        print("\n✗ Error: Could not find the technical_documents.py script.")
        print("  Make sure the file exists in: Document Collection/NASA_Web_Scraping/")
        return False
    except Exception as e:
        logger.error(f"Error downloading technical documents: {e}")
        print(f"\n✗ Error during download: {e}")
        return False


def process_pdfs_to_chunks(input_dir: Path, output_file: Path, timeout: int = 60):
    """Process PDFs into chunks for RAG."""
    print("\n" + "-" * 40)
    print("Starting PDF Chunking Process...")
    print("-" * 40)
    
    if not input_dir.exists():
        print(f"\n✗ Error: Input directory does not exist: {input_dir}")
        return False
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Add the Section Chunking directory to path
        chunking_dir = SCRIPT_DIR / "Section Chunking"
        sys.path.insert(0, str(chunking_dir))
        
        from section_based_chunking import SectionChunker
        
        # Create chunker and process
        chunker = SectionChunker(
            input_dir=str(input_dir),
            output_file=str(output_file),
            timeout_seconds=timeout
        )
        
        chunker.process_pdfs()
        
        print(f"\n✓ Chunks saved to: {output_file}")
        return True
        
    except ImportError as e:
        logger.error(f"Could not import section_based_chunking module: {e}")
        print("\n✗ Error: Could not find the section_based_chunking.py script.")
        print("  Make sure the file exists in: Section Chunking/")
        return False
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        print(f"\n✗ Error during chunking: {e}")
        return False


def run_lessons_learned_flow():
    """Interactive flow for downloading lessons learned."""
    from datetime import datetime
    
    print("\n" + "=" * 40)
    print("NASA Lessons Learned Configuration")
    print("=" * 40)
    
    output_dir = get_directory_input(
        "Where should the CSV be saved?",
        DEFAULT_LESSONS_OUTPUT
    )
    
    print("\n📅 Date Range Configuration:")
    print("   NASA LLIS contains lessons from 2000 to present.")
    
    start_year = get_numeric_input(
        "Start year",
        default=2000, min_val=1990, max_val=datetime.now().year
    )
    
    end_year = get_numeric_input(
        "End year",
        default=datetime.now().year, min_val=start_year, max_val=datetime.now().year
    )
    
    max_workers = get_numeric_input(
        "Number of parallel workers",
        default=4, min_val=1, max_val=8
    )
    
    # Estimate time
    years_to_process = end_year - start_year + 1
    estimated_time = years_to_process * 2  # Rough estimate: ~2 minutes per year
    
    print(f"\n📊 Will download lessons from {start_year} to {end_year}")
    print(f"   Estimated years to process: {years_to_process}")
    print(f"   Estimated time: {estimated_time}-{estimated_time*2} minutes")
    print("\n⚠️  Note: This will open multiple browser windows (headless).")
    print("    The process may take considerable time for large date ranges.\n")
    
    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        return download_lessons_learned(output_dir, start_year=start_year, end_year=end_year, max_workers=max_workers)
    else:
        print("Cancelled.")
        return False


def run_technical_documents_flow():
    """Interactive flow for downloading technical documents. Returns the output directory on success."""
    print("\n" + "=" * 40)
    print("NASA Technical Documents Configuration")
    print("=" * 40)
    
    print("\n📄 Document Type: Conference Papers")
    print("   Source: NASA Technical Reports Server (NTRS)")
    print("   URL: https://ntrs.nasa.gov/search?stiTypeDetails=Conference%20Paper")
    
    output_dir = get_directory_input(
        "Where should PDFs be saved?",
        DEFAULT_PDFS_OUTPUT / "conference"
    )
    
    max_docs = get_numeric_input(
        "Maximum number of documents to download",
        default=100, min_val=1, max_val=10000
    )
    
    # Estimate download time (~5 seconds per PDF on average)
    estimated_time = (max_docs * 5) // 60
    
    print(f"\n📊 Will download up to {max_docs} Conference Papers")
    print(f"   Estimated time: {estimated_time}-{estimated_time*2} minutes")
    print("\n⚠️  Note: This will open a Chrome browser window for scraping.")
    print("    Downloads are saved incrementally, so you can resume if interrupted.\n")
    
    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        success = download_technical_documents(output_dir, max_docs=max_docs, doc_type="conference")
        if success:
            return output_dir  # Return the directory where PDFs were saved
        return None
    else:
        print("Cancelled.")
        return None


def run_chunking_flow(suggested_input_dir: Path = None):
    """Interactive flow for processing PDFs to chunks."""
    print("\n" + "=" * 40)
    print("PDF Chunking Configuration")
    print("=" * 40)
    
    # Use suggested directory if provided, otherwise use default
    default_input = suggested_input_dir if suggested_input_dir else DEFAULT_PDFS_OUTPUT
    
    # Show available PDF directories
    print("\nScanning for PDF directories...")
    pdf_dirs = []
    doc_collection = SCRIPT_DIR / "Document Collection"
    if doc_collection.exists():
        for root, dirs, files in os.walk(doc_collection):
            pdf_count = len([f for f in files if f.endswith('.pdf')])
            if pdf_count > 0:
                pdf_dirs.append((Path(root), pdf_count))
    
    if pdf_dirs:
        print("\nFound PDF directories:")
        for i, (dir_path, count) in enumerate(pdf_dirs, 1):
            print(f"  [{i}] {dir_path.relative_to(SCRIPT_DIR)} ({count} PDFs)")
        print(f"  [0] Enter custom path")
        
        dir_choice = input(f"\nSelect directory [1-{len(pdf_dirs)}, or 0 for custom]: ").strip()
        
        if dir_choice == '0':
            input_dir = get_directory_input(
                "Enter the path to PDF files:",
                default_input
            )
        elif dir_choice.isdigit() and 1 <= int(dir_choice) <= len(pdf_dirs):
            input_dir = pdf_dirs[int(dir_choice) - 1][0]
        else:
            input_dir = default_input
    else:
        input_dir = get_directory_input(
            "Where are the PDF files located?",
            default_input
        )
    
    # Verify PDFs exist
    if input_dir.exists():
        pdf_count = len([f for f in os.listdir(input_dir) if f.endswith('.pdf')])
        print(f"\n📁 Found {pdf_count} PDF files in: {input_dir}")
        if pdf_count == 0:
            print("⚠️  No PDF files found! Please check the directory.")
            return False
    else:
        print(f"\n⚠️  Directory does not exist: {input_dir}")
        return False
    
    output_file = get_directory_input(
        "Where should the chunks JSON be saved?",
        DEFAULT_CHUNKS_OUTPUT / "section_chunks.json"
    )
    
    timeout = get_numeric_input(
        "Timeout per PDF (seconds)",
        default=60, min_val=10, max_val=300
    )
    
    confirm = input("\nStart chunking? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        return process_pdfs_to_chunks(input_dir, output_file, timeout=timeout)
    else:
        print("Cancelled.")
        return False


def main():
    """Main entry point."""
    print_banner()
    
    while True:
        print_menu()
        choice = get_user_choice("Enter your choice [1-5]: ", ['1', '2', '3', '4', '5'])
        
        if choice == '1':
            run_lessons_learned_flow()
        
        elif choice == '2':
            pdf_dir = run_technical_documents_flow()
            if pdf_dir:
                # Ask about chunking after download
                chunk_confirm = input("\nWould you like to process these Conference Papers into chunks now? [Y/n]: ").strip().lower()
                if chunk_confirm in ['', 'y', 'yes']:
                    run_chunking_flow(suggested_input_dir=pdf_dir)
        
        elif choice == '3':
            print("\n" + "=" * 40)
            print("Downloading Both Sources")
            print("  1. NASA Lessons Learned (CSV)")
            print("  2. NASA Conference Papers (PDFs)")
            print("=" * 40)
            
            # Run lessons learned first
            lessons_success = run_lessons_learned_flow()
            
            # Then conference papers
            pdf_dir = run_technical_documents_flow()
            
            if lessons_success and pdf_dir:
                print("\n✓ Both downloads completed successfully!")
                
                # Ask about chunking
                chunk_confirm = input("\nWould you like to process the Conference Papers into chunks now? [Y/n]: ").strip().lower()
                if chunk_confirm in ['', 'y', 'yes']:
                    run_chunking_flow(suggested_input_dir=pdf_dir)
        
        elif choice == '4':
            run_chunking_flow()
        
        elif choice == '5':
            print("\nGoodbye! 👋\n")
            sys.exit(0)
        
        # Ask if user wants to continue
        print("\n" + "-" * 40)
        continue_choice = input("Return to main menu? [Y/n]: ").strip().lower()
        if continue_choice not in ['', 'y', 'yes']:
            print("\nGoodbye! 👋\n")
            break


if __name__ == "__main__":
    main()

