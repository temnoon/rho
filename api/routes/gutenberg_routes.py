"""
Project Gutenberg integration for quantum narrative testbed.

Provides access to classic literature for transformation testing with
finite storage management and comparison frameworks.
"""

import os
import json
import logging
import requests
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gutenberg", tags=["gutenberg"])

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GUTENBERG_CACHE_DIR = os.path.join(DATA_DIR, "gutenberg_cache")
TESTBED_DIR = os.path.join(DATA_DIR, "testbed")
MAX_CACHED_BOOKS = 50  # Finite storage limit
MAX_TEST_RESULTS = 100  # Finite test result storage
CACHE_EXPIRY_DAYS = 30

# Ensure directories exist
os.makedirs(GUTENBERG_CACHE_DIR, exist_ok=True)
os.makedirs(TESTBED_DIR, exist_ok=True)

# Request/Response Models
class GutenbergSearchRequest(BaseModel):
    query: str
    limit: int = 10

class BookInfo(BaseModel):
    id: int
    title: str
    authors: List[str]
    subjects: List[str]
    download_count: int
    text_url: Optional[str] = None

class TestCase(BaseModel):
    id: str
    book_id: int
    book_title: str
    excerpt_start: int
    excerpt_length: int
    excerpt_text: str
    created_at: str
    description: str

class TransformationTest(BaseModel):
    test_case_id: str
    transformation_type: str
    parameters: Dict
    original_text: str
    transformed_text: str
    quantum_distance: float
    execution_time: float
    audit_trail: Dict
    created_at: str

def get_book_hash(book_id: int) -> str:
    """Generate hash for book caching."""
    return hashlib.md5(f"gutenberg_{book_id}".encode()).hexdigest()

def cleanup_old_cache():
    """Remove old cached books to maintain finite storage."""
    try:
        cache_files = []
        for filename in os.listdir(GUTENBERG_CACHE_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(GUTENBERG_CACHE_DIR, filename)
                stat = os.stat(filepath)
                cache_files.append((filepath, stat.st_mtime))
        
        # Sort by modification time and remove oldest if over limit
        cache_files.sort(key=lambda x: x[1])
        while len(cache_files) > MAX_CACHED_BOOKS:
            oldest_file = cache_files.pop(0)
            os.remove(oldest_file[0])
            logger.info(f"Removed old cache file: {oldest_file[0]}")
            
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")

@router.get("/search")
async def search_books(query: str, limit: int = 10) -> List[BookInfo]:
    """
    Search Project Gutenberg catalog for books.
    
    Uses the Gutenberg API to find books matching the query.
    """
    try:
        # Search using Gutenberg API
        search_url = f"https://gutendex.com/books/?search={query}&page_size={limit}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        books = []
        
        for book in data.get('results', []):
            # Find plain text URL
            text_url = None
            for format_type, url in book.get('formats', {}).items():
                if 'text/plain' in format_type and 'utf-8' in format_type:
                    text_url = url
                    break
            
            books.append(BookInfo(
                id=book['id'],
                title=book['title'],
                authors=[author['name'] for author in book.get('authors', [])],
                subjects=book.get('subjects', []),
                download_count=book.get('download_count', 0),
                text_url=text_url
            ))
        
        return books
        
    except Exception as e:
        logger.error(f"Gutenberg search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/books/{book_id}")
async def get_book(book_id: int) -> Dict:
    """
    Get full book text with caching and finite storage management.
    """
    book_hash = get_book_hash(book_id)
    cache_file = os.path.join(GUTENBERG_CACHE_DIR, f"{book_hash}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            if datetime.now() - cached_time < timedelta(days=CACHE_EXPIRY_DAYS):
                logger.info(f"Returning cached book {book_id}")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
    
    try:
        # Get book info first
        info_url = f"https://gutendex.com/books/{book_id}/"
        info_response = requests.get(info_url, timeout=10)
        info_response.raise_for_status()
        book_info = info_response.json()
        
        # Find plain text URL
        text_url = None
        for format_type, url in book_info.get('formats', {}).items():
            if 'text/plain' in format_type and 'utf-8' in format_type:
                text_url = url
                break
        
        if not text_url:
            raise HTTPException(status_code=404, detail="Plain text version not available")
        
        # Download the text
        text_response = requests.get(text_url, timeout=30)
        text_response.raise_for_status()
        
        # Process text (remove Gutenberg headers/footers)
        text = text_response.text
        
        # Find start of actual content (after Gutenberg header)
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG EBOOK"
        ]
        
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                # Find end of line after marker
                start_pos = text.find('\n', pos) + 1
                break
        
        # Find end of content (before Gutenberg footer)
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "***END OF THE PROJECT GUTENBERG EBOOK"
        ]
        
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                end_pos = pos
                break
        
        clean_text = text[start_pos:end_pos].strip()
        
        # Create book data
        book_data = {
            'id': book_id,
            'title': book_info['title'],
            'authors': [author['name'] for author in book_info.get('authors', [])],
            'subjects': book_info.get('subjects', []),
            'text': clean_text,
            'word_count': len(clean_text.split()),
            'char_count': len(clean_text),
            'cached_at': datetime.now().isoformat()
        }
        
        # Cache the book (with finite storage management)
        cleanup_old_cache()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cached book {book_id}: {book_info['title']}")
        return book_data
        
    except Exception as e:
        logger.error(f"Failed to fetch book {book_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch book: {str(e)}")

@router.post("/test-cases")
async def create_test_case(
    book_id: int,
    start_position: int,
    length: int,
    description: str
) -> TestCase:
    """
    Create a test case excerpt from a Gutenberg book.
    """
    try:
        # Get the book
        book_data = await get_book(book_id)
        
        # Extract excerpt
        text = book_data['text']
        if start_position < 0 or start_position >= len(text):
            raise HTTPException(status_code=400, detail="Invalid start position")
        
        end_position = min(start_position + length, len(text))
        excerpt = text[start_position:end_position]
        
        # Create test case
        test_case_id = hashlib.md5(f"{book_id}_{start_position}_{length}_{description}".encode()).hexdigest()
        
        test_case = TestCase(
            id=test_case_id,
            book_id=book_id,
            book_title=book_data['title'],
            excerpt_start=start_position,
            excerpt_length=len(excerpt),
            excerpt_text=excerpt,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        # Save test case
        test_case_file = os.path.join(TESTBED_DIR, f"testcase_{test_case_id}.json")
        with open(test_case_file, 'w', encoding='utf-8') as f:
            json.dump(test_case.dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created test case {test_case_id}: {description}")
        return test_case
        
    except Exception as e:
        logger.error(f"Test case creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test case: {str(e)}")

@router.get("/test-cases")
async def list_test_cases() -> List[TestCase]:
    """
    List all saved test cases.
    """
    try:
        test_cases = []
        
        for filename in os.listdir(TESTBED_DIR):
            if filename.startswith('testcase_') and filename.endswith('.json'):
                filepath = os.path.join(TESTBED_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    test_case_data = json.load(f)
                test_cases.append(TestCase(**test_case_data))
        
        # Sort by creation time
        test_cases.sort(key=lambda x: x.created_at, reverse=True)
        return test_cases
        
    except Exception as e:
        logger.error(f"Failed to list test cases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test cases: {str(e)}")

@router.post("/test-results")
async def save_test_result(result: TransformationTest) -> Dict:
    """
    Save transformation test result with finite storage management.
    """
    try:
        # Generate result ID
        result_id = hashlib.md5(
            f"{result.test_case_id}_{result.transformation_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        result_data = result.dict()
        result_data['id'] = result_id
        
        # Save result
        result_file = os.path.join(TESTBED_DIR, f"result_{result_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # Cleanup old results to maintain finite storage
        cleanup_old_results()
        
        logger.info(f"Saved test result {result_id}")
        return {"id": result_id, "status": "saved"}
        
    except Exception as e:
        logger.error(f"Failed to save test result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save test result: {str(e)}")

def cleanup_old_results():
    """Remove old test results to maintain finite storage."""
    try:
        result_files = []
        for filename in os.listdir(TESTBED_DIR):
            if filename.startswith('result_') and filename.endswith('.json'):
                filepath = os.path.join(TESTBED_DIR, filename)
                stat = os.stat(filepath)
                result_files.append((filepath, stat.st_mtime))
        
        # Sort by modification time and remove oldest if over limit
        result_files.sort(key=lambda x: x[1])
        while len(result_files) > MAX_TEST_RESULTS:
            oldest_file = result_files.pop(0)
            os.remove(oldest_file[0])
            logger.info(f"Removed old test result: {oldest_file[0]}")
            
    except Exception as e:
        logger.warning(f"Result cleanup failed: {e}")

@router.get("/test-results")
async def list_test_results(test_case_id: Optional[str] = None) -> List[Dict]:
    """
    List test results, optionally filtered by test case.
    """
    try:
        results = []
        
        for filename in os.listdir(TESTBED_DIR):
            if filename.startswith('result_') and filename.endswith('.json'):
                filepath = os.path.join(TESTBED_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                if test_case_id is None or result_data.get('test_case_id') == test_case_id:
                    results.append(result_data)
        
        # Sort by creation time
        results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Failed to list test results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test results: {str(e)}")

@router.get("/popular-books")
async def get_popular_books() -> List[BookInfo]:
    """
    Get a curated list of popular books good for testing.
    """
    # Hand-curated list of books that work well for narrative transformation
    popular_book_ids = [
        11,    # Alice's Adventures in Wonderland
        1342,  # Pride and Prejudice
        74,    # The Adventures of Tom Sawyer
        84,    # Frankenstein
        2701,  # Moby Dick
        1661,  # The Adventures of Sherlock Holmes
        98,    # A Tale of Two Cities
        345,   # Dracula
        174,   # The Picture of Dorian Gray
        76,    # Adventures of Huckleberry Finn
    ]
    
    try:
        books = []
        for book_id in popular_book_ids:
            try:
                info_url = f"https://gutendex.com/books/{book_id}/"
                response = requests.get(info_url, timeout=5)
                if response.status_code == 200:
                    book_info = response.json()
                    
                    # Find text URL
                    text_url = None
                    for format_type, url in book_info.get('formats', {}).items():
                        if 'text/plain' in format_type and 'utf-8' in format_type:
                            text_url = url
                            break
                    
                    books.append(BookInfo(
                        id=book_info['id'],
                        title=book_info['title'],
                        authors=[author['name'] for author in book_info.get('authors', [])],
                        subjects=book_info.get('subjects', []),
                        download_count=book_info.get('download_count', 0),
                        text_url=text_url
                    ))
            except Exception as e:
                logger.warning(f"Failed to fetch book {book_id}: {e}")
                continue
        
        return books
        
    except Exception as e:
        logger.error(f"Failed to get popular books: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get popular books: {str(e)}")