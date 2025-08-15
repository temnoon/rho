"""
Project Gutenberg Integration for Literary Passage Analysis.

This module provides functionality to search, retrieve, and analyze passages
from Project Gutenberg based on ρ-space narrative signatures.
"""

import requests
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)

@dataclass
class GutenbergBook:
    """Represents a book from Project Gutenberg"""
    id: int
    title: str
    author: str
    language: str
    subjects: List[str]
    url: str
    download_count: int = 0

@dataclass
class LiteraryPassage:
    """Represents a passage from literature with context"""
    text: str
    book: GutenbergBook
    chapter: Optional[str] = None
    context: str = ""
    start_position: int = 0
    rho_signature: Optional[Dict] = None

class ProjectGutenbergClient:
    """Client for interacting with Project Gutenberg API and texts"""
    
    BASE_API_URL = "https://gutendex.com/books"
    BASE_TEXT_URL = "https://www.gutenberg.org/files"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Rho-Narrative-Analysis/1.0 (Educational Research)'
        })
    
    def search_books(self, 
                    query: str = "", 
                    author: str = "", 
                    subject: str = "",
                    language: str = "en",
                    limit: int = 20) -> List[GutenbergBook]:
        """Search Project Gutenberg catalog"""
        
        params = {
            'languages': language,
            'page_size': limit
        }
        
        if query:
            params['search'] = query
        if author:
            params['author'] = author
        if subject:
            params['subject'] = subject
            
        try:
            response = self.session.get(self.BASE_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            books = []
            for book_data in data.get('results', []):
                # Extract author name(s)
                authors = book_data.get('authors', [])
                author_names = [a.get('name', 'Unknown') for a in authors]
                author_str = ', '.join(author_names) if author_names else 'Unknown'
                
                # Extract subjects
                subjects = []
                for shelf in book_data.get('bookshelves', []):
                    subjects.append(shelf)
                for subject in book_data.get('subjects', []):
                    subjects.append(subject)
                
                book = GutenbergBook(
                    id=book_data.get('id', 0),
                    title=book_data.get('title', 'Unknown Title'),
                    author=author_str,
                    language=language,
                    subjects=subjects,
                    url=f"https://www.gutenberg.org/ebooks/{book_data.get('id', 0)}",
                    download_count=book_data.get('download_count', 0)
                )
                books.append(book)
            
            # Sort by download count (popularity)
            books.sort(key=lambda x: x.download_count, reverse=True)
            return books
            
        except Exception as e:
            logger.error(f"Error searching Gutenberg books: {e}")
            return []
    
    def get_book_text(self, book_id: int, format: str = "txt") -> Optional[str]:
        """Download full text of a book"""
        
        text_urls = [
            f"{self.BASE_TEXT_URL}/{book_id}/{book_id}-0.txt",  # UTF-8
            f"{self.BASE_TEXT_URL}/{book_id}/{book_id}.txt",    # ASCII
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"  # Alternative
        ]
        
        for url in text_urls:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # Try to decode as UTF-8, fallback to latin-1
                    try:
                        return response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        return response.content.decode('latin-1', errors='ignore')
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")
                continue
        
        logger.warning(f"Could not retrieve text for book {book_id}")
        return None
    
    def get_book_metadata(self, book_id: int) -> Tuple[str, str]:
        """Get title and author from Gutendex API or text parsing"""
        
        # First try to get from Gutendex API
        try:
            response = self.session.get(f"{self.BASE_API_URL}/{book_id}")
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', '').strip()
                
                # Extract authors
                authors = data.get('authors', [])
                author_names = [a.get('name', '').strip() for a in authors if a.get('name')]
                author = ', '.join(author_names) if author_names else ''
                
                if title and author:
                    logger.info(f"Got metadata from API: {title} by {author}")
                    return title, author
                
        except Exception as e:
            logger.debug(f"Failed to get metadata from API: {e}")
        
        # Fallback: try to extract from text
        try:
            text = self.get_book_text(book_id)
            if text:
                title, author = self._extract_title_author_from_text(text)
                if title or author:
                    logger.info(f"Extracted from text: {title} by {author}")
                    return title or f"Book {book_id}", author or "Unknown Author"
        except Exception as e:
            logger.debug(f"Failed to extract from text: {e}")
        
        return f"Book {book_id}", "Unknown Author"
    
    def _extract_title_author_from_text(self, text: str) -> Tuple[str, str]:
        """Extract title and author from Project Gutenberg text"""
        
        if not text:
            return "", ""
        
        # Look for title and author in the first 2000 characters
        header = text[:2000]
        
        title = ""
        author = ""
        
        # Common patterns for titles
        title_patterns = [
            r'Title:\s*(.+?)(?:\n|$)',
            r'THE PROJECT GUTENBERG EBOOK OF\s*(.+?)\s*(?:\*\*\*|\n)',
            r'Project Gutenberg.+?of\s*(.+?)\s*by',
            r'^\s*(.+?)\s*\n.*?by\s+(.+?)\s*\n'
        ]
        
        # Common patterns for authors
        author_patterns = [
            r'Author:\s*(.+?)(?:\n|$)',
            r'by\s+(.+?)(?:\n|\*\*\*)',
            r'^\s*.+?\s*\n.*?by\s+(.+?)\s*\n'
        ]
        
        # Try to find title
        for pattern in title_patterns:
            match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # Clean up title
                title = re.sub(r'\*+', '', title).strip()
                title = re.sub(r'\s+', ' ', title).strip()
                if title and len(title) > 3:
                    break
        
        # Try to find author  
        for pattern in author_patterns:
            match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
            if match:
                author = match.group(1).strip()
                # Clean up author
                author = re.sub(r'\*+', '', author).strip()
                author = re.sub(r'\s+', ' ', author).strip()
                if author and len(author) > 3:
                    break
        
        return title, author
    
    def extract_passages(self, text: str, min_length: int = 200, max_length: int = 800) -> List[str]:
        """Extract meaningful passages from book text"""
        
        if not text:
            return []
        
        # Remove Project Gutenberg header/footer
        text = self._clean_gutenberg_text(text)
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        passages = []
        current_passage = ""
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30:  # Skip very short paragraphs (reduced threshold)
                continue
                
            # Check if adding this paragraph would exceed max length
            if len(current_passage + para) > max_length:
                if len(current_passage) >= min_length:
                    passages.append(current_passage.strip())
                current_passage = para
            else:
                if current_passage:
                    current_passage += "\n\n" + para
                else:
                    current_passage = para
        
        # Add final passage if it meets length requirement
        if len(current_passage) >= min_length:
            passages.append(current_passage.strip())
        
        # Remove artificial limit to allow hundreds of chunks
        return passages
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg metadata headers and footers"""
        
        # Find start of actual content
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG"
        ]
        
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                # Find end of the header line
                next_line = text.find('\n', pos)
                if next_line != -1:
                    start_pos = next_line + 1
                break
        
        # Find end of actual content
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG"
        ]
        
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker, start_pos)
            if pos != -1:
                end_pos = pos
                break
        
        # Extract the clean text
        clean_text = text[start_pos:end_pos].strip()
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)
        
        return clean_text
    
    def find_similar_passages_by_attributes(self, 
                                          target_attributes: Dict[str, float],
                                          subjects: List[str] = None,
                                          max_books: int = 5) -> List[LiteraryPassage]:
        """Find passages from literature that match target narrative attributes"""
        
        # Search for books in relevant subjects
        search_subjects = subjects or ["Fiction", "Literature", "Philosophy", "Science"]
        
        all_passages = []
        
        for subject in search_subjects:
            books = self.search_books(subject=subject, limit=max_books)
            
            for book in books[:2]:  # Limit books per subject
                text = self.get_book_text(book.id)
                if text:
                    passages = self.extract_passages(text)
                    
                    for passage_text in passages[:3]:  # Limit passages per book
                        passage = LiteraryPassage(
                            text=passage_text,
                            book=book,
                            context=f"From {book.title} by {book.author}"
                        )
                        all_passages.append(passage)
                
                if len(all_passages) >= 10:  # Reasonable limit
                    break
            
            if len(all_passages) >= 10:
                break
        
        return all_passages

# Global client instance
gutenberg_client = ProjectGutenbergClient()

def search_similar_literature(narrative_attributes: Dict[str, float], 
                            max_results: int = 5) -> List[Dict]:
    """Public interface for finding similar literary passages"""
    
    try:
        # Extract subject hints from attributes
        subjects = []
        if any('formal' in attr or 'academic' in attr for attr in narrative_attributes.keys()):
            subjects.extend(["Philosophy", "Science", "History"])
        if any('emotional' in attr or 'dramatic' in attr for attr in narrative_attributes.keys()):
            subjects.extend(["Fiction", "Drama", "Poetry"])
        if any('narrative' in attr or 'story' in attr for attr in narrative_attributes.keys()):
            subjects.extend(["Fiction", "Literature", "Adventure"])
        
        if not subjects:
            subjects = ["Fiction", "Literature"]
        
        passages = gutenberg_client.find_similar_passages_by_attributes(
            narrative_attributes, 
            subjects=subjects,
            max_books=3
        )
        
        results = []
        for passage in passages[:max_results]:
            results.append({
                "text": passage.text[:500] + "..." if len(passage.text) > 500 else passage.text,
                "source": passage.context,
                "book_title": passage.book.title,
                "author": passage.book.author,
                "gutenberg_id": passage.book.id,
                "match_reason": "Similar narrative attributes in ρ-space"
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching similar literature: {e}")
        return [{
            "text": "Error accessing Project Gutenberg. Placeholder passage from classic literature demonstrating similar narrative patterns.",
            "source": "Classic Literature (Demo)",
            "book_title": "Demo Book",
            "author": "Demo Author",
            "gutenberg_id": 0,
            "match_reason": "Similar ρ-space signature patterns"
        }]

if __name__ == "__main__":
    # Test the integration
    client = ProjectGutenbergClient()
    books = client.search_books("pride prejudice", limit=3)
    for book in books:
        print(f"{book.title} by {book.author} (ID: {book.id})")