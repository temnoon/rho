import React, { useState, useCallback } from 'react';

const TestbedInterface = ({
  showTestbed,
  setShowTestbed
}) => {
  const [gutenbergBooks, setGutenbergBooks] = useState([]);
  const [testCases, setTestCases] = useState([]);
  const [selectedBook, setSelectedBook] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  // Testbed functions
  const searchGutenbergBooks = useCallback(async (query) => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    try {
      const response = await fetch(`http://localhost:8192/gutenberg/search?query=${encodeURIComponent(query)}&limit=10`);
      const books = await response.json();
      setGutenbergBooks(books);
    } catch (error) {
      console.error('Book search failed:', error);
    } finally {
      setIsSearching(false);
    }
  }, []);

  const loadPopularBooks = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8192/gutenberg/popular-books');
      const books = await response.json();
      setGutenbergBooks(books);
    } catch (error) {
      console.error('Failed to load popular books:', error);
    }
  }, []);

  const loadBook = useCallback(async (bookId) => {
    try {
      const response = await fetch(`http://localhost:8192/gutenberg/books/${bookId}`);
      const book = await response.json();
      setSelectedBook(book);
    } catch (error) {
      console.error('Failed to load book:', error);
    }
  }, []);

  if (!showTestbed) {
    return (
      <div style={{
        maxWidth: '1200px',
        margin: '20px auto',
        background: 'rgba(255,255,255,0.95)',
        borderRadius: '20px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
        overflow: 'hidden'
      }}>
        <div style={{
          padding: '20px 30px',
          background: 'white'
        }}>
          <button
            onClick={() => setShowTestbed(true)}
            style={{
              background: 'none',
              border: 'none',
              color: '#667eea',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <span>🔽</span>
            Open Testbed Interface
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '20px auto',
      background: 'rgba(255,255,255,0.95)',
      borderRadius: '20px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 30px',
        borderBottom: '1px solid #e9ecef',
        background: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{ margin: 0, color: '#495057' }}>
          🧪 Transformation Testbed
        </h3>
        <button
          onClick={() => setShowTestbed(false)}
          style={{
            background: 'none',
            border: 'none',
            color: '#667eea',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <span>🔼</span>
          Close Testbed
        </button>
      </div>

      {/* Testbed Content */}
      <div style={{ padding: '30px' }}>
        {/* Search Interface */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
            📚 Project Gutenberg Search
          </h4>
          <div style={{
            display: 'flex',
            gap: '12px',
            marginBottom: '15px'
          }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for books by title or author..."
              style={{
                flex: 1,
                padding: '12px 16px',
                border: '2px solid #e9ecef',
                borderRadius: '8px',
                fontSize: '14px',
                outline: 'none'
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && searchQuery.trim()) {
                  searchGutenbergBooks(searchQuery);
                }
              }}
            />
            <button
              onClick={() => searchGutenbergBooks(searchQuery)}
              disabled={!searchQuery.trim() || isSearching}
              style={{
                padding: '12px 20px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                cursor: searchQuery.trim() && !isSearching ? 'pointer' : 'not-allowed',
                opacity: searchQuery.trim() && !isSearching ? 1 : 0.5
              }}
            >
              {isSearching ? '🔍' : 'Search'}
            </button>
            <button
              onClick={loadPopularBooks}
              style={{
                padding: '12px 20px',
                background: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                cursor: 'pointer'
              }}
            >
              📈 Popular
            </button>
          </div>
        </div>

        {/* Books Grid */}
        {gutenbergBooks.length > 0 && (
          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
              Search Results ({gutenbergBooks.length} books)
            </h4>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
              gap: '15px',
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              {gutenbergBooks.map((book, idx) => (
                <div
                  key={idx}
                  onClick={() => loadBook(book.id)}
                  style={{
                    padding: '15px',
                    background: 'white',
                    border: '2px solid #e9ecef',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.borderColor = '#667eea';
                    e.target.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.borderColor = '#e9ecef';
                    e.target.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ fontWeight: '600', marginBottom: '5px', fontSize: '14px' }}>
                    {book.title}
                  </div>
                  <div style={{ fontSize: '12px', color: '#6c757d' }}>
                    {book.author || 'Unknown Author'}
                  </div>
                  <div style={{ fontSize: '12px', color: '#667eea', marginTop: '5px' }}>
                    ID: {book.id}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Selected Book */}
        {selectedBook && (
          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
              📖 Selected Book: {selectedBook.title}
            </h4>
            <div style={{
              padding: '20px',
              background: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '8px'
            }}>
              <div style={{ marginBottom: '10px' }}>
                <strong>Author:</strong> {selectedBook.author || 'Unknown'}
              </div>
              <div style={{ marginBottom: '10px' }}>
                <strong>Language:</strong> {selectedBook.language || 'Unknown'}
              </div>
              {selectedBook.text && (
                <div>
                  <strong>Sample Text:</strong>
                  <div style={{
                    marginTop: '10px',
                    padding: '15px',
                    background: 'white',
                    border: '1px solid #dee2e6',
                    borderRadius: '6px',
                    fontSize: '14px',
                    lineHeight: '1.6',
                    maxHeight: '150px',
                    overflowY: 'auto'
                  }}>
                    {selectedBook.text.substring(0, 500)}...
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Test Cases */}
        <div>
          <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
            🧪 Test Cases
          </h4>
          <div style={{
            padding: '20px',
            background: '#f8f9fa',
            border: '1px solid #e9ecef',
            borderRadius: '8px',
            textAlign: 'center',
            color: '#6c757d'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '10px' }}>🚧</div>
            <div>Test case management coming soon...</div>
            <div style={{ fontSize: '12px', marginTop: '5px' }}>
              This will allow saving and managing transformation test scenarios
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestbedInterface;