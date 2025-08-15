import React, { useState, useEffect } from 'react';

// Book Reader Tab - Project Gutenberg Integration
export function BookReaderTab() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchMeta, setSearchMeta] = useState({ total: 0, offset: 0, limit: 20, has_more: false });
  const [authorFilter, setAuthorFilter] = useState('');
  const [resultsPerPage, setResultsPerPage] = useState(20);
  const [currentPage, setCurrentPage] = useState(0);
  const [selectedBook, setSelectedBook] = useState(null);
  const [bookProgress, setBookProgress] = useState(null);
  const [isReading, setIsReading] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState('meaning_flow');
  const [loading, setLoading] = useState(false);
  const [latestPassage, setLatestPassage] = useState(null);
  const [readingHistory, setReadingHistory] = useState([]);

  const safeFetch = async (url, options = {}) => {
    const baseUrl = 'http://localhost:8192';
    const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`;
    const response = await fetch(fullUrl, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response;
  };

  const searchBooks = async (page = 0, limit = resultsPerPage, authorFilter = '') => {
    if (!searchQuery.trim()) return;
    try {
      setLoading(true);
      const offset = page * limit;
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
        author_filter: authorFilter
      });
      const res = await safeFetch(`/gutenberg/search/${encodeURIComponent(searchQuery)}?${params}`);
      const data = await res.json();
      setSearchResults(data.books || data.results || []);
      setSearchMeta({
        total: data.total || 0,
        offset: data.offset || 0,
        limit: data.limit || resultsPerPage,
        has_more: data.has_more || false
      });
      setCurrentPage(page);
    } catch (err) {
      console.error('Search failed:', err);
      setSearchResults([]);
      setSearchMeta({ total: 0, offset: 0, limit: resultsPerPage, has_more: false });
    } finally {
      setLoading(false);
    }
  };

  const loadMoreResults = () => {
    if (searchMeta.has_more) {
      searchBooks(currentPage + 1, resultsPerPage, authorFilter);
    }
  };

  const ingestBook = async (bookId, title, author) => {
    try {
      setLoading(true);
      const res = await safeFetch('/gutenberg/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gutenberg_id: bookId,
          chunk_size: 800,
          reading_alpha: 0.25
        })
      });
      const data = await res.json();
      setSelectedBook({
        ...data,
        id: bookId,
        originalTitle: title,
        originalAuthor: author
      });
    } catch (err) {
      console.error('Book ingestion failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const readNextChunk = async () => {
    if (!selectedBook || !bookProgress) return;
    if (bookProgress.chunks_processed >= bookProgress.total_chunks) return;
    
    try {
      setIsReading(true);
      const res = await safeFetch(
        `/gutenberg/${selectedBook.book_rho_id}/read_chunk/${bookProgress.chunks_processed}`
      , {
        method: 'POST'
      });
      const data = await res.json();
      
      // Store the latest passage for display
      if (data.passage_preview) {
        setLatestPassage({
          chunk_index: data.chunk_index,
          text: data.passage_preview,
          timestamp: new Date().toLocaleTimeString()
        });
        
        // Add to reading history (keep last 10 for review)
        setReadingHistory(prev => [
          {
            chunk_index: data.chunk_index,
            text: data.passage_preview,
            timestamp: new Date().toLocaleTimeString()
          },
          ...prev.slice(0, 9)
        ]);
      }
      
      setBookProgress(prev => ({
        ...prev,
        chunks_processed: data.chunks_processed,
        progress: data.progress,
        matrix_state: data.matrix_state,
        latest_commentary: data.llm_commentary,
        hierarchical_status: data.hierarchical_status
      }));
    } catch (err) {
      console.error('Reading chunk failed:', err);
    } finally {
      setIsReading(false);
    }
  };

  const readMultipleChunks = async (numChunks) => {
    if (!selectedBook || !bookProgress) return;
    
    try {
      setIsReading(true);
      let currentChunk = bookProgress.chunks_processed;
      
      for (let i = 0; i < numChunks && currentChunk < bookProgress.total_chunks; i++) {
        const res = await safeFetch(
          `/gutenberg/${selectedBook.book_rho_id}/read_chunk/${currentChunk}`,
          { method: 'POST' }
        );
        const data = await res.json();
        
        // Store the latest passage for display
        if (data.passage_preview) {
          setLatestPassage({
            chunk_index: data.chunk_index,
            text: data.passage_preview,
            timestamp: new Date().toLocaleTimeString()
          });
          
          // Add to reading history (keep last 10 for review)
          setReadingHistory(prev => [
            {
              chunk_index: data.chunk_index,
              text: data.passage_preview,
              timestamp: new Date().toLocaleTimeString()
            },
            ...prev.slice(0, 9)
          ]);
        }
        
        setBookProgress(prev => ({
          ...prev,
          chunks_processed: data.chunks_processed,
          progress: data.progress,
          matrix_state: data.matrix_state,
          latest_commentary: data.llm_commentary,
          hierarchical_status: data.hierarchical_status
        }));
        
        currentChunk = data.chunks_processed;
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    } catch (err) {
      console.error('Reading multiple chunks failed:', err);
    } finally {
      setIsReading(false);
    }
  };

  const readWholeBook = async () => {
    if (!selectedBook || !bookProgress) return;
    
    try {
      setIsReading(true);
      let currentChunk = bookProgress.chunks_processed;
      const maxChunks = bookProgress.total_chunks;
      
      // Prevent infinite loops
      let attempts = 0;
      const maxAttempts = maxChunks + 10; // Safety margin
      
      while (currentChunk < maxChunks && attempts < maxAttempts) {
        // Read the next chunk explicitly
        const res = await safeFetch(
          `/gutenberg/${selectedBook.book_rho_id}/read_chunk/${currentChunk}`,
          { method: 'POST' }
        );
        const data = await res.json();
        
        // Store the latest passage for display (less frequently to avoid spam)
        if (data.passage_preview && currentChunk % 5 === 0) { // Only every 5th chunk
          setLatestPassage({
            chunk_index: data.chunk_index,
            text: data.passage_preview,
            timestamp: new Date().toLocaleTimeString()
          });
          
          // Add to reading history (keep last 10 for review)
          setReadingHistory(prev => [
            {
              chunk_index: data.chunk_index,
              text: data.passage_preview,
              timestamp: new Date().toLocaleTimeString()
            },
            ...prev.slice(0, 9)
          ]);
        }
        
        // Update local state
        setBookProgress(prev => ({
          ...prev,
          chunks_processed: data.chunks_processed,
          progress: data.progress,
          matrix_state: data.matrix_state,
          latest_commentary: data.llm_commentary,
          hierarchical_status: data.hierarchical_status
        }));
        
        // Move to next chunk
        currentChunk = data.chunks_processed;
        attempts++;
        
        // Small delay to show progress and prevent server overload
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Break if no progress is being made
        if (attempts > 5 && currentChunk === bookProgress.chunks_processed) {
          console.warn('No progress detected, stopping to prevent infinite loop');
          break;
        }
      }
      
      if (attempts >= maxAttempts) {
        console.warn('Reached maximum attempts, stopping to prevent infinite loop');
      }
      
    } catch (err) {
      console.error('Reading whole book failed:', err);
    } finally {
      setIsReading(false);
    }
  };

  const getProgress = async () => {
    if (!selectedBook) return;
    try {
      const res = await safeFetch(`/gutenberg/${selectedBook.book_rho_id}/progress`);
      const data = await res.json();
      setBookProgress(data);
    } catch (err) {
      console.error('Progress fetch failed:', err);
    }
  };

  useEffect(() => {
    if (selectedBook) {
      getProgress();
      const interval = setInterval(getProgress, 2000); // Update every 2 seconds
      return () => clearInterval(interval);
    }
  }, [selectedBook]);

  return (
    <div style={{ padding: 20 }}>
      <h2>Project Gutenberg Book Reader</h2>
      <p style={{ color: '#666', marginBottom: 20 }}>
        Watch how meaning emerges as Rho reads classic literature, chunk by chunk.
      </p>

      {/* Search Interface */}
      <div style={{ marginBottom: 30, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
        <h3>Find a Book</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr auto auto auto', gap: 10, marginBottom: 15, alignItems: 'end' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 600 }}>Search Query:</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search titles, authors, or genres (Wells, Horror, Shakespeare...)"
              style={{ width: '100%', padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc' }}
              onKeyPress={(e) => e.key === 'Enter' && searchBooks()}
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 600 }}>Author Filter:</label>
            <input
              type="text"
              value={authorFilter}
              onChange={(e) => setAuthorFilter(e.target.value)}
              placeholder="Filter by author"
              style={{ padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 600 }}>Results:</label>
            <select
              value={resultsPerPage}
              onChange={(e) => setResultsPerPage(Number(e.target.value))}
              style={{ padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
          <button
            onClick={() => searchBooks(0, resultsPerPage, authorFilter)}
            disabled={loading || !searchQuery.trim()}
            style={{
              padding: '8px 16px',
              background: loading || !searchQuery.trim() ? '#ccc' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: loading || !searchQuery.trim() ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Results */}
        {searchResults && searchResults.length > 0 && (
          <div>
            <h4>Search Results ({searchMeta.total} total)</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 15 }}>
              {searchResults.map((book, idx) => (
                <div key={idx} style={{
                  border: '1px solid #ddd',
                  borderRadius: 8,
                  padding: 15,
                  background: '#f9f9f9',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onClick={() => ingestBook(book.id, book.title, book.author)}
                onMouseEnter={(e) => e.currentTarget.style.background = '#f0f0f0'}
                onMouseLeave={(e) => e.currentTarget.style.background = '#f9f9f9'}
                >
                  <h5 style={{ margin: '0 0 8px 0', fontSize: 16, fontWeight: 600 }}>
                    {book.title}
                  </h5>
                  <p style={{ margin: '0 0 8px 0', color: '#666', fontSize: 14 }}>
                    by {book.author}
                  </p>
                  <div style={{ fontSize: 12, color: '#888' }}>
                    ID: {book.id}
                    {book.subjects && book.subjects.length > 0 && (
                      <>
                        <br />
                        Subjects: {book.subjects.slice(0, 3).join(', ')}
                        {book.subjects.length > 3 && '...'}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Book Progress - FIXED LAYOUT: Visualization LEFT, Text RIGHT */}
      {selectedBook && bookProgress && (
        <div style={{ marginBottom: 30 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '400px 1fr', gap: 20 }}>
            
            {/* LEFT COLUMN: Visualization and Controls */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 15 }}>
              
              {/* Book Info and Controls */}
              <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
                <h3>Now Reading</h3>
                <div style={{ marginBottom: 15 }}>
                  <div style={{ fontWeight: 600, fontSize: 16 }}>{bookProgress.title}</div>
                  <div style={{ color: '#666' }}>by {bookProgress.author}</div>
                </div>

                <div style={{ marginBottom: 15 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, marginBottom: 5 }}>
                    <span>Progress</span>
                    <span>{Math.round(bookProgress.completion_percentage || 0)}%</span>
                  </div>
                  <div style={{ 
                    background: '#f0f0f0', 
                    borderRadius: 10, 
                    height: 8,
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      background: '#2196f3', 
                      height: '100%', 
                      width: `${bookProgress.completion_percentage || 0}%`,
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
                    {bookProgress.chunks_processed} / {bookProgress.total_chunks} chunks
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <button
                    onClick={readNextChunk}
                    disabled={isReading || bookProgress.chunks_processed >= bookProgress.total_chunks}
                    style={{ 
                      padding: '8px 16px', 
                      background: bookProgress.chunks_processed >= bookProgress.total_chunks ? '#ccc' : '#4caf50',
                      color: 'white', 
                      border: 'none', 
                      borderRadius: 4
                    }}
                  >
                    {isReading ? 'Reading...' : 
                     bookProgress.chunks_processed >= bookProgress.total_chunks ? 'Complete' : 
                     'Read Next Chunk'}
                  </button>

                  <div style={{ display: 'flex', gap: 4 }}>
                    <button
                      onClick={() => readMultipleChunks(5)}
                      disabled={isReading || bookProgress.chunks_processed >= bookProgress.total_chunks}
                      style={{ 
                        flex: 1,
                        padding: '6px 8px', 
                        background: '#ff9800',
                        color: 'white', 
                        border: 'none', 
                        borderRadius: 4,
                        fontSize: 12
                      }}
                    >
                      Read 5
                    </button>
                    <button
                      onClick={() => readMultipleChunks(10)}
                      disabled={isReading || bookProgress.chunks_processed >= bookProgress.total_chunks}
                      style={{ 
                        flex: 1,
                        padding: '6px 8px', 
                        background: '#ff9800',
                        color: 'white', 
                        border: 'none', 
                        borderRadius: 4,
                        fontSize: 12
                      }}
                    >
                      Read 10
                    </button>
                  </div>

                  <div style={{ display: 'flex', gap: 4 }}>
                    <button
                      onClick={readWholeBook}
                      disabled={isReading || bookProgress.chunks_processed >= bookProgress.total_chunks}
                      style={{ 
                        flex: 1,
                        padding: '10px 16px', 
                        background: bookProgress.chunks_processed >= bookProgress.total_chunks ? '#ccc' : '#9c27b0',
                        color: 'white', 
                        border: 'none', 
                        borderRadius: 4,
                        fontWeight: 600
                      }}
                    >
                      🚀 Read Whole Book
                    </button>
                    
                    {(bookProgress.chunks_processed > bookProgress.total_chunks || isReading) && (
                      <button
                        onClick={() => {
                          setIsReading(false);
                          setSelectedBook(null);
                          setBookProgress(null);
                        }}
                        style={{ 
                          padding: '10px 12px', 
                          background: '#f44336',
                          color: 'white', 
                          border: 'none', 
                          borderRadius: 4,
                          fontWeight: 600
                        }}
                      >
                        🛑
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Compact Visualization */}
              <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <h4 style={{ margin: 0, fontSize: 16 }}>Meaning Emergence</h4>
                  <select 
                    value={visualizationMode} 
                    onChange={(e) => setVisualizationMode(e.target.value)}
                    style={{ padding: '2px 6px', borderRadius: 4, border: '1px solid #ccc', fontSize: 12 }}
                  >
                    <option value="meaning_flow">Meaning Flow</option>
                    <option value="eigenvalue_river">Eigenvalue River</option>
                    <option value="entropy_journey">Entropy Journey</option>
                  </select>
                </div>
                
                {/* Compact Visualization */}
                <div style={{
                  width: '100%',
                  height: 250,
                  background: '#f8f9fa',
                  borderRadius: 6,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px dashed #dee2e6',
                  fontSize: 12,
                  color: '#6c757d',
                  textAlign: 'center'
                }}>
                  <div>
                    <div style={{ fontSize: 32, marginBottom: 8 }}>📊</div>
                    <div>{visualizationMode.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                    <div style={{ fontSize: 10, marginTop: 5 }}>
                      {bookProgress && bookProgress.matrix_state ? 
                        `Entropy: ${bookProgress.matrix_state.entropy?.toFixed(3)} | Chunks: ${bookProgress.chunks_processed}` :
                        'Start reading to see patterns'
                      }
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* RIGHT COLUMN: Text Content */}
            <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 20, minHeight: 600 }}>
              <h3 style={{ marginTop: 0, marginBottom: 15 }}>Reading Content</h3>
              
              {/* Latest Passage - Full Text */}
              {latestPassage && (
                <div style={{ marginBottom: 30 }}>
                  <h4 style={{ 
                    margin: '0 0 15px 0',
                    fontSize: 16,
                    fontWeight: 600,
                    color: '#495057',
                    borderBottom: '2px solid #e9ecef',
                    paddingBottom: 8
                  }}>
                    Latest Passage (Chunk {latestPassage.chunk_index})
                  </h4>
                  <div style={{
                    fontFamily: 'Georgia, serif',
                    fontSize: 14,
                    lineHeight: 1.7,
                    color: '#343a40',
                    padding: '15px 20px',
                    background: '#f8f9fa',
                    borderRadius: 8,
                    border: '1px solid #e9ecef',
                    marginBottom: 10
                  }}>
                    {latestPassage.text}
                  </div>
                  <div style={{ 
                    fontSize: 12, 
                    color: '#6c757d',
                    textAlign: 'right'
                  }}>
                    Read at {latestPassage.timestamp}
                  </div>
                </div>
              )}

              {/* Reading History - Compact */}
              {readingHistory.length > 0 && (
                <div>
                  <h4 style={{ 
                    margin: '0 0 15px 0',
                    fontSize: 16,
                    fontWeight: 600,
                    color: '#495057',
                    borderBottom: '2px solid #e9ecef',
                    paddingBottom: 8
                  }}>
                    Recent Passages
                  </h4>
                  <div style={{ 
                    maxHeight: 400,
                    overflowY: 'auto',
                    border: '1px solid #e9ecef',
                    borderRadius: 8
                  }}>
                    {readingHistory.map((passage, idx) => (
                      <div key={idx} style={{
                        padding: '12px 15px',
                        borderBottom: idx < readingHistory.length - 1 ? '1px solid #e9ecef' : 'none',
                        background: idx % 2 === 0 ? '#f8f9fa' : 'white'
                      }}>
                        <div style={{ 
                          fontWeight: 600, 
                          marginBottom: 8,
                          color: '#6c757d',
                          fontSize: 12
                        }}>
                          Chunk {passage.chunk_index} - {passage.timestamp}
                        </div>
                        <div style={{
                          fontFamily: 'Georgia, serif',
                          color: '#343a40',
                          lineHeight: 1.6,
                          fontSize: 13
                        }}>
                          {passage.text}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Placeholder when no content */}
              {!latestPassage && readingHistory.length === 0 && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: 300,
                  color: '#6c757d',
                  textAlign: 'center',
                  fontSize: 14
                }}>
                  <div>
                    <div style={{ fontSize: 48, marginBottom: 15 }}>📖</div>
                    <div>Click "Read Next Chunk" to start reading</div>
                    <div style={{ fontSize: 12, marginTop: 5 }}>
                      The full text of each passage will appear here
                    </div>
                  </div>
                </div>
              )}
              
            </div>

          </div>
        </div>
      )}
    </div>
  );
}