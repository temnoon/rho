import React, { useState, useEffect, useCallback } from 'react';

// LLM Query Interface Tab
export function QueryTab() {
  const [query, setQuery] = useState('What is the favorite time of day, for many characters that you know of from literature?');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedMatrix, setSelectedMatrix] = useState('global_consciousness');
  const [availableMatrices, setAvailableMatrices] = useState([]);
  const [matricesLoading, setMatricesLoading] = useState(true);

  const safeFetch = async (path, options = {}) => {
    const baseUrl = 'http://localhost:8192';
    const fullUrl = path.startsWith('http') ? path : `${baseUrl}${path}`;
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

  const formatNumber = (n, digits = 4) => {
    if (!Number.isFinite(n)) return "NaN";
    return Number(n).toFixed(digits);
  };

  const fetchAvailableMatrices = useCallback(async () => {
    try {
      setMatricesLoading(true);
      const res = await safeFetch('/matrices/available');
      const data = await res.json();
      setAvailableMatrices(data.matrices || []);
      
      // Auto-select global consciousness if available, or first matrix
      if (data.matrices && data.matrices.length > 0) {
        const globalMatrix = data.matrices.find(m => m.type === 'global');
        if (globalMatrix && selectedMatrix === 'global_consciousness') {
          setSelectedMatrix(globalMatrix.rho_id);
        } else if (selectedMatrix === 'global_consciousness') {
          setSelectedMatrix(data.matrices[0].rho_id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch available matrices:', err);
    } finally {
      setMatricesLoading(false);
    }
  }, [selectedMatrix]);

  useEffect(() => {
    fetchAvailableMatrices();
  }, [fetchAvailableMatrices]);

  const submitQuery = async () => {
    if (!query.trim()) return;
    
    try {
      setLoading(true);
      const res = await safeFetch('/matrix/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, rho_id: selectedMatrix })
      });
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error('Failed to query matrix:', err);
      setResponse({ 
        query, 
        response: `Error: ${err.message}`,
        narrative_count: 0,
        matrix_state: null
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Matrix Query Interface</h2>
      <p style={{ color: '#666', marginBottom: 20 }}>
        Query the density matrix as if it were an LLM, but limited to loaded narratives.
      </p>

      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
          <label>Select Matrix:</label>
          <button 
            onClick={fetchAvailableMatrices}
            style={{ padding: '2px 8px', fontSize: 12, background: '#f0f0f0', border: '1px solid #ccc', borderRadius: 3 }}
            disabled={matricesLoading}
          >
            {matricesLoading ? '⟳' : '↻'}
          </button>
        </div>
        <select 
          value={selectedMatrix} 
          onChange={(e) => setSelectedMatrix(e.target.value)}
          style={{ 
            width: '100%',
            padding: '8px 12px', 
            borderRadius: 4, 
            border: '1px solid #ccc',
            background: 'white'
          }}
          disabled={matricesLoading}
        >
          {matricesLoading ? (
            <option>Loading matrices...</option>
          ) : availableMatrices.length === 0 ? (
            <option>No matrices available</option>
          ) : (
            availableMatrices.map(matrix => (
              <option key={matrix.rho_id} value={matrix.rho_id}>
                {matrix.display_name} - {matrix.description}
              </option>
            ))
          )}
        </select>
        {selectedMatrix && availableMatrices.length > 0 && (
          <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
            {(() => {
              const matrix = availableMatrices.find(m => m.rho_id === selectedMatrix);
              return matrix ? (
                <span>
                  Purity: {matrix.purity.toFixed(3)} | Entropy: {matrix.entropy.toFixed(3)} | Type: {matrix.type}
                </span>
              ) : null;
            })()}
          </div>
        )}
      </div>

      <div style={{ marginBottom: 20 }}>
        <label style={{ display: 'block', marginBottom: 5 }}>Your Query:</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask the matrix about its loaded narratives..."
          style={{ width: '100%', minHeight: 100, padding: 12, borderRadius: 4, border: '1px solid #ccc' }}
        />
      </div>

      <button 
        onClick={submitQuery}
        disabled={loading || !query.trim()}
        style={{ padding: '10px 20px', background: '#9c27b0', color: 'white', border: 'none', borderRadius: 4 }}
      >
        {loading ? 'Querying...' : 'Query Matrix'}
      </button>

      {response && (
        <div style={{ marginTop: 30 }}>
          <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 20 }}>
            <h3>Matrix Response</h3>
            <div style={{ background: '#f8f9fa', padding: 15, borderRadius: 4, marginBottom: 15 }}>
              <strong>Query:</strong> "{response.query}"
            </div>
            <div style={{ background: '#fff3e0', padding: 15, borderRadius: 4, marginBottom: 15 }}>
              <strong>Response:</strong><br/>
              {response.response}
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 15 }}>
              <div>
                <h4>Matrix State</h4>
                {response.matrix_state && (
                  <div>
                    <div>Narratives Loaded: {response.narrative_count}</div>
                    <div>Purity: {formatNumber(response.matrix_state.purity, 3)}</div>
                    <div>Entropy: {formatNumber(response.matrix_state.entropy, 3)}</div>
                    <div>Top Eigenvalue: {formatNumber(response.matrix_state.eigs[0], 3)}</div>
                  </div>
                )}
              </div>
              <div>
                <h4>Loaded Narrative Previews</h4>
                {response.loaded_narratives && response.loaded_narratives.length > 0 ? (
                  <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                    {response.loaded_narratives.map((text, idx) => (
                      <div key={idx} style={{ fontSize: 12, marginBottom: 5, padding: 5, background: '#f5f5f5', borderRadius: 3 }}>
                        {text}...
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: '#666' }}>No narratives loaded</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}