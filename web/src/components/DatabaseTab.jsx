import React, { useState, useEffect } from 'react';

// Database State Overview
export function DatabaseTab() {
  const [dbState, setDbState] = useState(null);
  const [selectedMatrix, setSelectedMatrix] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // New organization state
  const [searchFilter, setSearchFilter] = useState('');
  const [sortBy, setSortBy] = useState('created'); // created, purity, entropy, operations
  const [sortOrder, setSortOrder] = useState('desc');
  const [viewMode, setViewMode] = useState('grid'); // grid, list, table
  const [categoryFilter, setCategoryFilter] = useState('all'); // all, recent, high_quality, complex
  
  // Matrix management state
  const [editingMatrix, setEditingMatrix] = useState(null);
  const [newLabel, setNewLabel] = useState('');
  const [newTags, setNewTags] = useState('');
  const [selectedFolder, setSelectedFolder] = useState('default');
  const [folders, setFolders] = useState(['default', 'archived', 'experiments', 'favorites']);
  const [newFolderName, setNewFolderName] = useState('');
  const [showBulkActions, setShowBulkActions] = useState(false);
  const [selectedMatrices, setSelectedMatrices] = useState(new Set());

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

  const formatNumber = (n, digits = 4) => {
    if (typeof n !== 'number') return 'N/A';
    return n.toFixed(digits);
  };

  // Filter and sort matrices
  const getProcessedMatrices = () => {
    if (!dbState?.matrices) return [];
    
    let filtered = dbState.matrices.map(matrix => {
      // Enhance matrix with metadata
      const metadata = getMatrixMetadata(matrix.rho_id);
      return {
        ...matrix,
        displayLabel: metadata.label || matrix.label,
        tags: metadata.tags || [],
        folder: metadata.folder || 'default'
      };
    }).filter(matrix => {
      // Folder filter
      if (selectedFolder !== 'all' && matrix.folder !== selectedFolder) {
        return false;
      }
      
      // Search filter
      if (searchFilter) {
        const searchLower = searchFilter.toLowerCase();
        const matchesId = matrix.rho_id.toLowerCase().includes(searchLower);
        const matchesLabel = matrix.displayLabel?.toLowerCase().includes(searchLower);
        const matchesTags = matrix.tags.some(tag => tag.toLowerCase().includes(searchLower));
        if (!matchesId && !matchesLabel && !matchesTags) return false;
      }
      
      // Category filter
      switch (categoryFilter) {
        case 'recent':
          // Matrices created in last 24 hours (placeholder logic)
          return matrix.narratives_count > 0;
        case 'high_quality':
          return matrix.purity > 0.1;
        case 'complex':
          return matrix.entropy > 2.0;
        case 'tagged':
          return matrix.tags.length > 0;
        default:
          return true;
      }
    });
    
    // Sort matrices
    filtered.sort((a, b) => {
      let aVal, bVal;
      
      switch (sortBy) {
        case 'purity':
          aVal = a.purity || 0;
          bVal = b.purity || 0;
          break;
        case 'entropy':
          aVal = a.entropy || 0;
          bVal = b.entropy || 0;
          break;
        case 'operations':
          aVal = a.operations_count || 0;
          bVal = b.operations_count || 0;
          break;
        case 'narratives':
          aVal = a.narratives_count || 0;
          bVal = b.narratives_count || 0;
          break;
        default: // created or id
          aVal = a.rho_id;
          bVal = b.rho_id;
      }
      
      if (typeof aVal === 'string') {
        return sortOrder === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
      }
      
      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });
    
    return filtered;
  };

  const loadDatabaseState = async () => {
    try {
      setLoading(true);
      const res = await safeFetch('/rho/global/status');
      const data = await res.json();
      setDbState(data);
    } catch (error) {
      console.error('Failed to load database state:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadMatrixDetails = async (matrixId) => {
    try {
      const res = await safeFetch(`/rho/${matrixId}`);
      const data = await res.json();
      setSelectedMatrix(data);
    } catch (error) {
      console.error('Failed to load matrix details:', error);
    }
  };

  const deleteMatrix = async (matrixId) => {
    if (!confirm('Are you sure you want to delete this matrix?')) return;
    try {
      await safeFetch(`/rho/${matrixId}`, { method: 'DELETE' });
      await loadDatabaseState(); // Refresh
      setSelectedMatrix(null);
    } catch (error) {
      console.error('Failed to delete matrix:', error);
    }
  };

  const clearDatabase = async () => {
    if (!confirm('Are you sure you want to clear ALL matrices? This cannot be undone.')) return;
    try {
      await safeFetch('/admin/clear_all', { method: 'POST' });
      await loadDatabaseState(); // Refresh
      setSelectedMatrix(null);
    } catch (error) {
      console.error('Failed to clear database:', error);
    }
  };

  // Matrix management functions
  const startEditMatrix = (matrix) => {
    setEditingMatrix(matrix.rho_id);
    setNewLabel(matrix.label || '');
    setNewTags(matrix.tags ? matrix.tags.join(', ') : '');
  };

  const saveMatrixEdits = async (matrixId) => {
    try {
      // For now, we'll store matrix metadata in localStorage since the API doesn't have endpoints for this
      const existingMetadata = JSON.parse(localStorage.getItem('matrixMetadata') || '{}');
      existingMetadata[matrixId] = {
        label: newLabel,
        tags: newTags.split(',').map(t => t.trim()).filter(t => t),
        folder: selectedFolder,
        updated: new Date().toISOString()
      };
      localStorage.setItem('matrixMetadata', JSON.stringify(existingMetadata));
      
      await loadDatabaseState(); // Refresh to show changes
      setEditingMatrix(null);
      setNewLabel('');
      setNewTags('');
    } catch (error) {
      console.error('Failed to save matrix edits:', error);
    }
  };

  const moveToFolder = async (matrixId, folder) => {
    try {
      const existingMetadata = JSON.parse(localStorage.getItem('matrixMetadata') || '{}');
      if (!existingMetadata[matrixId]) {
        existingMetadata[matrixId] = { tags: [] };
      }
      existingMetadata[matrixId].folder = folder;
      existingMetadata[matrixId].updated = new Date().toISOString();
      localStorage.setItem('matrixMetadata', JSON.stringify(existingMetadata));
      
      await loadDatabaseState();
    } catch (error) {
      console.error('Failed to move matrix to folder:', error);
    }
  };

  const addFolder = () => {
    if (newFolderName.trim() && !folders.includes(newFolderName.trim())) {
      const updatedFolders = [...folders, newFolderName.trim()];
      setFolders(updatedFolders);
      localStorage.setItem('matrixFolders', JSON.stringify(updatedFolders));
      setNewFolderName('');
    }
  };

  const bulkDeleteMatrices = async () => {
    if (selectedMatrices.size === 0) return;
    if (!confirm(`Delete ${selectedMatrices.size} selected matrices? This cannot be undone.`)) return;
    
    try {
      for (const matrixId of selectedMatrices) {
        await safeFetch(`/rho/${matrixId}`, { method: 'DELETE' });
      }
      setSelectedMatrices(new Set());
      await loadDatabaseState();
    } catch (error) {
      console.error('Failed to bulk delete matrices:', error);
    }
  };

  const bulkMoveToFolder = async (folder) => {
    if (selectedMatrices.size === 0) return;
    
    try {
      const existingMetadata = JSON.parse(localStorage.getItem('matrixMetadata') || '{}');
      for (const matrixId of selectedMatrices) {
        if (!existingMetadata[matrixId]) {
          existingMetadata[matrixId] = { tags: [] };
        }
        existingMetadata[matrixId].folder = folder;
        existingMetadata[matrixId].updated = new Date().toISOString();
      }
      localStorage.setItem('matrixMetadata', JSON.stringify(existingMetadata));
      setSelectedMatrices(new Set());
      await loadDatabaseState();
    } catch (error) {
      console.error('Failed to bulk move matrices:', error);
    }
  };

  const toggleMatrixSelection = (matrixId) => {
    const newSelection = new Set(selectedMatrices);
    if (newSelection.has(matrixId)) {
      newSelection.delete(matrixId);
    } else {
      newSelection.add(matrixId);
    }
    setSelectedMatrices(newSelection);
  };

  // Load metadata from localStorage
  const getMatrixMetadata = (matrixId) => {
    const metadata = JSON.parse(localStorage.getItem('matrixMetadata') || '{}');
    return metadata[matrixId] || { tags: [], folder: 'default' };
  };

  useEffect(() => {
    loadDatabaseState();
    
    // Load saved folders
    const savedFolders = JSON.parse(localStorage.getItem('matrixFolders') || '[]');
    if (savedFolders.length > 0) {
      setFolders(savedFolders);
    }
    
    const interval = setInterval(loadDatabaseState, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div style={{ padding: 20 }}>Loading database state...</div>;

  const processedMatrices = getProcessedMatrices();

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2>Database Overview ({processedMatrices.length}/{dbState?.matrices?.length || 0} matrices)</h2>
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={loadDatabaseState}
            style={{ padding: '8px 16px', background: '#007bff', color: 'white', border: 'none', borderRadius: 4 }}
          >
            🔄 Refresh
          </button>
          <button
            onClick={clearDatabase}
            style={{ padding: '8px 16px', background: '#dc3545', color: 'white', border: 'none', borderRadius: 4 }}
          >
            🗑️ Clear All
          </button>
        </div>
      </div>

      {/* Folders and Bulk Actions */}
      <div style={{ 
        marginBottom: 20, 
        padding: 15, 
        backgroundColor: '#e8f5e9', 
        borderRadius: 8,
        border: '1px solid #4caf50'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 }}>
          <h3 style={{ margin: 0, fontSize: 16, color: '#2e7d32' }}>📁 Matrix Organization</h3>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => setShowBulkActions(!showBulkActions)}
              style={{
                padding: '6px 12px',
                backgroundColor: showBulkActions ? '#ff9800' : '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                fontSize: 12
              }}
            >
              {showBulkActions ? 'Hide Bulk Actions' : 'Show Bulk Actions'}
            </button>
            {selectedMatrices.size > 0 && (
              <span style={{ 
                padding: '6px 12px',
                backgroundColor: '#9c27b0',
                color: 'white',
                borderRadius: 4,
                fontSize: 12
              }}>
                {selectedMatrices.size} selected
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 15 }}>
          {/* Current Folder */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              📂 Current Folder
            </label>
            <select
              value={selectedFolder}
              onChange={(e) => setSelectedFolder(e.target.value)}
              style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value="all">All Folders</option>
              {folders.map(folder => (
                <option key={folder} value={folder}>
                  {folder} {folder === 'default' ? '(Default)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Add New Folder */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              ➕ Add Folder
            </label>
            <div style={{ display: 'flex', gap: 5 }}>
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="New folder name"
                style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
              />
              <button
                onClick={addFolder}
                disabled={!newFolderName.trim()}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#4caf50',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  fontSize: 12
                }}
              >
                Add
              </button>
            </div>
          </div>

          {/* Bulk Actions */}
          {showBulkActions && selectedMatrices.size > 0 && (
            <>
              <div>
                <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
                  🚚 Move Selected To
                </label>
                <select
                  onChange={(e) => e.target.value && bulkMoveToFolder(e.target.value)}
                  style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">-- Select folder --</option>
                  {folders.map(folder => (
                    <option key={folder} value={folder}>{folder}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
                  🗑️ Bulk Actions
                </label>
                <button
                  onClick={bulkDeleteMatrices}
                  style={{
                    width: '100%',
                    padding: '6px 10px',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    fontSize: 12
                  }}
                >
                  Delete Selected ({selectedMatrices.size})
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Filters and Controls */}
      <div style={{ 
        marginBottom: 20, 
        padding: 15, 
        backgroundColor: '#f8f9fa', 
        borderRadius: 8,
        border: '1px solid #dee2e6'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 15, alignItems: 'end' }}>
          {/* Search */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              🔍 Search
            </label>
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Search by ID, label, or tags..."
              style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
            />
          </div>

          {/* Category Filter */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              📂 Category
            </label>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value="all">All Matrices</option>
              <option value="recent">Recent Activity</option>
              <option value="high_quality">High Quality (ρ > 0.1)</option>
              <option value="complex">Complex (H > 2.0)</option>
              <option value="tagged">Tagged Only</option>
            </select>
          </div>

          {/* Sort By */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              📊 Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value="created">Creation Order</option>
              <option value="purity">Purity</option>
              <option value="entropy">Entropy</option>
              <option value="operations">Operations</option>
              <option value="narratives">Narratives</option>
            </select>
          </div>

          {/* Sort Order */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              🔃 Order
            </label>
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              style={{
                width: '100%',
                padding: '6px 10px',
                backgroundColor: sortOrder === 'desc' ? '#28a745' : '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer'
              }}
            >
              {sortOrder === 'desc' ? '↓ Descending' : '↑ Ascending'}
            </button>
          </div>

          {/* View Mode */}
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontSize: 12, fontWeight: 'bold' }}>
              👁️ View
            </label>
            <div style={{ display: 'flex', gap: 5 }}>
              {['grid', 'list'].map(mode => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  style={{
                    flex: 1,
                    padding: '6px 8px',
                    backgroundColor: viewMode === mode ? '#007bff' : '#e9ecef',
                    color: viewMode === mode ? 'white' : '#6c757d',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                    fontSize: 11
                  }}
                >
                  {mode === 'grid' ? '⊞' : '☰'} {mode}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {dbState && (
        <div style={{ display: 'grid', gridTemplateColumns: selectedMatrix ? '1fr 1fr' : '1fr', gap: 20 }}>
          {/* Matrix List */}
          <div>
            <h3>Quantum States ({processedMatrices.length})</h3>
            
            {processedMatrices.length === 0 ? (
              <div style={{ padding: 40, textAlign: 'center', color: '#666', border: '1px solid #ddd', borderRadius: 4 }}>
                {searchFilter || categoryFilter !== 'all' ? (
                  <>
                    <div style={{ fontSize: 16, marginBottom: 10 }}>🔍 No matrices match your filters</div>
                    <div style={{ fontSize: 14 }}>Try adjusting your search or category filters.</div>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 16, marginBottom: 10 }}>📊 No matrices found</div>
                    <div style={{ fontSize: 14 }}>Create one in the Narrative Distillation Studio.</div>
                  </>
                )}
              </div>
            ) : (
              <div style={{ 
                maxHeight: 600, 
                overflowY: 'auto', 
                border: '1px solid #ddd', 
                borderRadius: 4,
                display: viewMode === 'grid' ? 'grid' : 'block',
                gridTemplateColumns: viewMode === 'grid' ? 'repeat(auto-fill, minmax(280px, 1fr))' : '1fr',
                gap: viewMode === 'grid' ? 10 : 0
              }}>
                {processedMatrices.map((matrix, idx) => (
                  <div
                    key={matrix.rho_id || idx}
                    style={{
                      padding: viewMode === 'grid' ? 12 : 15,
                      border: viewMode === 'grid' ? '1px solid #e9ecef' : 'none',
                      borderBottom: viewMode === 'list' ? '1px solid #eee' : 'none',
                      borderRadius: viewMode === 'grid' ? 6 : 0,
                      cursor: 'pointer',
                      background: selectedMatrix?.rho_id === matrix.rho_id ? '#e3f2fd' : 'white',
                      transition: 'all 0.2s',
                      margin: viewMode === 'grid' ? 5 : 0
                    }}
                    onClick={() => loadMatrixDetails(matrix.rho_id)}
                    onMouseOver={(e) => {
                      if (selectedMatrix?.rho_id !== matrix.rho_id) {
                        e.target.style.backgroundColor = '#f8f9fa';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (selectedMatrix?.rho_id !== matrix.rho_id) {
                        e.target.style.backgroundColor = 'white';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                      <div style={{ fontWeight: 600, fontSize: viewMode === 'grid' ? 13 : 14 }}>
                        {matrix.label || `Matrix ${matrix.rho_id?.slice(0, 8)}...`}
                      </div>
                      {viewMode === 'grid' && (
                        <span style={{ 
                          fontSize: 10, 
                          backgroundColor: '#6c757d', 
                          color: 'white', 
                          padding: '2px 6px', 
                          borderRadius: 3 
                        }}>
                          #{idx + 1}
                        </span>
                      )}
                    </div>
                    
                    <div style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>
                      ID: {matrix.rho_id?.slice(0, 12)}...
                    </div>
                    
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: viewMode === 'grid' ? '1fr 1fr' : 'repeat(4, 1fr)', 
                      gap: 8, 
                      fontSize: 11, 
                      marginBottom: 10 
                    }}>
                      <div style={{ color: '#007bff' }}>
                        <strong>ρ:</strong> {formatNumber(matrix.purity, 3)}
                      </div>
                      <div style={{ color: '#dc3545' }}>
                        <strong>H:</strong> {formatNumber(matrix.entropy, 2)}
                      </div>
                      {viewMode === 'list' && (
                        <>
                          <div style={{ color: '#28a745' }}>
                            <strong>Ops:</strong> {matrix.operations_count || 0}
                          </div>
                          <div style={{ color: '#6f42c1' }}>
                            <strong>Texts:</strong> {matrix.narratives_count || 0}
                          </div>
                        </>
                      )}
                    </div>
                    
                    {matrix.eigenvals && matrix.eigenvals.length > 0 && (
                      <div style={{ fontSize: 10, color: '#495057', marginBottom: 8 }}>
                        λ₁: {formatNumber(matrix.eigenvals[0], 3)} | λ₂: {formatNumber(matrix.eigenvals[1], 3)}
                      </div>
                    )}
                    
                    <div style={{ display: 'flex', gap: 5, justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ fontSize: 10, color: '#999' }}>
                        {viewMode === 'grid' ? `${matrix.operations_count || 0} ops` : ''}
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMatrix(matrix.rho_id);
                        }}
                        style={{
                          padding: '3px 6px',
                          background: '#dc3545',
                          color: 'white',
                          border: 'none',
                          borderRadius: 3,
                          fontSize: 10,
                          cursor: 'pointer'
                        }}
                        title="Delete matrix"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Matrix Details */}
          {selectedMatrix && (
            <div>
              <h3>Matrix Details</h3>
              <div style={{ border: '1px solid #ddd', borderRadius: 4, padding: 15 }}>
                <div style={{ marginBottom: 15 }}>
                  <strong>Label:</strong> {selectedMatrix.label || 'Untitled Matrix'}
                </div>
                <div style={{ marginBottom: 15 }}>
                  <strong>ID:</strong> {selectedMatrix.rho_id}
                </div>
                <div style={{ marginBottom: 15 }}>
                  <strong>Dimension:</strong> {selectedMatrix.dimension || 64}
                </div>
                {selectedMatrix.purity !== undefined && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Purity:</strong> {formatNumber(selectedMatrix.purity)}
                  </div>
                )}
                {selectedMatrix.entropy !== undefined && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Entropy:</strong> {formatNumber(selectedMatrix.entropy)}
                  </div>
                )}
                {selectedMatrix.created_at && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Created:</strong> {new Date(selectedMatrix.created_at).toLocaleString()}
                  </div>
                )}
                {selectedMatrix.narratives && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Narratives:</strong> {selectedMatrix.narratives.length}
                  </div>
                )}
                {selectedMatrix.operations_count !== undefined && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Operations:</strong> {selectedMatrix.operations_count}
                  </div>
                )}

                {/* Eigenvalues Display */}
                {selectedMatrix.eigenvalues && selectedMatrix.eigenvalues.length > 0 && (
                  <div style={{ marginBottom: 15 }}>
                    <strong>Top Eigenvalues:</strong>
                    <div style={{ fontSize: 12, fontFamily: 'monospace', marginTop: 5 }}>
                      {selectedMatrix.eigenvalues.slice(0, 10).map((val, idx) => (
                        <div key={idx}>
                          λ{idx + 1}: {formatNumber(val)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recent Operations */}
                {selectedMatrix.recent_operations && selectedMatrix.recent_operations.length > 0 && (
                  <div>
                    <strong>Recent Operations:</strong>
                    <div style={{ maxHeight: 200, overflowY: 'auto', marginTop: 10 }}>
                      {selectedMatrix.recent_operations.map((op, idx) => (
                        <div key={idx} style={{ 
                          padding: 8,
                          border: '1px solid #eee',
                          borderRadius: 3,
                          marginBottom: 5,
                          fontSize: 12
                        }}>
                          <div><strong>{op.type || 'Operation'}</strong></div>
                          {op.timestamp && (
                            <div style={{ color: '#666' }}>
                              {new Date(op.timestamp).toLocaleString()}
                            </div>
                          )}
                          {op.details && (
                            <div style={{ color: '#666' }}>{op.details}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}