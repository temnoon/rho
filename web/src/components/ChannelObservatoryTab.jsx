import React, { useState, useEffect, useCallback } from 'react';
import { ChannelObservatory } from './ChannelObservatory.jsx';
import { apiUrl } from '../utils/api.js';

/**
 * Channel Observatory Tab - Integration wrapper for the main workbench
 * 
 * Provides matrix selection and integrates the Channel Observatory
 * with the overall application state management.
 */
export function ChannelObservatoryTab() {
  const [availableMatrices, setAvailableMatrices] = useState([]);
  const [selectedMatrix, setSelectedMatrix] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch available matrices
  const fetchAvailableMatrices = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(apiUrl('/rho/global/status'));
      if (response.ok) {
        const data = await response.json();
        setAvailableMatrices(data.matrices || []);
        
        // Auto-select first matrix if none selected
        if (!selectedMatrix && data.matrices.length > 0) {
          setSelectedMatrix(data.matrices[0].rho_id);
        }
      }
    } catch (error) {
      console.error('Failed to fetch available matrices:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedMatrix]);

  // Load matrices on component mount
  useEffect(() => {
    fetchAvailableMatrices();
  }, [fetchAvailableMatrices]);

  // Handle channel operations
  const handleChannelApplied = useCallback((operation, result) => {
    console.log(`Channel operation ${operation} completed:`, result);
    // Refresh matrices list after channel operations
    fetchAvailableMatrices();
  }, [fetchAvailableMatrices]);

  // Render matrix selector
  const renderMatrixSelector = () => (
    <div style={{ 
      marginBottom: 20,
      padding: 15,
      border: '1px solid #ddd',
      borderRadius: 8,
      backgroundColor: '#f8f9fa'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Select Quantum State for Monitoring</h4>
      
      {loading ? (
        <div style={{ fontSize: 12, color: '#666' }}>Loading matrices...</div>
      ) : availableMatrices.length === 0 ? (
        <div style={{ fontSize: 12, color: '#666' }}>
          No quantum states available. Create a matrix first using the Book Reader or Live Narrative tabs.
        </div>
      ) : (
        <div>
          <select
            value={selectedMatrix || ''}
            onChange={(e) => setSelectedMatrix(e.target.value)}
            style={{
              width: '100%',
              padding: 8,
              border: '1px solid #ddd',
              borderRadius: 4,
              fontSize: 12,
              marginBottom: 10
            }}
          >
            <option value="">-- Select a matrix --</option>
            {availableMatrices.map(matrix => (
              <option key={matrix.rho_id} value={matrix.rho_id}>
                {matrix.label} ({matrix.narratives_count} narratives, {matrix.operations_count} ops)
              </option>
            ))}
          </select>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
            {availableMatrices.map(matrix => (
              <div 
                key={matrix.rho_id}
                onClick={() => setSelectedMatrix(matrix.rho_id)}
                style={{
                  padding: 10,
                  border: `2px solid ${selectedMatrix === matrix.rho_id ? '#2196F3' : '#ddd'}`,
                  borderRadius: 6,
                  cursor: 'pointer',
                  backgroundColor: selectedMatrix === matrix.rho_id ? '#e3f2fd' : 'white',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: 12, marginBottom: 4 }}>
                  {matrix.label}
                </div>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>
                  {matrix.narratives_count} narratives, {matrix.operations_count} operations
                </div>
                {matrix.dual_pair && (
                  <div style={{ fontSize: 10, color: '#9C27B0' }}>
                    Dual pair: {matrix.dual_pair.substring(0, 8)}...
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      <button
        onClick={fetchAvailableMatrices}
        style={{
          marginTop: 10,
          padding: '6px 12px',
          border: '1px solid #666',
          borderRadius: 4,
          backgroundColor: 'white',
          cursor: 'pointer',
          fontSize: 12
        }}
      >
        🔄 Refresh Matrices
      </button>
    </div>
  );

  return (
    <div style={{ padding: 20 }}>
      <div style={{ 
        marginBottom: 20,
        borderBottom: '2px solid #eee',
        paddingBottom: 15
      }}>
        <h2 style={{ margin: 0, fontSize: 20, color: '#333' }}>
          🔭 Channel Observatory
        </h2>
        <div style={{ fontSize: 14, color: '#666', marginTop: 5 }}>
          Monitor and analyze quantum channel operations in the Post-Lexical Grammatological Laboratory
        </div>
      </div>

      {renderMatrixSelector()}
      
      <ChannelObservatory 
        rhoId={selectedMatrix}
        onChannelApplied={handleChannelApplied}
      />
    </div>
  );
}

export default ChannelObservatoryTab;