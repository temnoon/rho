import React from 'react';

/**
 * QuantumStateCard - Reusable component for displaying quantum state information
 * 
 * Extracted from NarrativeExplorer and NarrativeDistillationStudio patterns
 */
export function QuantumStateCard({ 
  rhoId, 
  diagnostics, 
  label = "Quantum State",
  status = "active", // active, pending, completed, error
  showActions = true,
  onAnalyze = null,
  onVisualize = null,
  onExport = null,
  style = {}
}) {
  const getStatusColor = () => {
    switch (status) {
      case 'active': return '#4CAF50';
      case 'pending': return '#FF9800';
      case 'completed': return '#2196F3';
      case 'error': return '#f44336';
      default: return '#9E9E9E';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'active': return '✅';
      case 'pending': return '⏳';
      case 'completed': return '🎯';
      case 'error': return '❌';
      default: return '⚪';
    }
  };

  return (
    <div style={{
      background: '#e8f5e9',
      border: `2px solid ${getStatusColor()}`,
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px',
      ...style
    }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
        <span style={{ fontSize: '24px', marginRight: '10px' }}>{getStatusIcon()}</span>
        <h3 style={{ margin: 0, color: '#2e7d32' }}>
          {label}
        </h3>
      </div>
      
      {rhoId && (
        <p style={{ color: '#333', marginBottom: '10px' }}>
          Matrix ID: <code style={{ 
            background: '#f5f5f5', 
            padding: '2px 6px', 
            borderRadius: '3px',
            fontFamily: 'monospace'
          }}>
            {rhoId}
          </code>
        </p>
      )}
      
      {diagnostics && (
        <div style={{ fontSize: '12px', color: '#666', marginBottom: '15px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
            {diagnostics.trace && (
              <div>
                <strong>Trace:</strong> {diagnostics.trace.toFixed(3)}
              </div>
            )}
            {diagnostics.purity && (
              <div>
                <strong>Purity:</strong> {diagnostics.purity.toFixed(3)}
              </div>
            )}
            {diagnostics.entropy && (
              <div>
                <strong>Entropy:</strong> {diagnostics.entropy.toFixed(2)}
              </div>
            )}
            {diagnostics.rank && (
              <div>
                <strong>Rank:</strong> {diagnostics.rank}
              </div>
            )}
          </div>
        </div>
      )}
      
      {showActions && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {onAnalyze && (
            <button
              onClick={onAnalyze}
              style={{
                padding: '8px 16px',
                background: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              🔬 Analyze
            </button>
          )}
          {onVisualize && (
            <button
              onClick={onVisualize}
              style={{
                padding: '8px 16px',
                background: '#9C27B0',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              📊 Visualize
            </button>
          )}
          {onExport && (
            <button
              onClick={onExport}
              style={{
                padding: '8px 16px',
                background: '#607D8B',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              💾 Export
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default QuantumStateCard;