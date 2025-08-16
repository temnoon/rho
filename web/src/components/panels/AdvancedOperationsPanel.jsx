import React, { useState } from 'react';
import { AgentMessage, InlineNotification } from '../common/index.js';

/**
 * Advanced Operations Panel
 * 
 * Provides access to expert-level quantum operations, matrix archaeology,
 * and research tools. Only visible when advanced sidebar is open.
 */
export function AdvancedOperationsPanel({ 
  state,
  onAction,
  style = {}
}) {
  const [activeSection, setActiveSection] = useState('matrix_operations');

  const sections = [
    { id: 'matrix_operations', label: 'Matrix Operations', icon: '🔮' },
    { id: 'archaeology', label: 'Matrix Archaeology', icon: '🏛️' },
    { id: 'research', label: 'Research Tools', icon: '🔬' },
    { id: 'diagnostics', label: 'System Diagnostics', icon: '⚙️' }
  ];

  const renderMatrixOperations = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#9C27B0' }}>
        🔮 Quantum Matrix Operations
      </h4>

      <div style={{ marginBottom: '20px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Current Matrix</h5>
        {state.currentRhoId ? (
          <div style={{
            padding: '10px',
            background: '#e8f5e9',
            borderRadius: '4px',
            fontSize: '12px',
            marginBottom: '10px'
          }}>
            <div><strong>ID:</strong> {state.currentRhoId}</div>
            {state.quantumDiagnostics && (
              <>
                <div><strong>Trace:</strong> {state.quantumDiagnostics.trace?.toFixed(4)}</div>
                <div><strong>Purity:</strong> {state.quantumDiagnostics.purity?.toFixed(4)}</div>
                <div><strong>Entropy:</strong> {state.quantumDiagnostics.entropy?.toFixed(4)}</div>
              </>
            )}
          </div>
        ) : (
          <InlineNotification
            message="No active quantum matrix"
            type="info"
            showIcon={false}
          />
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => onAction('matrix_spectral_analysis')}
            disabled={!state.currentRhoId}
            style={{
              padding: '8px 12px',
              background: state.currentRhoId ? '#f3e5f5' : '#f5f5f5',
              border: '1px solid #9C27B0',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: state.currentRhoId ? 'pointer' : 'not-allowed',
              textAlign: 'left',
              opacity: state.currentRhoId ? 1 : 0.5
            }}
          >
            📊 Spectral Decomposition
          </button>
          <button
            onClick={() => onAction('matrix_purification')}
            disabled={!state.currentRhoId}
            style={{
              padding: '8px 12px',
              background: state.currentRhoId ? '#fff3e0' : '#f5f5f5',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: state.currentRhoId ? 'pointer' : 'not-allowed',
              textAlign: 'left',
              opacity: state.currentRhoId ? 1 : 0.5
            }}
          >
            ✨ Matrix Purification
          </button>
          <button
            onClick={() => onAction('channel_tomography')}
            disabled={!state.currentRhoId}
            style={{
              padding: '8px 12px',
              background: state.currentRhoId ? '#e3f2fd' : '#f5f5f5',
              border: '1px solid #2196F3',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: state.currentRhoId ? 'pointer' : 'not-allowed',
              textAlign: 'left',
              opacity: state.currentRhoId ? 1 : 0.5
            }}
          >
            🔍 Channel Tomography
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Matrix Synthesis</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <button
            onClick={() => onAction('create_random_matrix')}
            style={{
              padding: '6px 10px',
              background: '#e8f5e9',
              border: '1px solid #4CAF50',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#2e7d32'
            }}
          >
            🎲 Random Density Matrix
          </button>
          <button
            onClick={() => onAction('create_pure_state')}
            style={{
              padding: '6px 10px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#e65100'
            }}
          >
            💎 Pure State Generator
          </button>
        </div>
      </div>
    </div>
  );

  const renderArchaeology = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#795548' }}>
        🏛️ Matrix Archaeology
      </h4>

      <AgentMessage type="info" icon="🔍">
        Matrix archaeology helps you discover patterns and relationships in your collection of quantum states.
      </AgentMessage>

      <div style={{ marginTop: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Library Management</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => onAction('browse_matrix_library')}
            style={{
              padding: '8px 12px',
              background: '#f3e5f5',
              border: '1px solid #9C27B0',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            📚 Browse Matrix Library
          </button>
          <button
            onClick={() => onAction('similarity_analysis')}
            style={{
              padding: '8px 12px',
              background: '#e3f2fd',
              border: '1px solid #2196F3',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            🔗 Similarity Analysis
          </button>
          <button
            onClick={() => onAction('cluster_matrices')}
            style={{
              padding: '8px 12px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            🌐 Cluster Analysis
          </button>
        </div>
      </div>

      <div style={{ marginTop: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Discovery Tools</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <button
            onClick={() => onAction('find_best_work')}
            style={{
              padding: '6px 10px',
              background: '#fff8e1',
              border: '1px solid #FFC107',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#f57c00'
            }}
          >
            🏆 Find Best Work
          </button>
          <button
            onClick={() => onAction('synthesis_recommendations')}
            style={{
              padding: '6px 10px',
              background: '#e8f5e9',
              border: '1px solid #4CAF50',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#2e7d32'
            }}
          >
            💡 Synthesis Recommendations
          </button>
        </div>
      </div>
    </div>
  );

  const renderResearchTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#FF5722' }}>
        🔬 Research Tools
      </h4>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Experimental Features</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => onAction('custom_povm_designer')}
            style={{
              padding: '8px 12px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            ⚗️ Custom POVM Designer
          </button>
          <button
            onClick={() => onAction('channel_observatory')}
            style={{
              padding: '8px 12px',
              background: '#e3f2fd',
              border: '1px solid #2196F3',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            🔭 Channel Observatory
          </button>
          <button
            onClick={() => onAction('batch_processor')}
            style={{
              padding: '8px 12px',
              background: '#f3e5f5',
              border: '1px solid #9C27B0',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            ⚡ Batch Processor
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Research Notebooks</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <button
            onClick={() => onAction('new_research_notebook')}
            style={{
              padding: '6px 10px',
              background: '#e8f5e9',
              border: '1px solid #4CAF50',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#2e7d32'
            }}
          >
            📔 New Research Notebook
          </button>
          <button
            onClick={() => onAction('experiment_templates')}
            style={{
              padding: '6px 10px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              color: '#e65100'
            }}
          >
            📋 Experiment Templates
          </button>
        </div>
      </div>
    </div>
  );

  const renderDiagnostics = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#607D8B' }}>
        ⚙️ System Diagnostics
      </h4>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Session Information</h5>
        <div style={{
          padding: '10px',
          background: '#f8f9fa',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#666'
        }}>
          <div><strong>Session ID:</strong> {state.sessionId?.substring(0, 12)}...</div>
          <div><strong>Last Updated:</strong> {state.lastUpdated ? new Date(state.lastUpdated).toLocaleTimeString() : 'Never'}</div>
          <div><strong>Auto-save:</strong> {state.autoSaveEnabled ? 'Enabled' : 'Disabled'}</div>
          <div><strong>Version:</strong> {state.version || '1.0.0'}</div>
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>System Operations</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            onClick={() => onAction('export_session_state')}
            style={{
              padding: '8px 12px',
              background: '#e3f2fd',
              border: '1px solid #2196F3',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            💾 Export Session State
          </button>
          <button
            onClick={() => onAction('import_session_state')}
            style={{
              padding: '8px 12px',
              background: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left'
            }}
          >
            📂 Import Session State
          </button>
          <button
            onClick={() => onAction('clear_session_cache')}
            style={{
              padding: '8px 12px',
              background: '#ffebee',
              border: '1px solid #f44336',
              borderRadius: '4px',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left',
              color: '#d32f2f'
            }}
          >
            🗑️ Clear Session Cache
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 10px 0', fontSize: '13px' }}>Health Check</h5>
        <button
          onClick={() => onAction('system_health_check')}
          style={{
            padding: '6px 10px',
            background: '#e8f5e9',
            border: '1px solid #4CAF50',
            borderRadius: '4px',
            fontSize: '11px',
            cursor: 'pointer',
            color: '#2e7d32',
            width: '100%'
          }}
        >
          🏥 Run System Health Check
        </button>
      </div>
    </div>
  );

  const renderSection = () => {
    switch (activeSection) {
      case 'matrix_operations':
        return renderMatrixOperations();
      case 'archaeology':
        return renderArchaeology();
      case 'research':
        return renderResearchTools();
      case 'diagnostics':
        return renderDiagnostics();
      default:
        return renderMatrixOperations();
    }
  };

  return (
    <div style={{
      background: '#f8f9fa',
      borderLeft: '2px solid #e0e0e0',
      padding: '20px',
      height: '100%',
      overflowY: 'auto',
      ...style
    }}>
      <h3 style={{ marginTop: 0, color: '#333', marginBottom: '20px' }}>
        🏛️ Advanced Operations
      </h3>

      {/* Section tabs */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        marginBottom: '20px'
      }}>
        {sections.map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            style={{
              padding: '8px 12px',
              background: activeSection === section.id ? '#2196F3' : 'white',
              color: activeSection === section.id ? 'white' : '#333',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease'
            }}
          >
            {section.icon} {section.label}
          </button>
        ))}
      </div>

      {/* Active section content */}
      {renderSection()}

      {/* Footer */}
      <div style={{
        marginTop: '20px',
        padding: '10px',
        background: 'rgba(33,150,243,0.1)',
        borderRadius: '4px',
        fontSize: '10px',
        color: '#1976D2',
        textAlign: 'center'
      }}>
        Expert tools for quantum narrative research
      </div>
    </div>
  );
}

export default AdvancedOperationsPanel;