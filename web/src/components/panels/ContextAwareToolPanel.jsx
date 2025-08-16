import React from 'react';
import { USER_MODES } from '../../utils/WorkflowState.js';
import { AgentMessage, InlineNotification } from '../common/index.js';

/**
 * Context-Aware Tool Panel
 * 
 * Provides contextual tools and guidance based on current workflow stage,
 * user mode, and system state. Adapts interface complexity to user expertise.
 */
export function ContextAwareToolPanel({ 
  stage,
  userMode,
  state,
  onAction,
  style = {}
}) {
  const mode = USER_MODES[userMode];
  
  // Get stage-specific tools and content
  const getStageContent = () => {
    switch (stage) {
      case 'input':
        return renderInputTools();
      case 'analysis':
        return renderAnalysisTools();
      case 'transform':
        return renderTransformTools();
      case 'visualize':
        return renderVisualizeTools();
      case 'export':
        return renderExportTools();
      default:
        return renderDefaultTools();
    }
  };

  // Input stage tools
  const renderInputTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#4CAF50' }}>
        📝 Input Assistance
      </h4>
      
      {mode.complexity >= 2 && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Import Options</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => onAction('import_file')}
              style={{
                padding: '8px 12px',
                background: '#f5f5f5',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer',
                textAlign: 'left'
              }}
            >
              📄 Import Text File
            </button>
            {mode.complexity >= 3 && (
              <button
                onClick={() => onAction('import_gutenberg')}
                style={{
                  padding: '8px 12px',
                  background: '#f5f5f5',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  textAlign: 'left'
                }}
              >
                📚 Browse Project Gutenberg
              </button>
            )}
          </div>
        </div>
      )}

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Text Analysis</h5>
        <div style={{ 
          padding: '10px',
          background: '#f8f9fa',
          borderRadius: '4px',
          fontSize: '12px',
          color: '#666'
        }}>
          <div>Characters: {state.narrativeText?.length || 0}</div>
          <div>Words: {state.narrativeText ? state.narrativeText.split(/\s+/).length : 0}</div>
          <div>Estimated reading time: {Math.ceil((state.narrativeText?.length || 0) / 200)} cycles</div>
        </div>
      </div>

      {mode.complexity >= 2 && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Quick Actions</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => onAction('sample_text')}
              style={{
                padding: '6px 10px',
                background: '#e3f2fd',
                border: '1px solid #2196F3',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
                color: '#1976D2'
              }}
            >
              Load Sample Text
            </button>
            <button
              onClick={() => onAction('clear_text')}
              disabled={!state.narrativeText}
              style={{
                padding: '6px 10px',
                background: state.narrativeText ? '#ffebee' : '#f5f5f5',
                border: '1px solid #f44336',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: state.narrativeText ? 'pointer' : 'not-allowed',
                color: state.narrativeText ? '#d32f2f' : '#999',
                opacity: state.narrativeText ? 1 : 0.5
              }}
            >
              Clear All Text
            </button>
          </div>
        </div>
      )}
    </div>
  );

  // Analysis stage tools
  const renderAnalysisTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#2196F3' }}>
        🔬 Analysis Tools
      </h4>

      {state.currentRhoId && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Quantum State</h5>
          <div style={{ 
            padding: '10px',
            background: '#e8f5e9',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#2e7d32'
          }}>
            <div><strong>Matrix ID:</strong> {state.currentRhoId.substring(0, 8)}...</div>
            {state.quantumDiagnostics && (
              <>
                <div><strong>Trace:</strong> {state.quantumDiagnostics.trace?.toFixed(3)}</div>
                <div><strong>Purity:</strong> {state.quantumDiagnostics.purity?.toFixed(3)}</div>
              </>
            )}
          </div>
        </div>
      )}

      {mode.complexity >= 2 && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Measurement Options</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => onAction('povm_comprehensive')}
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
              🎯 Comprehensive POVM
            </button>
            {mode.complexity >= 3 && (
              <button
                onClick={() => onAction('povm_custom')}
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
                ⚗️ Custom Measurements
              </button>
            )}
          </div>
        </div>
      )}

      {state.povmMeasurements && mode.complexity >= 3 && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Advanced Analysis</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => onAction('integrability_analysis')}
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
              🧮 Integrability Analysis
            </button>
            <button
              onClick={() => onAction('residue_analysis')}
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
              🔍 Residue Analysis
            </button>
          </div>
        </div>
      )}
    </div>
  );

  // Transform stage tools
  const renderTransformTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#FF9800' }}>
        ⚗️ Transform Tools
      </h4>

      <AgentMessage type="info" icon="🚧">
        Transform tools are being developed. Coming soon: invariant editing, sequence synthesis, and APLG operations.
      </AgentMessage>

      {mode.complexity >= 3 && (
        <div style={{ marginTop: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>APLG Operations</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => onAction('invariant_editor')}
              disabled={!state.extractedAttributes}
              style={{
                padding: '8px 12px',
                background: state.extractedAttributes ? '#fff3e0' : '#f5f5f5',
                border: '1px solid #FF9800',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: state.extractedAttributes ? 'pointer' : 'not-allowed',
                textAlign: 'left',
                opacity: state.extractedAttributes ? 1 : 0.5
              }}
            >
              🎭 Invariant Editor (Claim D)
            </button>
            <button
              onClick={() => onAction('sequence_synthesizer')}
              disabled={!state.extractedAttributes}
              style={{
                padding: '8px 12px',
                background: state.extractedAttributes ? '#f3e5f5' : '#f5f5f5',
                border: '1px solid #9C27B0',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: state.extractedAttributes ? 'pointer' : 'not-allowed',
                textAlign: 'left',
                opacity: state.extractedAttributes ? 1 : 0.5
              }}
            >
              📚 Sequence Synthesizer (Claim E)
            </button>
          </div>
        </div>
      )}
    </div>
  );

  // Visualize stage tools
  const renderVisualizeTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#9C27B0' }}>
        📊 Visualization Tools
      </h4>

      <AgentMessage type="info" icon="🎨">
        Visualization tools are in development. Soon you'll see Bures trajectories, eigenvalue flows, and entropy landscapes.
      </AgentMessage>

      {mode.complexity >= 2 && (
        <div style={{ marginTop: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Planned Visualizations</h5>
          <div style={{ 
            padding: '10px',
            background: '#f8f9fa',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#666'
          }}>
            <div>📈 Bures Trajectory Paths</div>
            <div>🌊 Eigenvalue River Flows</div>
            <div>🗻 Entropy Landscape Maps</div>
            <div>⚛️ Quantum State Explorer</div>
          </div>
        </div>
      )}
    </div>
  );

  // Export stage tools
  const renderExportTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#607D8B' }}>
        💾 Export Tools
      </h4>

      <div style={{ marginBottom: '15px' }}>
        <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Export Formats</h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <button
            onClick={() => onAction('export_json')}
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
            📄 JSON Export
          </button>
          {mode.complexity >= 2 && (
            <>
              <button
                onClick={() => onAction('export_csv')}
                disabled={!state.povmMeasurements}
                style={{
                  padding: '8px 12px',
                  background: state.povmMeasurements ? '#e8f5e9' : '#f5f5f5',
                  border: '1px solid #4CAF50',
                  borderRadius: '4px',
                  fontSize: '11px',
                  cursor: state.povmMeasurements ? 'pointer' : 'not-allowed',
                  textAlign: 'left',
                  opacity: state.povmMeasurements ? 1 : 0.5
                }}
              >
                📊 CSV Data Export
              </button>
              <button
                onClick={() => onAction('generate_report')}
                disabled={!state.extractedAttributes}
                style={{
                  padding: '8px 12px',
                  background: state.extractedAttributes ? '#fff3e0' : '#f5f5f5',
                  border: '1px solid #FF9800',
                  borderRadius: '4px',
                  fontSize: '11px',
                  cursor: state.extractedAttributes ? 'pointer' : 'not-allowed',
                  textAlign: 'left',
                  opacity: state.extractedAttributes ? 1 : 0.5
                }}
              >
                📝 Analysis Report
              </button>
            </>
          )}
        </div>
      </div>

      {mode.complexity >= 3 && (
        <div style={{ marginBottom: '15px' }}>
          <h5 style={{ margin: '0 0 8px 0', fontSize: '13px' }}>Archive Options</h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => onAction('save_to_library')}
              disabled={!state.currentRhoId}
              style={{
                padding: '6px 10px',
                background: state.currentRhoId ? '#f3e5f5' : '#f5f5f5',
                border: '1px solid #9C27B0',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: state.currentRhoId ? 'pointer' : 'not-allowed',
                color: state.currentRhoId ? '#7B1FA2' : '#999',
                opacity: state.currentRhoId ? 1 : 0.5
              }}
            >
              🏛️ Save to Matrix Library
            </button>
          </div>
        </div>
      )}
    </div>
  );

  // Default tools when no specific stage
  const renderDefaultTools = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#666' }}>
        🔧 General Tools
      </h4>
      <InlineNotification
        message="Select a workflow stage to see contextual tools and guidance."
        type="info"
        showIcon={true}
      />
    </div>
  );

  return (
    <div style={{
      padding: '20px',
      background: '#f8f9fa',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      ...style
    }}>
      {getStageContent()}
      
      {/* Mode indicator */}
      <div style={{
        marginTop: '20px',
        padding: '8px',
        background: 'rgba(33,150,243,0.1)',
        borderRadius: '4px',
        fontSize: '11px',
        color: '#1976D2',
        textAlign: 'center'
      }}>
        {mode.icon} {mode.label} Mode • Complexity Level {mode.complexity}
      </div>
    </div>
  );
}

export default ContextAwareToolPanel;