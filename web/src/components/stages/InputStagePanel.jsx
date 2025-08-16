import React from 'react';
import { AgentMessage } from '../common/index.js';

/**
 * Input Stage Panel
 * 
 * Handles narrative text input and preparation for quantum processing
 */
export function InputStagePanel({ 
  state, 
  updateState, 
  progressToStage,
  createQuantumState,
  loading
}) {
  return (
    <div style={{ padding: '30px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>
        📝 Narrative Input
      </h2>
      <AgentMessage type="info">
        Begin your quantum narrative journey by entering text. This will be transformed 
        into a density matrix (ρ) representing the essential meaning structure.
      </AgentMessage>
      
      <div style={{ marginBottom: '20px' }}>
        <label style={{ 
          display: 'block', 
          marginBottom: '8px', 
          fontWeight: 600,
          color: '#333'
        }}>
          Your Narrative Text:
        </label>
        <textarea
          id="narrative-input"
          value={state.narrativeText}
          onChange={(e) => updateState({ narrativeText: e.target.value })}
          placeholder="Enter your narrative text here... It can be a story, description, dialogue, or any text with meaning to explore."
          style={{
            width: '100%',
            minHeight: '200px',
            padding: '15px',
            borderRadius: '8px',
            border: '2px solid #e0e0e0',
            fontSize: '14px',
            lineHeight: 1.6,
            fontFamily: 'system-ui, sans-serif',
            resize: 'vertical'
          }}
        />
        <div style={{ 
          fontSize: '12px', 
          color: '#666', 
          marginTop: '5px',
          display: 'flex',
          justifyContent: 'space-between'
        }}>
          <span>
            Characters: {state.narrativeText.length} | Words: {state.narrativeText.trim().split(/\s+/).filter(w => w.length > 0).length}
          </span>
          <span>Minimum 10 characters recommended</span>
        </div>
      </div>

      {state.narrativeText.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={createQuantumState}
            disabled={loading || state.narrativeText.length < 10}
            style={{
              padding: '12px 24px',
              background: loading || state.narrativeText.length < 10 ? '#ccc' : '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: loading || state.narrativeText.length < 10 ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? '⚙️ Creating Quantum State...' : '⚛️ Create Quantum State'}
          </button>
        </div>
      )}

      {state.currentRhoId && (
        <>
          <AgentMessage type="success">
            ✨ Quantum state created successfully! Your text has been encoded into a 64-dimensional 
            density matrix. Ready for analysis.
          </AgentMessage>
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <button
              onClick={() => progressToStage('analysis')}
              style={{
                padding: '12px 24px',
                background: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Continue to Analysis →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default InputStagePanel;