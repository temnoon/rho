import React, { useState, useCallback } from 'react';
import { AgentMessage, InlineNotification } from '../common/index.js';

/**
 * APLG Operation Center
 * 
 * Central hub for Analytic Post-Lexical Grammatology operations.
 * Implements the theoretical framework for quantum narrative transformations.
 * 
 * APLG Claims:
 * A. Lexical Projection: Meaning exists independently of surface form
 * B. Quantum Superposition: Narratives exist in multiple interpretive states
 * C. Measurement Collapse: POVM operations extract definite attributes
 * D. Invariant Editing: Preserve essential meaning while changing form
 * E. Curriculum Synthesis: Compose sequences for optimal learning
 * F. Bures Preservation: Maintain narrative distance under transformation
 * G. Consent Gating: Risk assessment and ethical boundaries
 */

export function APLGOperationCenter({ 
  state,
  onOperation,
  style = {}
}) {
  const [activeOperation, setActiveOperation] = useState(null);
  const [operationParams, setOperationParams] = useState({});

  // APLG Claim definitions
  const aplgClaims = {
    lexical_projection: {
      id: 'lexical_projection',
      label: 'Lexical Projection (Claim A)',
      description: 'Project between surface lexical forms while preserving deep meaning structure',
      icon: '🎭',
      color: '#4CAF50',
      complexity: 2,
      available: true
    },
    quantum_superposition: {
      id: 'quantum_superposition',
      label: 'Quantum Superposition (Claim B)',
      description: 'Maintain multiple interpretive states in superposition until measurement',
      icon: '⚛️',
      color: '#2196F3',
      complexity: 3,
      available: true
    },
    measurement_collapse: {
      id: 'measurement_collapse',
      label: 'Measurement Collapse (Claim C)',
      description: 'Apply POVM measurements to collapse superposition into definite attributes',
      icon: '🔬',
      color: '#FF9800',
      complexity: 2,
      available: true
    },
    invariant_editing: {
      id: 'invariant_editing',
      label: 'Invariant Editing (Claim D)',
      description: 'Edit text while preserving essential narrative invariants',
      icon: '✏️',
      color: '#9C27B0',
      complexity: 4,
      available: false // Coming soon
    },
    curriculum_synthesis: {
      id: 'curriculum_synthesis',
      label: 'Curriculum Synthesis (Claim E)',
      description: 'Synthesize optimal learning sequences from quantum narrative states',
      icon: '📚',
      color: '#FF5722',
      complexity: 4,
      available: false // Coming soon
    },
    bures_preservation: {
      id: 'bures_preservation',
      label: 'Bures Preservation (Claim F)',
      description: 'Maintain Bures distance under narrative transformations',
      icon: '📏',
      color: '#607D8B',
      complexity: 3,
      available: false // Coming soon
    },
    consent_gating: {
      id: 'consent_gating',
      label: 'Consent Gating (Claim G)',
      description: 'Assess and gate operations based on ethical risk factors',
      icon: '🛡️',
      color: '#795548',
      complexity: 3,
      available: true
    }
  };

  // Handle operation selection
  const handleOperationSelect = useCallback((claimId) => {
    const claim = aplgClaims[claimId];
    if (!claim.available) {
      onOperation('show_notification', {
        message: `${claim.label} is coming soon in a future release`,
        type: 'info'
      });
      return;
    }

    setActiveOperation(claimId);
    setOperationParams({});
  }, [aplgClaims, onOperation]);

  // Execute operation
  const executeOperation = useCallback(() => {
    if (!activeOperation) return;

    const claim = aplgClaims[activeOperation];
    
    onOperation('execute_aplg_operation', {
      claim: activeOperation,
      parameters: operationParams,
      state: state
    });

    // Reset operation state
    setActiveOperation(null);
    setOperationParams({});
  }, [activeOperation, operationParams, state, onOperation, aplgClaims]);

  // Render operation selector
  const renderOperationSelector = () => (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#333' }}>
        🧠 APLG Framework Operations
      </h4>
      
      <AgentMessage type="info" icon="🔬">
        Analytic Post-Lexical Grammatology provides the theoretical foundation for quantum narrative transformations. 
        Select an operation to explore the deep structure of meaning.
      </AgentMessage>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
        gap: '12px',
        marginTop: '20px'
      }}>
        {Object.values(aplgClaims).map(claim => (
          <div
            key={claim.id}
            onClick={() => handleOperationSelect(claim.id)}
            style={{
              padding: '15px',
              border: `2px solid ${claim.available ? claim.color : '#ddd'}`,
              borderRadius: '8px',
              background: claim.available ? 'white' : '#f5f5f5',
              cursor: claim.available ? 'pointer' : 'not-allowed',
              opacity: claim.available ? 1 : 0.6,
              transition: 'all 0.2s ease',
              position: 'relative'
            }}
          >
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              marginBottom: '8px' 
            }}>
              <span style={{ fontSize: '20px', marginRight: '8px' }}>
                {claim.icon}
              </span>
              <h5 style={{ 
                margin: 0, 
                fontSize: '14px', 
                color: claim.available ? claim.color : '#999',
                fontWeight: 600
              }}>
                {claim.label}
              </h5>
            </div>
            
            <p style={{ 
              margin: 0, 
              fontSize: '12px', 
              color: '#666',
              lineHeight: 1.4
            }}>
              {claim.description}
            </p>

            <div style={{
              marginTop: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{
                fontSize: '10px',
                color: '#999',
                background: '#f0f0f0',
                padding: '2px 6px',
                borderRadius: '3px'
              }}>
                Complexity: {claim.complexity}/4
              </span>
              
              {!claim.available && (
                <span style={{
                  fontSize: '10px',
                  color: '#ff9800',
                  background: '#fff3e0',
                  padding: '2px 6px',
                  borderRadius: '3px'
                }}>
                  Coming Soon
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Render operation configuration
  const renderOperationConfig = () => {
    const claim = aplgClaims[activeOperation];
    if (!claim) return null;

    return (
      <div>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          marginBottom: '20px' 
        }}>
          <button
            onClick={() => setActiveOperation(null)}
            style={{
              padding: '6px 12px',
              background: '#f5f5f5',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px',
              cursor: 'pointer',
              marginRight: '15px'
            }}
          >
            ← Back
          </button>
          <h4 style={{ margin: 0, color: claim.color }}>
            {claim.icon} {claim.label}
          </h4>
        </div>

        <div style={{
          padding: '15px',
          background: '#f8f9fa',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <p style={{ margin: 0, fontSize: '14px', color: '#666' }}>
            {claim.description}
          </p>
        </div>

        {renderSpecificConfig(activeOperation)}

        <div style={{ 
          display: 'flex', 
          gap: '10px', 
          marginTop: '20px' 
        }}>
          <button
            onClick={executeOperation}
            style={{
              padding: '12px 24px',
              background: claim.color,
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Execute Operation
          </button>
          <button
            onClick={() => setActiveOperation(null)}
            style={{
              padding: '12px 24px',
              background: '#f5f5f5',
              color: '#666',
              border: '1px solid #ddd',
              borderRadius: '6px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  };

  // Render operation-specific configuration
  const renderSpecificConfig = (operationType) => {
    switch (operationType) {
      case 'lexical_projection':
        return renderLexicalProjectionConfig();
      case 'quantum_superposition':
        return renderQuantumSuperpositionConfig();
      case 'measurement_collapse':
        return renderMeasurementCollapseConfig();
      case 'consent_gating':
        return renderConsentGatingConfig();
      default:
        return (
          <InlineNotification
            message="Configuration interface for this operation is under development."
            type="info"
          />
        );
    }
  };

  const renderLexicalProjectionConfig = () => (
    <div>
      <h5 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>
        Projection Parameters
      </h5>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Target Style:
        </label>
        <select
          value={operationParams.targetStyle || 'academic'}
          onChange={(e) => setOperationParams({ ...operationParams, targetStyle: e.target.value })}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        >
          <option value="academic">Academic</option>
          <option value="conversational">Conversational</option>
          <option value="poetic">Poetic</option>
          <option value="technical">Technical</option>
          <option value="narrative">Narrative</option>
        </select>
      </div>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Preservation Strength: {operationParams.preservationStrength || 0.8}
        </label>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.1"
          value={operationParams.preservationStrength || 0.8}
          onChange={(e) => setOperationParams({ 
            ...operationParams, 
            preservationStrength: parseFloat(e.target.value) 
          })}
          style={{ width: '100%' }}
        />
        <div style={{ fontSize: '10px', color: '#666', marginTop: '2px' }}>
          Lower: More creative freedom | Higher: Stricter preservation
        </div>
      </div>
    </div>
  );

  const renderQuantumSuperpositionConfig = () => (
    <div>
      <h5 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>
        Superposition Parameters
      </h5>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Number of Interpretive States:
        </label>
        <input
          type="number"
          min="2"
          max="8"
          value={operationParams.numStates || 3}
          onChange={(e) => setOperationParams({ 
            ...operationParams, 
            numStates: parseInt(e.target.value) 
          })}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        />
      </div>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ 
          display: 'flex', 
          alignItems: 'center', 
          fontSize: '12px' 
        }}>
          <input
            type="checkbox"
            checked={operationParams.includeContradictory || false}
            onChange={(e) => setOperationParams({ 
              ...operationParams, 
              includeContradictory: e.target.checked 
            })}
            style={{ marginRight: '8px' }}
          />
          Include contradictory interpretations
        </label>
      </div>
    </div>
  );

  const renderMeasurementCollapseConfig = () => (
    <div>
      <h5 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>
        Measurement Parameters
      </h5>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          POVM Pack:
        </label>
        <select
          value={operationParams.povmPack || 'advanced_narrative_pack'}
          onChange={(e) => setOperationParams({ ...operationParams, povmPack: e.target.value })}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        >
          <option value="advanced_narrative_pack">Advanced Narrative Pack</option>
          <option value="basic_semantic_pack">Basic Semantic Pack</option>
          <option value="stylistic_analysis_pack">Stylistic Analysis Pack</option>
          <option value="custom_pack">Custom Pack</option>
        </select>
      </div>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Measurement Precision: {operationParams.precision || 0.95}
        </label>
        <input
          type="range"
          min="0.5"
          max="1.0"
          step="0.05"
          value={operationParams.precision || 0.95}
          onChange={(e) => setOperationParams({ 
            ...operationParams, 
            precision: parseFloat(e.target.value) 
          })}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );

  const renderConsentGatingConfig = () => (
    <div>
      <h5 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>
        Risk Assessment Parameters
      </h5>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Risk Tolerance:
        </label>
        <select
          value={operationParams.riskTolerance || 'moderate'}
          onChange={(e) => setOperationParams({ ...operationParams, riskTolerance: e.target.value })}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        >
          <option value="conservative">Conservative</option>
          <option value="moderate">Moderate</option>
          <option value="permissive">Permissive</option>
        </select>
      </div>
      <div style={{ marginBottom: '15px' }}>
        <label style={{ 
          display: 'flex', 
          alignItems: 'center', 
          fontSize: '12px' 
        }}>
          <input
            type="checkbox"
            checked={operationParams.requireExplicitConsent || true}
            onChange={(e) => setOperationParams({ 
              ...operationParams, 
              requireExplicitConsent: e.target.checked 
            })}
            style={{ marginRight: '8px' }}
          />
          Require explicit consent for risky operations
        </label>
      </div>
      <InlineNotification
        message="This operation will analyze potential risks and ethical considerations before proceeding with other APLG operations."
        type="info"
        showIcon={true}
      />
    </div>
  );

  return (
    <div style={{
      padding: '20px',
      background: 'white',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      ...style
    }}>
      {!activeOperation ? renderOperationSelector() : renderOperationConfig()}
    </div>
  );
}

export default APLGOperationCenter;