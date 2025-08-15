/**
 * NarrativeTab - Advanced narrative interface with live parameter controls.
 * 
 * This component implements the key insight that narratives should be the primary
 * interface for quantum state manipulation, with real-time parameter sliders
 * that immediately show changes via unitary operations.
 */

import React, { useState, useEffect, useCallback } from 'react';
import ParameterSliders from './ParameterSliders.jsx';
import AdvancedLinguisticInterface from './AdvancedLinguisticInterface.jsx';
import POVMCreator from './POVMCreator.jsx';
import MatrixVisualization from './MatrixVisualization.jsx';
import { useSharedNarrative } from '../hooks/useSharedNarrative.js';
import { useQuantumState } from '../hooks/useQuantumState.js';
import { apiUrl, safeFetch } from '../utils/api.js';

function NarrativeTab() {
  const { sharedNarrative, updateSharedNarrative } = useSharedNarrative();
  const { 
    currentRho, 
    createMatrix, 
    readNarrative, 
    measureMatrix,
    steerMatrix,
    applyChannel 
  } = useQuantumState();
  
  const [narrativeText, setNarrativeText] = useState(
    sharedNarrative || "In the quantum realm of narrative possibilities, each word carries the weight of infinite interpretations..."
  );
  
  const [originalNarrative, setOriginalNarrative] = useState("");
  const [transformedNarrative, setTransformedNarrative] = useState("");
  
  const [attributes, setAttributes] = useState({
    involved_production: 0.0,
    narrative_concerns: 0.0,
    elaborated_reference: 0.0,
    tenor_formality: 0.0,
    tenor_affect: 0.0,
    temporal_perspective: 0.0
  });
  
  const [isReading, setIsReading] = useState(false);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [useAdvancedInterface, setUseAdvancedInterface] = useState(true);
  const [lastOperation, setLastOperation] = useState(null);
  const [measurements, setMeasurements] = useState(null);
  const [generationHistory, setGenerationHistory] = useState([]);

  // Initialize matrix on component mount
  useEffect(() => {
    if (!currentRho.rho_id) {
      initializeNarrativeMatrix();
    }
  }, []);

  // Auto-measure only when rho_id changes (new matrix created)
  useEffect(() => {
    if (currentRho.rho_id && autoUpdate) {
      measureCurrentState();
    }
  }, [currentRho.rho_id, autoUpdate]);

  const initializeNarrativeMatrix = async () => {
    try {
      const rho_id = await createMatrix(narrativeText, "Narrative Interface");
      if (rho_id && narrativeText.trim()) {
        // Read the initial narrative text first
        await readNarrative(rho_id, narrativeText);
        // Skip POVM generation on initialization - use existing advanced_narrative_pack
        // POVM generation can be triggered later when more narrative content is available
      }
    } catch (error) {
      console.error('Failed to initialize narrative matrix:', error);
    }
  };

  const generateNarrativePOVM = async (rho_id) => {
    try {
      const response = await safeFetch('/advanced/povm/generate_from_narrative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rho_id: rho_id,
          pack_name: 'advanced_narrative_pack',
          n_measurements: 8
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Generated narrative-optimized POVM pack:', result);
        setLastOperation({
          type: 'povm_generation',
          result: result,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      // If POVM generation fails (e.g., insufficient samples), just log and continue
      // The system will use the existing advanced_narrative_pack
      console.log('POVM generation skipped:', error.message);
      setLastOperation({
        type: 'povm_generation_skipped',
        reason: 'Using existing advanced pack due to insufficient narrative samples',
        timestamp: new Date().toISOString()
      });
    }
  };

  const handleNarrativeChange = useCallback(async (newText) => {
    setNarrativeText(newText);
    updateSharedNarrative(newText);
    
    if (autoUpdate && currentRho.rho_id && newText.trim()) {
      setIsReading(true);
      try {
        await readNarrative(currentRho.rho_id, newText);
        setLastOperation({
          type: 'narrative_update',
          text: newText.slice(0, 100) + '...',
          timestamp: new Date().toISOString()
        });
        // Measure the updated state
        await measureCurrentState();
      } catch (error) {
        console.error('Failed to update narrative:', error);
      } finally {
        setIsReading(false);
      }
    }
  }, [currentRho.rho_id, autoUpdate, updateSharedNarrative, readNarrative]);

  const handleAttributeChange = useCallback(async (attributeName, value) => {
    const newAttributes = { ...attributes, [attributeName]: value };
    setAttributes(newAttributes);
    
    // Only apply changes immediately if autoUpdate is enabled
    if (autoUpdate && currentRho.rho_id) {
      await applyAttributeChanges({ [attributeName]: value });
    }
  }, [attributes, currentRho.rho_id, autoUpdate]);

  const applyAttributeChanges = useCallback(async (targetAttributes) => {
    if (!currentRho.rho_id) return;
    
    try {
      // Store the original narrative for comparison
      setOriginalNarrative(narrativeText);
      
      // Apply unitary steering to hit the target attribute values
      await steerMatrix(currentRho.rho_id, targetAttributes);
      
      setLastOperation({
        type: 'attribute_steering',
        attributes: targetAttributes,
        timestamp: new Date().toISOString()
      });
      
      // Measure the new state to update UI
      await measureCurrentState();
      
      // Generate new narrative with the steered quantum state
      try {
        await regenerateNarrative();
        setLastOperation({
          type: 'steering_and_generation_complete',
          message: 'Successfully applied attribute changes and generated new narrative text.',
          attributes: targetAttributes,
          timestamp: new Date().toISOString()
        });
      } catch (narrativeError) {
        console.warn('Narrative generation failed, but quantum state was updated:', narrativeError);
        setLastOperation({
          type: 'steering_complete',
          message: 'Quantum state successfully steered. Narrative generation is not yet available.',
          attributes: targetAttributes,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      console.error('Failed to steer attributes:', error);
    }
  }, [currentRho.rho_id, steerMatrix]);

  const handleManualUpdate = useCallback(async () => {
    await applyAttributeChanges(attributes);
  }, [attributes, applyAttributeChanges]);

  const handlePOVMCreated = useCallback((newPOVM) => {
    setLastOperation({
      type: 'povm_created',
      pack_name: newPOVM.pack_id,
      pack_type: newPOVM.pack_type,
      timestamp: new Date().toISOString()
    });
  }, []);

  const measureCurrentState = async () => {
    if (!currentRho.rho_id) return;
    
    try {
      const response = await measureMatrix(currentRho.rho_id, 'advanced_narrative_pack');
      setMeasurements(response);
      
      // Update attribute sliders based on measurements
      if (response && response.measurements) {
        const newAttributes = {};
        Object.entries(response.measurements).forEach(([key, value]) => {
          if (key.endsWith('_high')) {
            const attrName = key.replace('_high', '');
            newAttributes[attrName] = value;
          }
        });
        setAttributes(prev => ({ ...prev, ...newAttributes }));
      }
    } catch (error) {
      console.error('Failed to measure state:', error);
    }
  };

  const regenerateNarrative = async () => {
    if (!currentRho.rho_id) return;
    
    try {
      // Use the current quantum state to generate a new narrative variation
      const response = await safeFetch('/attributes/regenerate_narrative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_text: narrativeText,
          adjusted_rho_id: currentRho.rho_id
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        if (result.transformed_text) {
          // Store the transformed narrative for side-by-side comparison
          setTransformedNarrative(result.transformed_text);
          
          setGenerationHistory(prev => [...prev, {
            original: narrativeText,
            generated: result.transformed_text,
            attributes: { ...attributes },
            timestamp: new Date().toISOString()
          }]);
          
          // Keep original in the main text area, show transformed separately
          // setNarrativeText(result.transformed_text);
          // updateSharedNarrative(result.transformed_text);
        }
      }
    } catch (error) {
      console.error('Failed to regenerate narrative:', error);
    }
  };

  const applyStyleTransformation = async (styleName) => {
    if (!currentRho.rho_id) return;
    
    try {
      await applyChannel(currentRho.rho_id, 'style', { 
        style_name: styleName,
        strength: 0.7 
      });
      
      setLastOperation({
        type: 'style_transformation',
        style: styleName,
        timestamp: new Date().toISOString()
      });
      
      // Regenerate narrative with new style
      await regenerateNarrative();
      
    } catch (error) {
      console.error('Failed to apply style transformation:', error);
    }
  };

  const resetToMaximallyMixed = async () => {
    if (!currentRho.rho_id) return;
    
    try {
      await applyChannel(currentRho.rho_id, 'depolarize', { strength: 0.5 });
      
      setLastOperation({
        type: 'depolarization',
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('Failed to apply depolarization:', error);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', height: '100vh' }}>
      <div style={{ display: 'flex', gap: '20px', height: '100%' }}>
        
        {/* Left Sidebar - Controls */}
        <div style={{ 
          width: '380px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          overflowY: 'auto',
          maxHeight: '100%'
        }}>
          {/* Header Controls */}
          <div style={{ 
            border: '2px solid #2196f3', 
            borderRadius: '8px', 
            padding: '16px'
          }}>
            <h3 style={{ margin: '0 0 12px 0', color: '#2196f3' }}>Narrative Controls</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ fontSize: '14px' }} title="When enabled, changes to sliders immediately update the quantum state">
                <input
                  type="checkbox"
                  checked={autoUpdate}
                  onChange={(e) => setAutoUpdate(e.target.checked)}
                  style={{ marginRight: '4px' }}
                />
                Auto-update
              </label>
              <label style={{ fontSize: '14px' }} title="Show advanced linguistic controls based on computational linguistics research">
                <input
                  type="checkbox"
                  checked={useAdvancedInterface}
                  onChange={(e) => setUseAdvancedInterface(e.target.checked)}
                  style={{ marginRight: '4px' }}
                />
                Advanced Interface
              </label>
              {!autoUpdate && (
                <div style={{ 
                  fontSize: '12px', 
                  color: '#666', 
                  fontStyle: 'italic',
                  marginTop: '4px'
                }}>
                  Manual mode: Use "Apply Changes" button
                </div>
              )}
              {!autoUpdate && !isReading && (
                <button
                  onClick={handleManualUpdate}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    marginTop: '8px'
                  }}
                >
                  🎯 Apply Changes & Generate New Text
                </button>
              )}
              {isReading && (
                <div style={{ 
                  fontSize: '12px', 
                  color: '#ff9800',
                  fontWeight: 'bold',
                  marginTop: '8px'
                }}>
                  ⚡ Processing...
                </div>
              )}
            </div>
          </div>

          {/* Parameter Controls */}
          {useAdvancedInterface ? (
            <AdvancedLinguisticInterface
              currentRho={currentRho}
              onAttributeChange={handleAttributeChange}
              disabled={isReading}
              manualMode={!autoUpdate}
              onManualUpdate={handleManualUpdate}
            />
          ) : (
            <ParameterSliders
              attributes={attributes}
              onAttributeChange={handleAttributeChange}
              disabled={isReading}
              manualMode={!autoUpdate}
              onManualUpdate={handleManualUpdate}
            />
          )}

          {/* Matrix Visualization */}
          {currentRho.diagnostics && (
            <div style={{ 
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '12px'
            }}>
              <MatrixVisualization
                title="Quantum State"
                eigs={currentRho.diagnostics.eigenvals}
                purity={currentRho.diagnostics.purity}
                entropy={currentRho.diagnostics.entropy}
                size={200}
              />
            </div>
          )}

          {/* Measurements Display */}
          {measurements && (
            <div style={{ 
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '12px'
            }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>Live Measurements</h4>
              <div style={{ fontSize: '12px', maxHeight: '120px', overflowY: 'auto' }}>
                {Object.entries(measurements.measurements || {}).map(([key, value]) => (
                  <div key={key} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    marginBottom: '2px'
                  }}>
                    <span>{key.replace('_', ' ')}</span>
                    <span style={{ fontWeight: 'bold' }}>
                      {(value * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Last Operation */}
          {lastOperation && (
            <div style={{ 
              padding: '8px',
              backgroundColor: '#e3f2fd',
              borderRadius: '4px',
              fontSize: '12px'
            }}>
              <div style={{ fontWeight: 'bold' }}>Last Operation:</div>
              <div>{lastOperation.type.replace('_', ' ')}</div>
              {lastOperation.pack_name && (
                <div>Pack: {lastOperation.pack_name}</div>
              )}
              <div style={{ fontSize: '10px', color: '#666' }}>
                {new Date(lastOperation.timestamp).toLocaleTimeString()}
              </div>
            </div>
          )}
        </div>
        
        {/* Main Content Panel */}
        <div style={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          overflowY: 'auto',
          maxHeight: '100%'
        }}>
          {/* Original Narrative Input */}
          <div style={{ 
            border: '2px solid #2196f3', 
            borderRadius: '8px', 
            padding: '16px'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '12px'
            }}>
              <h3 style={{ margin: 0, color: '#2196f3' }}>Original Narrative</h3>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                <label style={{ fontSize: '14px' }} title="When enabled, changes to sliders immediately update the quantum state">
                  <input
                    type="checkbox"
                    checked={autoUpdate}
                    onChange={(e) => setAutoUpdate(e.target.checked)}
                    style={{ marginRight: '4px' }}
                  />
                  Auto-update
                </label>
                <label style={{ fontSize: '14px' }} title="Show advanced linguistic controls based on computational linguistics research">
                  <input
                    type="checkbox"
                    checked={useAdvancedInterface}
                    onChange={(e) => setUseAdvancedInterface(e.target.checked)}
                    style={{ marginRight: '4px' }}
                  />
                  Advanced Interface
                </label>
                {!autoUpdate && (
                  <span style={{ 
                    fontSize: '12px', 
                    color: '#666', 
                    fontStyle: 'italic',
                    marginLeft: '8px'
                  }}>
                    Manual mode: Use "Apply Changes" button
                  </span>
                )}
                {!autoUpdate && !isReading && (
                  <button
                    onClick={handleManualUpdate}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#007bff',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '14px',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      marginLeft: '8px'
                    }}
                  >
                    🎯 Apply Changes & Generate New Text
                  </button>
                )}
                {isReading && (
                  <div style={{ 
                    fontSize: '12px', 
                    color: '#ff9800',
                    fontWeight: 'bold'
                  }}>
                    ⚡ Processing...
                  </div>
                )}
              </div>
            </div>
            
            <textarea
              value={narrativeText}
              onChange={(e) => handleNarrativeChange(e.target.value)}
              style={{
                width: '100%',
                minHeight: '200px',
                padding: '12px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '16px',
                lineHeight: '1.5',
                fontFamily: 'Georgia, serif',
                resize: 'vertical'
              }}
              placeholder="Enter your narrative here... The quantum state will evolve with your words."
            />
            
            <div style={{ 
              marginTop: '12px',
              display: 'flex',
              gap: '8px',
              flexWrap: 'wrap'
            }}>
              <button
                onClick={() => applyStyleTransformation('noir')}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#333',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Noir Style
              </button>
              <button
                onClick={() => applyStyleTransformation('romantic')}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#e91e63',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Romantic Style
              </button>
              <button
                onClick={() => applyStyleTransformation('minimalist')}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#9e9e9e',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Minimalist Style
              </button>
              <button
                onClick={resetToMaximallyMixed}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#ff5722',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Reset (Depolarize)
              </button>
            </div>
          </div>
          
          {/* Side-by-Side Comparison */}
          {originalNarrative && transformedNarrative && (
            <div style={{ 
              marginBottom: '16px',
              border: '2px solid #4caf50',
              borderRadius: '8px',
              padding: '16px',
              backgroundColor: '#f8fff8'
            }}>
              <h4 style={{ 
                margin: '0 0 12px 0', 
                color: '#4caf50',
                fontSize: '16px',
                fontWeight: 'bold'
              }}>
                📖 Original vs Transformed Narrative
              </h4>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: '1fr 1fr', 
                gap: '16px',
                minHeight: '200px'
              }}>
                <div style={{ 
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  padding: '12px',
                  backgroundColor: 'white'
                }}>
                  <h5 style={{ 
                    margin: '0 0 8px 0', 
                    fontSize: '14px', 
                    color: '#666',
                    fontWeight: 'bold'
                  }}>
                    Original
                  </h5>
                  <div style={{ 
                    fontSize: '14px', 
                    lineHeight: '1.6',
                    fontFamily: 'Georgia, serif',
                    maxHeight: '300px',
                    overflowY: 'auto'
                  }}>
                    {originalNarrative}
                  </div>
                </div>
                <div style={{ 
                  border: '1px solid #4caf50',
                  borderRadius: '4px',
                  padding: '12px',
                  backgroundColor: '#f8fff8'
                }}>
                  <h5 style={{ 
                    margin: '0 0 8px 0', 
                    fontSize: '14px', 
                    color: '#4caf50',
                    fontWeight: 'bold'
                  }}>
                    ✨ Transformed 
                  </h5>
                  <div style={{ 
                    fontSize: '14px', 
                    lineHeight: '1.6',
                    fontFamily: 'Georgia, serif',
                    maxHeight: '300px',
                    overflowY: 'auto'
                  }}>
                    {transformedNarrative}
                  </div>
                </div>
              </div>
              <div style={{ 
                marginTop: '12px',
                textAlign: 'center'
              }}>
                <button
                  onClick={() => {
                    setOriginalNarrative("");
                    setTransformedNarrative("");
                  }}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  Clear Comparison
                </button>
              </div>
            </div>
          )}
          
          {/* Generation History */}
          {generationHistory.length > 0 && (
            <div style={{ 
              border: '1px solid #ddd', 
              borderRadius: '8px', 
              padding: '16px',
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              <h4 style={{ margin: '0 0 12px 0' }}>Generation History</h4>
              {generationHistory.slice(-3).map((entry, i) => (
                <div key={i} style={{ 
                  marginBottom: '12px', 
                  padding: '8px',
                  backgroundColor: '#f5f5f5',
                  borderRadius: '4px',
                  fontSize: '14px'
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </div>
                  <div style={{ fontStyle: 'italic' }}>
                    {entry.generated.slice(0, 100)}...
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Control Panel */}
        <div style={{ flex: 1 }}>
          {/* Parameter Controls */}
          {useAdvancedInterface ? (
            <AdvancedLinguisticInterface
              currentRho={currentRho}
              onAttributeChange={handleAttributeChange}
              disabled={isReading}
              manualMode={!autoUpdate}
              onManualUpdate={handleManualUpdate}
            />
          ) : (
            <ParameterSliders
              attributes={attributes}
              onAttributeChange={handleAttributeChange}
              disabled={isReading}
              manualMode={!autoUpdate}
              onManualUpdate={handleManualUpdate}
            />
          )}
          
          {/* Matrix Visualization */}
          {currentRho.diagnostics && (
            <div style={{ marginTop: '20px' }}>
              <MatrixVisualization
                title="Narrative Quantum State"
                eigs={currentRho.diagnostics.eigenvals}
                purity={currentRho.diagnostics.purity}
                entropy={currentRho.diagnostics.entropy}
                size={250}
              />
            </div>
          )}
          
          {/* Measurements Display */}
          {measurements && (
            <div style={{ 
              marginTop: '20px',
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '12px'
            }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>Live Measurements</h4>
              <div style={{ fontSize: '12px', maxHeight: '120px', overflowY: 'auto' }}>
                {Object.entries(measurements.measurements || {}).map(([key, value]) => (
                  <div key={key} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    marginBottom: '2px'
                  }}>
                    <span>{key.replace('_', ' ')}</span>
                    <span style={{ fontWeight: 'bold' }}>
                      {(value * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* POVM Creator */}
          <POVMCreator 
            onPOVMCreated={handlePOVMCreated}
            disabled={isReading}
          />
          
          {/* Last Operation */}
          {lastOperation && (
            <div style={{ 
              marginTop: '16px',
              padding: '8px',
              backgroundColor: '#e3f2fd',
              borderRadius: '4px',
              fontSize: '12px'
            }}>
              <div style={{ fontWeight: 'bold' }}>Last Operation:</div>
              <div>{lastOperation.type.replace('_', ' ')}</div>
              {lastOperation.pack_name && (
                <div>Pack: {lastOperation.pack_name}</div>
              )}
              <div style={{ fontSize: '10px', color: '#666' }}>
                {new Date(lastOperation.timestamp).toLocaleTimeString()}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default NarrativeTab;