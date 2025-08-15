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
  const [isGenerating, setIsGenerating] = useState(false);
  const [autoUpdate, setAutoUpdate] = useState(false); // Default to manual mode
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
      console.log('Creating new matrix for this narrative (not using global matrix)');
      const rho_id = await createMatrix(narrativeText, `Narrative_${Date.now()}`);
      if (rho_id && narrativeText.trim()) {
        console.log(`Created individual matrix ${rho_id} for this specific narrative`);
        await readNarrative(rho_id, narrativeText);
      }
    } catch (error) {
      console.error('Failed to initialize narrative matrix:', error);
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
    
    if (autoUpdate && currentRho.rho_id) {
      await applyAttributeChanges({ [attributeName]: value });
    }
  }, [attributes, currentRho.rho_id, autoUpdate]);

  const applyAttributeChanges = useCallback(async (targetAttributes) => {
    if (!currentRho.rho_id || isGenerating) return;
    
    setIsGenerating(true);
    
    try {
      console.log('Setting originalNarrative:', narrativeText.slice(0, 100));
      setOriginalNarrative(narrativeText);
      
      await steerMatrix(currentRho.rho_id, targetAttributes);
      
      setLastOperation({
        type: 'attribute_steering',
        attributes: targetAttributes,
        timestamp: new Date().toISOString()
      });
      
      await measureCurrentState();
      
      try {
        await regenerateNarrative();
        setLastOperation({
          type: 'steering_and_generation_complete',
          message: 'Successfully applied attribute changes and generated new narrative text.',
          attributes: targetAttributes,
          timestamp: new Date().toISOString()
        });
      } catch (narrativeError) {
        console.warn('Narrative generation requires LLM integration:', narrativeError);
        
        // Show the current quantum state measurements instead
        const errorMessage = narrativeError.message?.includes('timeout') 
          ? 'Quantum state successfully steered. Narrative generation timed out (Ollama may be slow). Check measurements panel for updated attributes.'
          : narrativeError.message?.includes('503') || narrativeError.message?.includes('service')
          ? 'Quantum state successfully steered. Ollama LLM service is not available. Please ensure Ollama is running.'
          : 'Quantum state successfully steered. Narrative text generation failed. Check measurements panel to see updated attribute values.';
          
        setLastOperation({
          type: 'steering_complete_no_llm',
          message: errorMessage,
          attributes: targetAttributes,
          timestamp: new Date().toISOString(),
          error: narrativeError.message
        });
        
        // Clear any previous comparison since no new text was generated
        setTransformedNarrative("");
        setOriginalNarrative(""); // Also clear original to prevent showing identical content
      }
      
    } catch (error) {
      console.error('Failed to steer attributes:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [currentRho.rho_id, steerMatrix, narrativeText, isGenerating]);

  const handleManualUpdate = useCallback(async () => {
    await applyAttributeChanges(attributes);
  }, [attributes, applyAttributeChanges]);

  const measureCurrentState = async () => {
    if (!currentRho.rho_id) return;
    
    try {
      const response = await measureMatrix(currentRho.rho_id, 'advanced_narrative_pack');
      setMeasurements(response);
      
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
      const response = await safeFetch('/advanced/regenerate_narrative', {
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
          console.log('Setting transformedNarrative:', result.transformed_text.slice(0, 100));
          setTransformedNarrative(result.transformed_text);
          
          setGenerationHistory(prev => [...prev, {
            original: narrativeText,
            generated: result.transformed_text,
            attributes: { ...attributes },
            timestamp: new Date().toISOString()
          }]);
        }
      } else if (response.status === 501) {
        // LLM integration required - this is expected
        throw new Error('LLM integration required for narrative transformation');
      } else {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
    } catch (error) {
      // Rethrow the error to be handled by caller
      throw error;
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      minHeight: '100vh',
      height: '100%',
      padding: '16px', 
      gap: '16px',
      maxWidth: '100vw',
      boxSizing: 'border-box',
      overflow: 'hidden'
    }}>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
      
      {/* Left Sidebar - Controls */}
      <div style={{ 
        width: '380px',
        minWidth: '380px',
        maxWidth: '380px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        height: 'calc(100vh - 32px)',
        overflowY: 'auto',
        backgroundColor: '#f8f9fa',
        padding: '16px',
        borderRadius: '8px',
        border: '1px solid #e0e0e0',
        boxSizing: 'border-box'
      }}>
        {/* Control Panel Header */}
        <div style={{ 
          border: '2px solid #2196f3', 
          borderRadius: '8px', 
          padding: '16px',
          backgroundColor: 'white'
        }}>
          <h3 style={{ margin: '0 0 8px 0', color: '#2196f3' }}>Narrative Controls</h3>
          {currentRho.rho_id && (
            <div style={{ 
              fontSize: '12px', 
              color: '#666',
              marginBottom: '8px',
              fontFamily: 'monospace',
              backgroundColor: '#f0f7ff',
              padding: '4px 8px',
              borderRadius: '4px'
            }}>
              Using individual matrix: {currentRho.rho_id.slice(0, 8)}...
            </div>
          )}
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
                Manual mode: Adjust sliders then click Apply
              </div>
            )}
            {!autoUpdate && (
              <button
                onClick={handleManualUpdate}
                disabled={isReading || isGenerating}
                style={{
                  padding: '10px 16px',
                  backgroundColor: (isReading || isGenerating) ? '#6c757d' : '#007bff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  cursor: (isReading || isGenerating) ? 'not-allowed' : 'pointer',
                  marginTop: '8px',
                  opacity: (isReading || isGenerating) ? 0.7 : 1,
                  position: 'relative'
                }}
              >
                {(isReading || isGenerating) ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      display: 'inline-block',
                      width: '12px',
                      height: '12px',
                      border: '2px solid #ffffff40',
                      borderTop: '2px solid white',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }}></span>
                    Generating...
                  </span>
                ) : (
                  '🎯 Apply Changes & Generate New Text'
                )}
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
        <div style={{ backgroundColor: 'white', borderRadius: '8px', padding: '16px', border: '1px solid #e0e0e0' }}>
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
        </div>

        {/* Matrix Visualization */}
        {currentRho.diagnostics && (
          <div style={{ 
            border: '1px solid #ddd',
            borderRadius: '8px',
            padding: '12px',
            backgroundColor: 'white'
          }}>
            <MatrixVisualization
              title="Quantum State"
              eigs={currentRho.diagnostics.eigenvals}
              purity={currentRho.diagnostics.purity}
              entropy={currentRho.diagnostics.entropy}
              size={180}
            />
          </div>
        )}

        {/* Measurements Display */}
        {measurements && (
          <div style={{ 
            border: '1px solid #ddd',
            borderRadius: '8px',
            padding: '12px',
            backgroundColor: 'white'
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
      </div>
      
      {/* Main Content Panel */}
      <div style={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        height: 'calc(100vh - 32px)',
        overflowY: 'auto',
        overflowX: 'hidden',
        paddingRight: '8px',
        boxSizing: 'border-box',
        minWidth: 0
      }}>
        
        {/* Original Narrative Input */}
        <div style={{ 
          border: '2px solid #2196f3', 
          borderRadius: '8px', 
          padding: '16px',
          backgroundColor: 'white'
        }}>
          <h3 style={{ margin: '0 0 12px 0', color: '#2196f3' }}>Original Narrative</h3>
          <textarea
            value={narrativeText}
            onChange={(e) => handleNarrativeChange(e.target.value)}
            style={{
              width: '100%',
              minHeight: '150px',
              maxHeight: '300px',
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
        </div>

        {/* Quantum State Status Panel */}
        {lastOperation && lastOperation.type === 'steering_complete_no_llm' && (
          <div style={{ 
            border: '2px solid #ff9800',
            borderRadius: '8px',
            padding: '20px',
            backgroundColor: '#fff3e0'
          }}>
            <h3 style={{ 
              margin: '0 0 12px 0', 
              color: '#ff9800',
              fontSize: '18px'
            }}>
              ⚡ Quantum State Successfully Modified
            </h3>
            <p style={{ margin: '0 0 16px 0', fontSize: '16px', lineHeight: '1.5' }}>
              Your attribute changes have been applied to the quantum density matrix using unitary steering operations. 
              The POVM measurements below show the new linguistic attribute values.
            </p>
            <div style={{ 
              padding: '12px',
              backgroundColor: '#fff8e1',
              borderRadius: '4px',
              border: '1px solid #ffcc02'
            }}>
              <strong>Note:</strong> To see actual narrative text transformations, this system requires integration 
              with an LLM service (OpenAI GPT, Anthropic Claude, etc.) that can generate new text based on the 
              quantum measurements. Currently, only the mathematical quantum operations are implemented.
            </div>
            <div style={{ 
              marginTop: '16px',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              {Object.entries(lastOperation.attributes || {}).map(([attr, value]) => (
                <div key={attr} style={{
                  padding: '4px 8px',
                  backgroundColor: '#ffcc02',
                  borderRadius: '4px',
                  fontSize: '14px',
                  fontWeight: 'bold'
                }}>
                  {attr.replace('_', ' ')}: {value.toFixed(3)}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Side-by-Side Comparison - MAIN FOCUS */}
        {(() => {
          const shouldShow = originalNarrative && transformedNarrative && originalNarrative !== transformedNarrative;
          console.log('Comparison condition:', {
            originalNarrative: !!originalNarrative,
            transformedNarrative: !!transformedNarrative,
            different: originalNarrative !== transformedNarrative,
            shouldShow
          });
          return shouldShow;
        })() && (
          <div style={{ 
            border: '3px solid #4caf50',
            borderRadius: '12px',
            padding: '16px',
            backgroundColor: '#f8fff8',
            display: 'flex',
            flexDirection: 'column',
            minHeight: '400px',
            maxHeight: '500px',
            marginBottom: '24px',
            boxSizing: 'border-box',
            flexShrink: 0
          }}>
            <div style={{ 
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px'
            }}>
              <h2 style={{ 
                margin: 0, 
                color: '#4caf50',
                fontSize: '20px',
                fontWeight: 'bold'
              }}>
                📖 Original vs Transformed Narrative
              </h2>
              <button
                onClick={() => {
                  setOriginalNarrative("");
                  setTransformedNarrative("");
                }}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  cursor: 'pointer'
                }}
              >
                Clear Comparison
              </button>
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr', 
              gap: '16px',
              flex: 1,
              minHeight: 0,
              maxHeight: 'calc(100% - 60px)',
              overflow: 'hidden'
            }}>
              <div style={{ 
                border: '2px solid #ddd',
                borderRadius: '8px',
                padding: '16px',
                backgroundColor: 'white',
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                minHeight: 0,
                boxSizing: 'border-box'
              }}>
                <h4 style={{ 
                  margin: '0 0 12px 0', 
                  fontSize: '16px', 
                  color: '#666',
                  fontWeight: 'bold'
                }}>
                  Original
                </h4>
                <div style={{ 
                  fontSize: '14px', 
                  lineHeight: '1.5',
                  fontFamily: 'Georgia, serif',
                  flex: 1,
                  minHeight: '0',
                  overflowY: 'auto',
                  overflowX: 'hidden',
                  padding: '12px',
                  border: '1px solid #f0f0f0',
                  borderRadius: '4px',
                  whiteSpace: 'pre-wrap',
                  wordWrap: 'break-word',
                  backgroundColor: '#fafafa',
                  boxSizing: 'border-box'
                }}>
                  {originalNarrative}
                </div>
              </div>
              
              <div style={{ 
                border: '2px solid #4caf50',
                borderRadius: '8px',
                padding: '16px',
                backgroundColor: '#f8fff8',
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                minHeight: 0,
                boxSizing: 'border-box'
              }}>
                <h4 style={{ 
                  margin: '0 0 12px 0', 
                  fontSize: '16px', 
                  color: '#4caf50',
                  fontWeight: 'bold'
                }}>
                  ✨ Transformed 
                </h4>
                <div style={{ 
                  fontSize: '14px', 
                  lineHeight: '1.5',
                  fontFamily: 'Georgia, serif',
                  flex: 1,
                  minHeight: '0',
                  overflowY: 'auto',
                  overflowX: 'hidden',
                  padding: '12px',
                  border: '1px solid #e8f5e8',
                  borderRadius: '4px',
                  backgroundColor: 'white',
                  whiteSpace: 'pre-wrap',
                  wordWrap: 'break-word',
                  boxSizing: 'border-box'
                }}>
                  {transformedNarrative}
                </div>
              </div>
            </div>
          </div>
        )}


        {/* Generation History */}
        {generationHistory.length > 0 && (
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '8px', 
            padding: '12px',
            backgroundColor: 'white',
            flexShrink: 0
          }}>
            <h4 style={{ margin: '0 0 12px 0' }}>Recent Transformations</h4>
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
    </div>
  );
}

export default NarrativeTab;