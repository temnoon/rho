/**
 * ParameterSliders - Real-time quantum attribute controls.
 * 
 * This component provides intuitive sliders for adjusting quantum state attributes
 * in real-time, implementing the framework's unitary steering operations.
 */

import React from 'react';

function ParameterSliders({ attributes, onAttributeChange, disabled = false, manualMode = false, onManualUpdate }) {
  
  const attributeDefinitions = {
    reliability: {
      label: 'Narrator Reliability',
      description: 'How trustworthy is the narrator?',
      color: '#4caf50',
      range: [-1, 1]
    },
    formality: {
      label: 'Formality Level',
      description: 'Formal vs. informal language style',
      color: '#2196f3',
      range: [-1, 1]
    },
    emotional_intensity: {
      label: 'Emotional Intensity',
      description: 'Subdued vs. intense emotional expression',
      color: '#e91e63',
      range: [-1, 1]
    },
    personal_focus: {
      label: 'Personal Focus',
      description: 'Impersonal vs. deeply personal perspective',
      color: '#ff9800',
      range: [-1, 1]
    },
    temporal_focus: {
      label: 'Temporal Focus',
      description: 'Timeless vs. time-bound narrative',
      color: '#9c27b0',
      range: [-1, 1]
    },
    certainty: {
      label: 'Certainty Level',
      description: 'Uncertain vs. confident tone',
      color: '#607d8b',
      range: [-1, 1]
    }
  };

  const handleSliderChange = (attributeName, event) => {
    const value = parseFloat(event.target.value);
    onAttributeChange(attributeName, value);
  };

  const formatValue = (value) => {
    if (value > 0.1) return `+${(value * 100).toFixed(0)}%`;
    if (value < -0.1) return `${(value * 100).toFixed(0)}%`;
    return 'neutral';
  };

  const getSliderStyle = (attributeName, value) => {
    const def = attributeDefinitions[attributeName];
    const intensity = Math.abs(value);
    
    return {
      width: '100%',
      height: '6px',
      borderRadius: '3px',
      outline: 'none',
      background: `linear-gradient(to right, 
        ${def.color}44 0%, 
        ${def.color} ${50 + value * 50}%, 
        ${def.color}44 100%)`,
      opacity: disabled ? 0.5 : 1,
      cursor: disabled ? 'not-allowed' : 'pointer'
    };
  };

  return (
    <div style={{ 
      border: '1px solid #ddd', 
      borderRadius: '8px', 
      padding: '16px',
      backgroundColor: disabled ? '#f9f9f9' : 'white'
    }}>
      <h4 style={{ 
        margin: '0 0 16px 0', 
        color: disabled ? '#999' : '#333',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <span>⚡</span>
        Quantum Attribute Controls
        {disabled && (
          <span style={{ 
            fontSize: '12px', 
            color: '#ff9800',
            fontWeight: 'normal'
          }}>
            (Auto-update disabled)
          </span>
        )}
      </h4>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {Object.entries(attributes).map(([attributeName, value]) => {
          const def = attributeDefinitions[attributeName];
          if (!def) return null;
          
          return (
            <div key={attributeName} style={{ position: 'relative' }}>
              {/* Attribute Label and Value */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                marginBottom: '6px'
              }}>
                <label style={{ 
                  fontSize: '13px', 
                  fontWeight: '600',
                  color: disabled ? '#999' : def.color
                }}>
                  {def.label}
                </label>
                <div style={{ 
                  fontSize: '12px', 
                  fontWeight: 'bold',
                  color: disabled ? '#999' : def.color,
                  minWidth: '60px',
                  textAlign: 'right'
                }}>
                  {formatValue(value)}
                </div>
              </div>
              
              {/* Slider */}
              <div style={{ position: 'relative' }}>
                <input
                  type="range"
                  min={def.range[0]}
                  max={def.range[1]}
                  step="0.05"
                  value={value}
                  onChange={(e) => handleSliderChange(attributeName, e)}
                  disabled={disabled}
                  style={getSliderStyle(attributeName, value)}
                />
                
                {/* Center marker */}
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: '2px',
                  height: '10px',
                  backgroundColor: '#666',
                  pointerEvents: 'none',
                  opacity: 0.5
                }} />
              </div>
              
              {/* Range Labels */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                fontSize: '10px',
                color: disabled ? '#bbb' : '#666',
                marginTop: '2px'
              }}>
                <span>Low</span>
                <span>High</span>
              </div>
              
              {/* Description */}
              <div style={{ 
                fontSize: '11px', 
                color: disabled ? '#bbb' : '#777',
                fontStyle: 'italic',
                marginTop: '4px'
              }}>
                {def.description}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Quick Presets */}
      <div style={{ 
        marginTop: '16px',
        paddingTop: '16px',
        borderTop: '1px solid #eee'
      }}>
        <div style={{ 
          fontSize: '12px', 
          fontWeight: '600',
          marginBottom: '8px',
          color: disabled ? '#999' : '#333'
        }}>
          Quick Presets:
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <button
            onClick={() => {
              if (!disabled) {
                Object.keys(attributes).forEach(attr => {
                  onAttributeChange(attr, 0);
                });
              }
            }}
            disabled={disabled}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              border: '1px solid #ddd',
              borderRadius: '3px',
              backgroundColor: disabled ? '#f5f5f5' : 'white',
              cursor: disabled ? 'not-allowed' : 'pointer',
              color: disabled ? '#999' : '#333'
            }}
          >
            Reset All
          </button>
          <button
            onClick={() => {
              if (!disabled) {
                onAttributeChange('reliability', 0.8);
                onAttributeChange('formality', 0.3);
                onAttributeChange('certainty', 0.6);
              }
            }}
            disabled={disabled}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              border: '1px solid #4caf50',
              borderRadius: '3px',
              backgroundColor: disabled ? '#f5f5f5' : '#4caf50',
              color: disabled ? '#999' : 'white',
              cursor: disabled ? 'not-allowed' : 'pointer'
            }}
          >
            Trustworthy
          </button>
          <button
            onClick={() => {
              if (!disabled) {
                onAttributeChange('reliability', -0.6);
                onAttributeChange('emotional_intensity', 0.8);
                onAttributeChange('certainty', -0.4);
              }
            }}
            disabled={disabled}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              border: '1px solid #e91e63',
              borderRadius: '3px',
              backgroundColor: disabled ? '#f5f5f5' : '#e91e63',
              color: disabled ? '#999' : 'white',
              cursor: disabled ? 'not-allowed' : 'pointer'
            }}
          >
            Dramatic
          </button>
          <button
            onClick={() => {
              if (!disabled) {
                onAttributeChange('formality', -0.7);
                onAttributeChange('personal_focus', 0.9);
                onAttributeChange('emotional_intensity', 0.4);
              }
            }}
            disabled={disabled}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              border: '1px solid #ff9800',
              borderRadius: '3px',
              backgroundColor: disabled ? '#f5f5f5' : '#ff9800',
              color: disabled ? '#999' : 'white',
              cursor: disabled ? 'not-allowed' : 'pointer'
            }}
          >
            Personal
          </button>
        </div>
        
        {/* Manual Update Button */}
        {manualMode && onManualUpdate && (
          <div style={{ marginTop: '16px', textAlign: 'center' }}>
            <button
              onClick={onManualUpdate}
              disabled={disabled}
              style={{
                padding: '12px 24px',
                fontSize: '14px',
                fontWeight: 'bold',
                border: 'none',
                borderRadius: '6px',
                backgroundColor: disabled ? '#f5f5f5' : '#2196f3',
                color: disabled ? '#999' : 'white',
                cursor: disabled ? 'not-allowed' : 'pointer',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => {
                if (!disabled) {
                  e.target.style.backgroundColor = '#1976d2';
                  e.target.style.transform = 'translateY(-1px)';
                }
              }}
              onMouseOut={(e) => {
                if (!disabled) {
                  e.target.style.backgroundColor = '#2196f3';
                  e.target.style.transform = 'translateY(0)';
                }
              }}
            >
              🎯 Update with changes in slider
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default ParameterSliders;