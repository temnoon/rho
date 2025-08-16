import React, { useState, useCallback } from 'react';

const CompassRosePOVM = ({ onTransformationApply, currentRhoState, isTransforming }) => {
  const [selectedOperator, setSelectedOperator] = useState('stance_rotation');
  const [operatorDirection, setOperatorDirection] = useState('north');
  const [operatorMagnitude, setOperatorMagnitude] = useState(0.5);
  const [transformationPasses, setTransformationPasses] = useState([]);
  const [showPassHistory, setShowPassHistory] = useState(false);

  // Define POVM operators with their semantic axes
  const povmOperators = {
    stance_rotation: {
      name: 'Stance Rotation',
      icon: '🎭',
      description: 'Academic ↔ Intimate perspective',
      axes: {
        north: 'More Academic/Formal',
        south: 'More Intimate/Personal', 
        east: 'More Authoritative',
        west: 'More Collaborative'
      },
      color: '#e74c3c'
    },
    reliability_calibration: {
      name: 'Reliability Calibration',
      icon: '⚖️',
      description: 'Certainty and trustworthiness signals',
      axes: {
        north: 'More Certain/Definitive',
        south: 'More Tentative/Exploratory',
        east: 'More Evidence-Based', 
        west: 'More Intuitive'
      },
      color: '#3498db'
    },
    affect_tuning: {
      name: 'Affect Tuning',
      icon: '💝',
      description: 'Emotional valence and arousal',
      axes: {
        north: 'Higher Arousal/Energy',
        south: 'Lower Arousal/Calm',
        east: 'More Positive Valence',
        west: 'More Neutral/Complex'
      },
      color: '#e67e22'
    },
    causal_grain: {
      name: 'Causal Grain Control',
      icon: '🔬',
      description: 'Abstract ↔ Concrete detail level',
      axes: {
        north: 'More Abstract/Theoretical',
        south: 'More Concrete/Practical',
        east: 'More Systemic/Global',
        west: 'More Local/Specific'
      },
      color: '#9b59b6'
    },
    ambiguity_coherence: {
      name: 'Ambiguity/Coherence Balance',
      icon: '🌊',
      description: 'Clarity vs productive uncertainty',
      axes: {
        north: 'More Coherent/Clear',
        south: 'More Ambiguous/Open',
        east: 'More Linear/Sequential',
        west: 'More Associative/Lateral'
      },
      color: '#1abc9c'
    },
    dialogic_engagement: {
      name: 'Dialogic Engagement',
      icon: '💬',
      description: 'Reader interaction and participation',
      axes: {
        north: 'More Engaging/Interactive',
        south: 'More Contemplative/Reflective',
        east: 'More Questioning/Probing',
        west: 'More Declarative/Informing'
      },
      color: '#f39c12'
    },
    lifeworld_anchoring: {
      name: 'Lifeworld Anchoring',
      icon: '🏠',
      description: 'Lived experience and embodiment',
      axes: {
        north: 'More Embodied/Sensory',
        south: 'More Conceptual/Mental',
        east: 'More Situational/Contextual',
        west: 'More Universal/General'
      },
      color: '#27ae60'
    },
    residue_creation: {
      name: 'Residue Creation',
      icon: '✨',
      description: 'Memorability and linguistic traces',
      axes: {
        north: 'More Memorable/Striking',
        south: 'More Subtle/Understated',
        east: 'More Rhythmic/Musical',
        west: 'More Semantic/Meaningful'
      },
      color: '#8e44ad'
    }
  };

  const CompassRose = ({ operator, selectedDirection, onDirectionChange, magnitude, onMagnitudeChange }) => {
    const centerSize = 60;
    const compassSize = 140;
    const directionSize = 30;
    
    const directions = [
      { id: 'north', angle: -90, x: 0, y: -55 },
      { id: 'east', angle: 0, x: 55, y: 0 },
      { id: 'south', angle: 90, x: 0, y: 55 },
      { id: 'west', angle: 180, x: -55, y: 0 }
    ];

    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        gap: '12px',
        padding: '16px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '8px' }}>
          <div style={{ fontSize: '24px', marginBottom: '4px' }}>{operator.icon}</div>
          <div style={{ fontSize: '14px', fontWeight: '600', color: operator.color }}>
            {operator.name}
          </div>
          <div style={{ fontSize: '11px', color: '#666', lineHeight: '1.3' }}>
            {operator.description}
          </div>
        </div>

        {/* Compass Rose SVG */}
        <div style={{ position: 'relative', width: compassSize, height: compassSize }}>
          <svg width={compassSize} height={compassSize} style={{ position: 'absolute' }}>
            {/* Background circle */}
            <circle 
              cx={compassSize/2} 
              cy={compassSize/2} 
              r={compassSize/2 - 5} 
              fill="#f8f9fa" 
              stroke="#e9ecef" 
              strokeWidth="2"
            />
            
            {/* Center circle */}
            <circle 
              cx={compassSize/2} 
              cy={compassSize/2} 
              r={centerSize/2} 
              fill={selectedDirection ? operator.color : '#dee2e6'}
              stroke="white" 
              strokeWidth="3"
              style={{ cursor: 'pointer' }}
              onClick={() => onDirectionChange('')}
            />

            {/* Direction indicators */}
            {directions.map(dir => (
              <g key={dir.id}>
                <circle
                  cx={compassSize/2 + dir.x}
                  cy={compassSize/2 + dir.y}
                  r={directionSize/2}
                  fill={selectedDirection === dir.id ? operator.color : '#e9ecef'}
                  stroke={selectedDirection === dir.id ? 'white' : '#dee2e6'}
                  strokeWidth="2"
                  style={{ cursor: 'pointer' }}
                  onClick={() => onDirectionChange(dir.id)}
                />
                
                {/* Direction arrows */}
                <polygon
                  points={`${compassSize/2 + dir.x},${compassSize/2 + dir.y - 8} ${compassSize/2 + dir.x - 6},${compassSize/2 + dir.y + 4} ${compassSize/2 + dir.x + 6},${compassSize/2 + dir.y + 4}`}
                  fill="white"
                  transform={`rotate(${dir.angle}, ${compassSize/2 + dir.x}, ${compassSize/2 + dir.y})`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onDirectionChange(dir.id)}
                />
              </g>
            ))}

            {/* Magnitude indicator */}
            {selectedDirection && (
              <circle
                cx={compassSize/2}
                cy={compassSize/2}
                r={20 + (magnitude * 25)}
                fill="none"
                stroke={operator.color}
                strokeWidth="2"
                strokeDasharray="4,2"
                opacity="0.6"
              />
            )}
          </svg>
        </div>

        {/* Direction label */}
        {selectedDirection && (
          <div style={{ 
            textAlign: 'center', 
            fontSize: '12px', 
            color: operator.color,
            fontWeight: '500',
            maxWidth: '180px',
            lineHeight: '1.3'
          }}>
            {operator.axes[selectedDirection]}
          </div>
        )}

        {/* Magnitude slider */}
        {selectedDirection && (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '12px', color: '#666', textAlign: 'center' }}>
              Magnitude: {magnitude.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={magnitude}
              onChange={(e) => onMagnitudeChange(parseFloat(e.target.value))}
              style={{
                width: '100%',
                accentColor: operator.color
              }}
            />
          </div>
        )}
      </div>
    );
  };

  const handleApplyTransformation = useCallback(() => {
    if (!selectedOperator || !operatorDirection) return;

    const operator = povmOperators[selectedOperator];
    const transformationPass = {
      id: Date.now(),
      operator: selectedOperator,
      operatorName: operator.name,
      direction: operatorDirection,
      directionLabel: operator.axes[operatorDirection],
      magnitude: operatorMagnitude,
      timestamp: new Date().toISOString()
    };

    setTransformationPasses(prev => [...prev, transformationPass]);
    
    // Apply the transformation
    onTransformationApply({
      type: 'povm_compass',
      operator: selectedOperator,
      direction: operatorDirection,
      magnitude: operatorMagnitude,
      passHistory: [...transformationPasses, transformationPass]
    });

    // Reset for next pass
    setOperatorDirection('');
    setOperatorMagnitude(0.5);
  }, [selectedOperator, operatorDirection, operatorMagnitude, transformationPasses, onTransformationApply]);

  const handleClearPasses = useCallback(() => {
    setTransformationPasses([]);
    setOperatorDirection('');
    setOperatorMagnitude(0.5);
  }, []);

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '16px',
      padding: '16px',
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      border: '1px solid #e9ecef'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ 
          margin: '0 0 8px 0', 
          color: '#2c3e50',
          fontSize: '18px',
          fontWeight: '600'
        }}>
          🧭 Compass Rose POVM Navigation
        </h3>
        <p style={{ 
          margin: '0', 
          fontSize: '13px', 
          color: '#666',
          lineHeight: '1.4'
        }}>
          Navigate semantic space with precision. Apply multiple transformation passes before final projection.
        </p>
      </div>

      {/* Operator Selection */}
      <div>
        <label style={{ 
          display: 'block', 
          fontSize: '14px', 
          fontWeight: '500', 
          color: '#2c3e50',
          marginBottom: '8px'
        }}>
          Select POVM Operator:
        </label>
        <select
          value={selectedOperator}
          onChange={(e) => {
            setSelectedOperator(e.target.value);
            setOperatorDirection('');
            setOperatorMagnitude(0.5);
          }}
          style={{
            width: '100%',
            padding: '8px',
            borderRadius: '6px',
            border: '1px solid #ddd',
            fontSize: '14px'
          }}
        >
          {Object.entries(povmOperators).map(([key, operator]) => (
            <option key={key} value={key}>
              {operator.icon} {operator.name}
            </option>
          ))}
        </select>
      </div>

      {/* Compass Rose */}
      <CompassRose
        operator={povmOperators[selectedOperator]}
        selectedDirection={operatorDirection}
        onDirectionChange={setOperatorDirection}
        magnitude={operatorMagnitude}
        onMagnitudeChange={setOperatorMagnitude}
      />

      {/* Apply Transformation Button */}
      <button
        onClick={handleApplyTransformation}
        disabled={!operatorDirection || isTransforming}
        style={{
          padding: '12px 16px',
          backgroundColor: operatorDirection ? povmOperators[selectedOperator].color : '#dee2e6',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: operatorDirection && !isTransforming ? 'pointer' : 'not-allowed',
          opacity: operatorDirection && !isTransforming ? 1 : 0.6,
          transition: 'all 0.2s'
        }}
      >
        {isTransforming ? '🔄 Applying...' : `Apply ${povmOperators[selectedOperator].name}`}
      </button>

      {/* Transformation Passes History */}
      {transformationPasses.length > 0 && (
        <div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '8px'
          }}>
            <label style={{ 
              fontSize: '14px', 
              fontWeight: '500', 
              color: '#2c3e50'
            }}>
              Transformation Passes ({transformationPasses.length}):
            </label>
            <button
              onClick={() => setShowPassHistory(!showPassHistory)}
              style={{
                padding: '4px 8px',
                backgroundColor: 'transparent',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              {showPassHistory ? 'Hide' : 'Show'}
            </button>
          </div>

          {showPassHistory && (
            <div style={{
              maxHeight: '200px',
              overflowY: 'auto',
              backgroundColor: 'white',
              border: '1px solid #e9ecef',
              borderRadius: '6px',
              padding: '8px'
            }}>
              {transformationPasses.map((pass, index) => (
                <div key={pass.id} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '6px 8px',
                  backgroundColor: index % 2 === 0 ? '#f8f9fa' : 'white',
                  borderRadius: '4px',
                  fontSize: '12px',
                  marginBottom: '2px'
                }}>
                  <span>
                    <strong>{pass.operatorName}</strong> → {pass.directionLabel}
                  </span>
                  <span style={{ color: '#666' }}>
                    {pass.magnitude.toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleClearPasses}
            style={{
              width: '100%',
              marginTop: '8px',
              padding: '8px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer'
            }}
          >
            Clear All Passes
          </button>
        </div>
      )}
    </div>
  );
};

export default CompassRosePOVM;