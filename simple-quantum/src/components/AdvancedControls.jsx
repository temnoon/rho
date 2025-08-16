import React from 'react';

const AdvancedControls = ({
  showAdvanced,
  setShowAdvanced,
  advancedParams,
  setAdvancedParams
}) => {
  const resetToDefaults = () => {
    setAdvancedParams({
      strength: 0.7,
      creativity: 0.8,
      preservation: 0.8,
      complexity: 0.5,
      temperature: 0.3,
      language: ''
    });
  };

  return (
    <div style={{
      padding: '20px 30px',
      borderTop: '1px solid #e9ecef',
      background: 'white'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            background: 'none',
            border: 'none',
            color: '#667eea',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <span>{showAdvanced ? '🔼' : '🔽'}</span>
          Advanced Controls
        </button>
      </div>

      {/* Advanced Controls Panel */}
      {showAdvanced && (
        <div style={{
          marginTop: '20px',
          padding: '25px',
          background: '#f8f9fa',
          borderRadius: '12px',
          border: '1px solid #e9ecef'
        }}>
          <h4 style={{ margin: '0 0 20px 0', color: '#495057' }}>
            ⚙️ Advanced Quantum Parameters
          </h4>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
            
            {/* Transformation Strength */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
                🔥 Transformation Strength: {(advancedParams.strength * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={advancedParams.strength}
                onChange={(e) => setAdvancedParams(prev => ({ ...prev, strength: parseFloat(e.target.value) }))}
                style={{ width: '100%', marginBottom: '5px' }}
              />
              <div style={{ fontSize: '12px', color: '#6c757d' }}>
                How dramatically to transform the text
              </div>
            </div>

            {/* Creativity Level */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
                🎨 Creativity Level: {(advancedParams.creativity * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={advancedParams.creativity}
                onChange={(e) => setAdvancedParams(prev => ({ ...prev, creativity: parseFloat(e.target.value) }))}
                style={{ width: '100%', marginBottom: '5px' }}
              />
              <div style={{ fontSize: '12px', color: '#6c757d' }}>
                Level of creative interpretation
              </div>
            </div>

            {/* Preservation Level */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
                🛡️ Meaning Preservation: {(advancedParams.preservation * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={advancedParams.preservation}
                onChange={(e) => setAdvancedParams(prev => ({ ...prev, preservation: parseFloat(e.target.value) }))}
                style={{ width: '100%', marginBottom: '5px' }}
              />
              <div style={{ fontSize: '12px', color: '#6c757d' }}>
                How much original meaning to preserve
              </div>
            </div>

            {/* Complexity Target */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
                🧠 Complexity Target: {(advancedParams.complexity * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={advancedParams.complexity}
                onChange={(e) => setAdvancedParams(prev => ({ ...prev, complexity: parseFloat(e.target.value) }))}
                style={{ width: '100%', marginBottom: '5px' }}
              />
              <div style={{ fontSize: '12px', color: '#6c757d' }}>
                Desired linguistic complexity level
              </div>
            </div>

            {/* Language Selection */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
                🌍 Target Language
              </label>
              <select
                value={advancedParams.language}
                onChange={(e) => setAdvancedParams(prev => ({ ...prev, language: e.target.value }))}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #dee2e6',
                  borderRadius: '6px',
                  background: 'white'
                }}
              >
                <option value="">Same Language</option>
                <option value="Translate to French">🇫🇷 French</option>
                <option value="Translate to Spanish">🇪🇸 Spanish</option>
                <option value="Translate to German">🇩🇪 German</option>
                <option value="Translate to Italian">🇮🇹 Italian</option>
                <option value="Translate to Portuguese">🇵🇹 Portuguese</option>
                <option value="Translate to Japanese">🇯🇵 Japanese</option>
                <option value="Translate to Chinese">🇨🇳 Chinese</option>
                <option value="Translate to Russian">🇷🇺 Russian</option>
              </select>
            </div>

            {/* Reset Button */}
            <div style={{ display: 'flex', alignItems: 'end' }}>
              <button
                onClick={resetToDefaults}
                style={{
                  padding: '12px 20px',
                  background: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                🔄 Reset to Defaults
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedControls;