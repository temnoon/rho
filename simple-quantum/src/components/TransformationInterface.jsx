import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import CompassRosePOVM from '../CompassRosePOVM';

const TransformationInterface = ({
  inputText,
  setInputText,
  transformedText,
  isTransforming,
  transformRequest,
  setTransformRequest,
  transformationMode,
  setTransformationMode,
  quantumDistance,
  currentRhoState,
  showAdvanced,
  setShowAdvanced,
  advancedParams,
  setAdvancedParams,
  onTransform,
  onQuickTransform,
  onCompassTransformation,
  onCopyToClipboard
}) => {
  // Sample texts for quick start
  const sampleTexts = [
    "The old man walked slowly down the street, his thoughts heavy with memories of better days.",
    "Sarah discovered the hidden letter in her grandmother's attic, its yellow pages crackling with age.",
    "The storm approached the small coastal town, waves crashing against the weathered pier.",
    "In the quiet library, she found a book that seemed to glow with an inner light."
  ];

  const quickTransforms = [
    { name: '🎭 Poetic', prompt: 'make this more poetic and lyrical' },
    { name: '📚 Academic', prompt: 'rewrite in formal academic style' },
    { name: '💬 Casual', prompt: 'make this more casual and conversational' },
    { name: '🌟 Magical', prompt: 'add magical and mystical elements' },
    { name: '🔮 Mysterious', prompt: 'make this more mysterious and intriguing' },
    { name: '⚡ Dramatic', prompt: 'increase the drama and tension' }
  ];

  return (
    <div style={{
      maxWidth: window.innerWidth < 768 ? '100%' : window.innerWidth < 1200 ? '95%' : '1400px',
      margin: '0 auto',
      background: 'white',
      borderRadius: '20px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      overflow: 'hidden'
    }}>
      
      {/* Quick Start Samples */}
      {!inputText && (
        <div style={{
          padding: '30px',
          background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
          borderBottom: '1px solid #dee2e6'
        }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#495057' }}>
            ✨ Quick Start - Click a sample:
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '10px'
          }}>
            {sampleTexts.map((text, idx) => (
              <button
                key={idx}
                onClick={() => setInputText(text)}
                style={{
                  padding: '15px',
                  background: 'white',
                  border: '2px solid #e9ecef',
                  borderRadius: '12px',
                  fontSize: '14px',
                  lineHeight: '1.4',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  textAlign: 'left'
                }}
                onMouseEnter={(e) => {
                  e.target.style.borderColor = '#667eea';
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.borderColor = '#e9ecef';
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                {text}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Text Input/Output Area */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: inputText && transformedText ? 
          (window.innerWidth < 768 ? '1fr' : '1fr 1fr') : '1fr',
        height: 'calc(45vh - 200px)', // Fill screen minus header/controls
        minHeight: '300px',
        maxHeight: 'calc(45vh - 200px)',
        gap: '1px',
        backgroundColor: '#e9ecef'
      }}>
        
        {/* Input Text */}
        <div style={{
          padding: '20px',
          backgroundColor: 'white',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h3 style={{ margin: 0, color: '#495057' }}>
              📝 Your Story
            </h3>
            {inputText && (
              <button
                onClick={() => onCopyToClipboard(inputText)}
                style={{
                  padding: '6px 12px',
                  background: '#f8f9fa',
                  border: '1px solid #dee2e6',
                  borderRadius: '6px',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}
              >
                Copy Original
              </button>
            )}
          </div>
          
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Paste or type your story here..."
            style={{
              width: '100%',
              flex: '1',
              padding: '20px',
              border: '2px solid #e9ecef',
              borderRadius: '12px',
              fontSize: '16px',
              lineHeight: '1.6',
              fontFamily: 'inherit',
              resize: 'none',
              outline: 'none',
              transition: 'border-color 0.2s ease',
              boxSizing: 'border-box'
            }}
            onFocus={(e) => e.target.style.borderColor = '#667eea'}
            onBlur={(e) => e.target.style.borderColor = '#e9ecef'}
          />
          
          <div style={{
            marginTop: '10px',
            fontSize: '14px',
            color: '#6c757d',
            display: 'flex',
            justifyContent: 'space-between'
          }}>
            <span>{inputText.split(' ').filter(w => w.length > 0).length} words</span>
            {inputText && (
              <button
                onClick={() => setInputText('')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#dc3545',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Output Text */}
        {inputText && transformedText && (
          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '15px'
            }}>
              <h3 style={{ margin: 0, color: '#495057' }}>
                ✨ Quantum Transform
              </h3>
              <div style={{ display: 'flex', gap: '8px' }}>
                {quantumDistance > 0 && (
                  <span style={{
                    padding: '4px 8px',
                    background: '#e3f2fd',
                    color: '#1976d2',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}>
                    δ: {quantumDistance.toExponential(2)}
                  </span>
                )}
                <button
                  onClick={() => onCopyToClipboard(transformedText)}
                  style={{
                    padding: '6px 12px',
                    background: '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  Copy Result
                </button>
              </div>
            </div>
            
            <div style={{
              flex: '1',
              padding: '20px',
              background: '#f8f9fa',
              border: '2px solid #e9ecef',
              borderRadius: '12px',
              fontSize: '16px',
              lineHeight: '1.6',
              overflow: 'auto',
              boxSizing: 'border-box'
            }}>
              {isTransforming ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '200px',
                  color: '#6c757d'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', marginBottom: '10px' }}>🌀</div>
                    <div>Quantum transformation in progress...</div>
                  </div>
                </div>
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkMath, remarkGfm]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    // Ensure proper styling for markdown elements
                    p: ({children}) => <p style={{margin: '0.5em 0'}}>{children}</p>,
                    h1: ({children}) => <h1 style={{margin: '1em 0 0.5em 0', color: '#333'}}>{children}</h1>,
                    h2: ({children}) => <h2 style={{margin: '1em 0 0.5em 0', color: '#333'}}>{children}</h2>,
                    h3: ({children}) => <h3 style={{margin: '1em 0 0.5em 0', color: '#333'}}>{children}</h3>,
                    code: ({inline, children}) => (
                      <code style={{
                        backgroundColor: inline ? '#f1f3f4' : 'transparent',
                        padding: inline ? '2px 4px' : '0',
                        borderRadius: '3px',
                        fontFamily: 'Monaco, Consolas, monospace',
                        fontSize: '0.9em'
                      }}>
                        {children}
                      </code>
                    ),
                    blockquote: ({children}) => (
                      <blockquote style={{
                        borderLeft: '4px solid #667eea',
                        paddingLeft: '1em',
                        margin: '1em 0',
                        fontStyle: 'italic',
                        color: '#666'
                      }}>
                        {children}
                      </blockquote>
                    )
                  }}
                >
                  {transformedText}
                </ReactMarkdown>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Transform Controls */}
      <div style={{
        padding: '30px',
        background: '#f8f9fa',
        borderTop: '1px solid #e9ecef'
      }}>
        
        {/* Transformation Mode Toggle */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h4 style={{ margin: '0', color: '#495057' }}>
              {transformationMode === 'compass' ? '🧭 Compass Rose POVM Navigation' : '⚡ Quick Transforms'}
            </h4>
            <div style={{
              display: 'flex',
              backgroundColor: '#e9ecef',
              borderRadius: '20px',
              padding: '3px'
            }}>
              <button
                onClick={() => setTransformationMode('quick')}
                style={{
                  padding: '8px 16px',
                  backgroundColor: transformationMode === 'quick' ? '#667eea' : 'transparent',
                  color: transformationMode === 'quick' ? 'white' : '#666',
                  border: 'none',
                  borderRadius: '17px',
                  fontSize: '12px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                ⚡ Quick
              </button>
              <button
                onClick={() => setTransformationMode('compass')}
                style={{
                  padding: '8px 16px',
                  backgroundColor: transformationMode === 'compass' ? '#667eea' : 'transparent',
                  color: transformationMode === 'compass' ? 'white' : '#666',
                  border: 'none',
                  borderRadius: '17px',
                  fontSize: '12px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                🧭 Compass
              </button>
            </div>
          </div>

          {transformationMode === 'quick' ? (
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '10px'
            }}>
              {quickTransforms.map((transform, idx) => (
                <button
                  key={idx}
                  onClick={() => onQuickTransform(transform.prompt)}
                  disabled={!inputText || isTransforming}
                  style={{
                    padding: '12px 20px',
                    background: 'white',
                    border: '2px solid #667eea',
                    borderRadius: '25px',
                    color: '#667eea',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: inputText && !isTransforming ? 'pointer' : 'not-allowed',
                    opacity: inputText && !isTransforming ? 1 : 0.5,
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (inputText && !isTransforming) {
                      e.target.style.background = '#667eea';
                      e.target.style.color = 'white';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'white';
                    e.target.style.color = '#667eea';
                  }}
                >
                  {transform.name}
                </button>
              ))}
            </div>
          ) : (
            <CompassRosePOVM
              onTransformationApply={onCompassTransformation}
              currentRhoState={currentRhoState}
              isTransforming={isTransforming}
            />
          )}

          {/* Custom Transform */}
          <div style={{
            display: 'flex',
            gap: '12px',
            alignItems: 'stretch',
            marginTop: '20px'
          }}>
            <input
              type="text"
              value={transformRequest}
              onChange={(e) => setTransformRequest(e.target.value)}
              placeholder="Or describe your own transformation..."
              disabled={!inputText}
              style={{
                flex: 1,
                padding: '15px 20px',
                border: '2px solid #e9ecef',
                borderRadius: '12px',
                fontSize: '16px',
                outline: 'none',
                opacity: inputText ? 1 : 0.5
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && inputText && !isTransforming) {
                  onTransform();
                }
              }}
            />
            <button
              onClick={() => onTransform()}
              disabled={!inputText || isTransforming}
              style={{
                padding: '15px 30px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: '600',
                cursor: inputText && !isTransforming ? 'pointer' : 'not-allowed',
                opacity: inputText && !isTransforming ? 1 : 0.5,
                minWidth: '120px'
              }}
            >
              {isTransforming ? '🌀' : '✨ Transform'}
            </button>
          </div>
        </div>

        {/* Advanced Controls - Integrated */}
        <div style={{
          borderTop: '1px solid #e9ecef',
          background: 'white'
        }}>
          <div style={{
            padding: '20px 30px',
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
              margin: '0 30px 30px 30px',
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
                    onClick={() => setAdvancedParams({
                      strength: 0.7,
                      creativity: 0.8,
                      preservation: 0.8,
                      complexity: 0.5,
                      temperature: 0.3,
                      language: ''
                    })}
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
      </div>
    </div>
  );
};

export default TransformationInterface;