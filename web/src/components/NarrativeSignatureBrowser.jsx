import React, { useState, useEffect } from 'react';

const NarrativeSignatureBrowser = ({ onSignatureSelect, currentText = "" }) => {
  const [taxonomy, setTaxonomy] = useState({});
  const [savedSignatures, setSavedSignatures] = useState([]);
  const [activeTab, setActiveTab] = useState('builder');
  const [customSignature, setCustomSignature] = useState({
    name: '',
    namespace: {},
    style: {},
    persona: {},
    description: ''
  });
  const [extractionText, setExtractionText] = useState('');
  const [extractionName, setExtractionName] = useState('');
  const [loading, setLoading] = useState(false);
  const [balancingResult, setBalancingResult] = useState(null);

  const safeFetch = async (url, options = {}) => {
    const response = await fetch(`http://localhost:8192${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  };

  useEffect(() => {
    loadTaxonomy();
    loadSavedSignatures();
  }, []);

  const loadTaxonomy = async () => {
    try {
      const response = await safeFetch('/narrative-attributes/taxonomy');
      const data = await response.json();
      setTaxonomy(data.taxonomy || {});
    } catch (error) {
      console.error('Failed to load taxonomy:', error);
    }
  };

  const loadSavedSignatures = async () => {
    try {
      const response = await safeFetch('/narrative-attributes/signatures');
      const data = await response.json();
      setSavedSignatures(data.signatures || []);
    } catch (error) {
      console.error('Failed to load signatures:', error);
    }
  };

  const handleAttributeChange = (category, attribute, value) => {
    const numValue = parseFloat(value) || 0;
    setCustomSignature(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [attribute]: numValue > 0.05 ? numValue : undefined
      }
    }));
    
    // Clean up undefined values
    setCustomSignature(prev => ({
      ...prev,
      [category]: Object.fromEntries(
        Object.entries(prev[category]).filter(([_, v]) => v !== undefined && v > 0.05)
      )
    }));
  };

  const extractSignature = async () => {
    if (!extractionText.trim()) return;
    
    setLoading(true);
    try {
      const response = await safeFetch('/narrative-attributes/extract', {
        method: 'POST',
        body: JSON.stringify({
          text: extractionText,
          signature_name: extractionName || undefined,
          use_llm: true
        })
      });
      const data = await response.json();
      
      if (data.success) {
        setCustomSignature({
          name: extractionName || '',
          namespace: data.signature.namespace || {},
          style: data.signature.style || {},
          persona: data.signature.persona || {},
          description: `Extracted from: ${extractionText.slice(0, 100)}...`
        });
        
        if (extractionName) {
          loadSavedSignatures(); // Refresh list
        }
      }
    } catch (error) {
      console.error('Failed to extract signature:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveCustomSignature = async () => {
    if (!customSignature.name) return;
    
    setLoading(true);
    try {
      const response = await safeFetch('/narrative-attributes/create', {
        method: 'POST',
        body: JSON.stringify(customSignature)
      });
      const data = await response.json();
      
      if (data.success) {
        loadSavedSignatures();
        alert(`Saved signature: ${customSignature.name}`);
      }
    } catch (error) {
      console.error('Failed to save signature:', error);
    } finally {
      setLoading(false);
    }
  };

  const checkBalance = async () => {
    setLoading(true);
    try {
      const response = await safeFetch('/narrative-attributes/balance', {
        method: 'POST',
        body: JSON.stringify(customSignature)
      });
      const data = await response.json();
      
      if (data.success) {
        setBalancingResult(data);
      }
    } catch (error) {
      console.error('Failed to check balance:', error);
    } finally {
      setLoading(false);
    }
  };

  const applyBalancing = () => {
    if (balancingResult?.balanced_signature) {
      setCustomSignature(prev => ({
        ...prev,
        namespace: balancingResult.balanced_signature.namespace,
        style: balancingResult.balanced_signature.style,
        persona: balancingResult.balanced_signature.persona
      }));
      setBalancingResult(null);
    }
  };

  const renderAttributeSlider = (category, attribute, description) => {
    const value = customSignature[category][attribute] || 0;
    
    return (
      <div key={attribute} style={{ marginBottom: '12px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '4px'
        }}>
          <label style={{ fontSize: '13px', fontWeight: '500' }}>
            {attribute.replace(/_/g, ' ')}
          </label>
          <span style={{ 
            fontSize: '12px', 
            color: value > 0.7 ? '#ff6b35' : value > 0.3 ? '#ffa500' : '#666',
            fontWeight: '600'
          }}>
            {value.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={value}
          onChange={(e) => handleAttributeChange(category, attribute, e.target.value)}
          style={{ 
            width: '100%',
            height: '4px',
            background: `linear-gradient(to right, #ddd ${value * 100}%, #f0f0f0 ${value * 100}%)`
          }}
        />
        <div style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>
          {description}
        </div>
      </div>
    );
  };

  const renderAttributeCategory = (categoryName, attributes) => (
    <div style={{ marginBottom: '20px' }}>
      <h4 style={{ 
        margin: '0 0 12px 0',
        color: '#333',
        fontSize: '16px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        {categoryName}
      </h4>
      <div style={{ 
        border: '1px solid #e0e0e0',
        borderRadius: '6px',
        padding: '12px',
        backgroundColor: '#fafafa'
      }}>
        {Object.entries(attributes).map(([attr, desc]) => 
          <React.Fragment key={attr}>{renderAttributeSlider(categoryName, attr, desc)}</React.Fragment>
        )}
      </div>
    </div>
  );

  const renderSignatureBuilder = () => (
    <div>
      <div style={{ marginBottom: '20px', padding: '12px', backgroundColor: '#f0f8ff', borderRadius: '6px' }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#2196f3' }}>🎭 Signature Builder</h3>
        <p style={{ margin: 0, fontSize: '14px', color: '#666' }}>
          Create custom narrative signatures by adjusting attributes across three dimensions
        </p>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500' }}>
          Signature Name
        </label>
        <input
          type="text"
          value={customSignature.name}
          onChange={(e) => setCustomSignature(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Enter signature name..."
          style={{ 
            width: '100%', 
            padding: '8px', 
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500' }}>
          Description
        </label>
        <textarea
          value={customSignature.description}
          onChange={(e) => setCustomSignature(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Describe this signature..."
          rows={2}
          style={{ 
            width: '100%', 
            padding: '8px', 
            border: '1px solid #ddd',
            borderRadius: '4px',
            resize: 'vertical'
          }}
        />
      </div>

      {/* Render attribute categories */}
      {Object.entries(taxonomy).map(([category, attributes]) => 
        <div key={category}>{renderAttributeCategory(category, attributes)}</div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '8px', marginTop: '20px' }}>
        <button
          onClick={checkBalance}
          disabled={loading}
          style={{
            padding: '10px 16px',
            backgroundColor: '#ffa500',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          🤖 Check Balance
        </button>
        <button
          onClick={saveCustomSignature}
          disabled={loading || !customSignature.name}
          style={{
            padding: '10px 16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            opacity: !customSignature.name ? 0.6 : 1
          }}
        >
          💾 Save Signature
        </button>
        <button
          onClick={() => onSignatureSelect && onSignatureSelect(customSignature)}
          style={{
            padding: '10px 16px',
            backgroundColor: '#2196f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          🚀 Apply to Text
        </button>
      </div>

      {/* Balancing suggestions */}
      {balancingResult && (
        <div style={{ 
          marginTop: '20px',
          padding: '12px',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffeaa7',
          borderRadius: '6px'
        }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#856404' }}>
            🤖 AI Balancing Suggestions
          </h4>
          
          {balancingResult.issues && balancingResult.issues.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <strong>Issues Found:</strong>
              <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                {balancingResult.issues.map((issue, i) => (
                  <li key={i} style={{ fontSize: '13px', color: '#856404' }}>
                    {issue.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {balancingResult.suggestions && balancingResult.suggestions.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <strong>Suggestions:</strong>
              <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                {balancingResult.suggestions.map((suggestion, i) => (
                  <li key={i} style={{ fontSize: '13px', color: '#856404' }}>
                    {suggestion.attribute}: {suggestion.current_weight.toFixed(2)} → {suggestion.suggested_weight.toFixed(2)}
                    <br />
                    <em>{suggestion.reason}</em>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            onClick={applyBalancing}
            style={{
              padding: '6px 12px',
              backgroundColor: '#856404',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            Apply Suggestions
          </button>
        </div>
      )}
    </div>
  );

  const renderExtractor = () => (
    <div>
      <div style={{ marginBottom: '20px', padding: '12px', backgroundColor: '#f0fff0', borderRadius: '6px' }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#4CAF50' }}>🔍 Signature Extractor</h3>
        <p style={{ margin: 0, fontSize: '14px', color: '#666' }}>
          Extract narrative signatures from existing text using AI analysis
        </p>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500' }}>
          Text to Analyze
        </label>
        <textarea
          value={extractionText}
          onChange={(e) => setExtractionText(e.target.value)}
          placeholder="Paste text here to extract its narrative signature..."
          rows={6}
          style={{ 
            width: '100%', 
            padding: '8px', 
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '13px',
            lineHeight: '1.4'
          }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500' }}>
          Signature Name (optional)
        </label>
        <input
          type="text"
          value={extractionName}
          onChange={(e) => setExtractionName(e.target.value)}
          placeholder="Name to save this signature..."
          style={{ 
            width: '100%', 
            padding: '8px', 
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
      </div>

      <button
        onClick={extractSignature}
        disabled={loading || !extractionText.trim()}
        style={{
          padding: '10px 16px',
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '14px',
          opacity: !extractionText.trim() ? 0.6 : 1
        }}
      >
        {loading ? '🔄 Analyzing...' : '🔍 Extract Signature'}
      </button>

      {currentText && (
        <div style={{ marginTop: '20px' }}>
          <button
            onClick={() => setExtractionText(currentText)}
            style={{
              padding: '8px 12px',
              backgroundColor: '#2196f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            📋 Use Current Text
          </button>
        </div>
      )}
    </div>
  );

  const renderSavedSignatures = () => (
    <div>
      <div style={{ marginBottom: '20px', padding: '12px', backgroundColor: '#fff8f0', borderRadius: '6px' }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#ff9800' }}>📚 Saved Signatures</h3>
        <p style={{ margin: 0, fontSize: '14px', color: '#666' }}>
          Browse and apply previously saved narrative signatures
        </p>
      </div>

      <div style={{ display: 'grid', gap: '12px' }}>
        {savedSignatures.map((signature, index) => (
          <div key={index} style={{
            border: '1px solid #e0e0e0',
            borderRadius: '6px',
            padding: '12px',
            backgroundColor: '#fafafa'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <h4 style={{ margin: '0 0 4px 0', color: '#333' }}>
                  {signature.name}
                </h4>
                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                  Namespace: {signature.namespace_attributes} • 
                  Style: {signature.style_attributes} • 
                  Persona: {signature.persona_attributes}
                </div>
                {signature.metadata?.description && (
                  <div style={{ fontSize: '12px', color: '#888', fontStyle: 'italic' }}>
                    {signature.metadata.description}
                  </div>
                )}
              </div>
              <button
                onClick={() => onSignatureSelect && onSignatureSelect({ name: signature.name })}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#2196f3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  marginLeft: '8px'
                }}
              >
                Apply
              </button>
            </div>
          </div>
        ))}
        
        {savedSignatures.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '40px', 
            color: '#888',
            fontStyle: 'italic'
          }}>
            No saved signatures yet. Create some using the Builder or Extractor!
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div style={{ padding: '20px', maxWidth: '800px' }}>
      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        borderBottom: '2px solid #e0e0e0',
        marginBottom: '20px'
      }}>
        {[
          { id: 'builder', label: '🎭 Builder', color: '#2196f3' },
          { id: 'extractor', label: '🔍 Extractor', color: '#4CAF50' },
          { id: 'saved', label: '📚 Saved', color: '#ff9800' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 20px',
              border: 'none',
              borderBottom: activeTab === tab.id ? `3px solid ${tab.color}` : '3px solid transparent',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: activeTab === tab.id ? '600' : '400',
              color: activeTab === tab.id ? tab.color : '#666'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'builder' && renderSignatureBuilder()}
      {activeTab === 'extractor' && renderExtractor()}
      {activeTab === 'saved' && renderSavedSignatures()}
    </div>
  );
};

export default NarrativeSignatureBrowser;