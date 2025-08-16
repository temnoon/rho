import React, { useEffect, useRef, useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { UnifiedWorkbench } from "./components/UnifiedWorkbench.jsx";
import AttributeBrowser from "./AttributeBrowser.jsx";
import NarrativeExplorer from "./components/NarrativeExplorer.jsx";
import NarrativeTab from "./components/NarrativeTab.jsx";
import { BookReaderTab } from "./components/BookReaderTab.jsx";
import { DatabaseTab } from "./components/DatabaseTab.jsx";
import { DualMatrixTab } from "./components/DualMatrixTab.jsx";
import { QueryTab } from "./components/QueryTab.jsx";
import { ChannelObservatoryTab } from "./components/ChannelObservatoryTab.jsx";
import { LiminalSpaceTab } from "./components/LiminalSpaceTab.jsx";
import { NarrativeDistillationStudio } from "./components/NarrativeDistillationStudio.jsx";
import MatrixArchaeologyStudio from "./components/MatrixArchaeologyStudio.jsx";

/**
 * Enhanced Rho Narrative Humanizer Workbench
 * 
 * Advanced interface for managing density matrices, analyzing narratives,
 * and understanding post-lexical consciousness through transparent POVM operations.
 */

const API = "http://localhost:8192";

function apiUrl(path) {
  if (!path) return API;
  const base = API.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path.replace(/^\/+/, "") : path;
  return `${base}/${suffix}`;
}

async function safeFetch(path, opts = {}) {
  const url = apiUrl(path);
  try {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let bodyText = "<no body>";
      try {
        bodyText = await res.text();
      } catch (e) {
        bodyText = `<unable to read body: ${String(e)}>`;
      }
      const err = new Error(`${url} returned ${res.status} ${res.statusText} - ${bodyText}`);
      err.status = res.status;
      throw err;
    }
    return res;
  } catch (err) {
    console.error("[RHO] Network error for", url, err);
    throw err;
  }
}

function formatNumber(n, digits = 4) {
  if (!Number.isFinite(n)) return "NaN";
  return Number(n).toFixed(digits);
}

// Enhanced Matrix Visualization - shows actual eigenvalues as heatmap
function MatrixVisualization({ title, eigs = [], purity = 0, entropy = 0, size = 300 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !eigs.length) return;
    
    const ctx = canvas.getContext("2d");
    canvas.width = size;
    canvas.height = size;
    ctx.clearRect(0, 0, size, size);

    // Create visualization based on actual eigenvalue spectrum
    const gridSize = 8; // 8x8 grid for 64-dimensional space
    const cellSize = size / gridSize;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const idx = i * gridSize + j;
        const eigVal = eigs[idx] || 0;
        
        // Color intensity based on eigenvalue
        const intensity = Math.min(255, Math.floor(eigVal * 1000)); // Scale for visibility
        const r = intensity;
        const g = Math.floor(intensity * 0.7);
        const b = Math.floor(intensity * 0.3);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(i * cellSize, j * cellSize, cellSize - 1, cellSize - 1);
        
        // Add eigenvalue text for top components
        if (idx < 8 && eigVal > 0.01) {
          ctx.fillStyle = 'white';
          ctx.font = '8px monospace';
          ctx.fillText(eigVal.toFixed(3), i * cellSize + 2, j * cellSize + 10);
        }
      }
    }
  }, [eigs, size]);

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>{title}</div>
      <canvas 
        ref={canvasRef} 
        width={size} 
        height={size}
        style={{ border: '1px solid #ccc', borderRadius: 4 }}
      />
      <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
        <div>Purity: {formatNumber(purity, 3)}</div>
        <div>Entropy: {formatNumber(entropy, 3)}</div>
        <div>Top λ: {eigs[0] ? formatNumber(eigs[0], 3) : 'N/A'}</div>
      </div>
    </div>
  );
}





// Global Rho Status Header
function GlobalRhoStatusHeader() {
  const [globalStatus, setGlobalStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchGlobalStatus = useCallback(async () => {
    try {
      const res = await safeFetch('/rho/global/status');
      const data = await res.json();
      setGlobalStatus(data);
    } catch (err) {
      console.error('Failed to fetch global rho status:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGlobalStatus();
    const interval = setInterval(fetchGlobalStatus, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, [fetchGlobalStatus]);

  if (loading) {
    return (
      <div style={{ background: '#e3f2fd', padding: '8px 20px', fontSize: 14, color: '#1565c0' }}>
        🧠 Loading global rho consciousness...
      </div>
    );
  }

  if (!globalStatus) {
    return (
      <div style={{ background: '#ffebee', padding: '8px 20px', fontSize: 14, color: '#c62828' }}>
        ⚠️ Global rho consciousness not available
      </div>
    );
  }

  const { meta, matrix_state, processing_queue } = globalStatus;
  const statusColor = meta.books_processed > 0 ? '#e8f5e8' : '#fff3cd';
  const textColor = meta.books_processed > 0 ? '#2e7d32' : '#8c6c00';

  return (
    <div style={{ 
      background: statusColor, 
      padding: '10px 20px', 
      fontSize: 14, 
      color: textColor,
      borderBottom: '1px solid #ddd'
    }}>
      <div style={{ display: 'flex', gap: '30px', alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 600 }}>
          🧠 Global Literary Consciousness
        </div>
        <div>
          📚 <strong>{meta.books_processed}</strong> books processed
        </div>
        <div>
          📝 <strong>{meta.total_chunks?.toLocaleString() || 0}</strong> chunks
        </div>
        <div>
          🔢 <strong>{meta.total_tokens?.toLocaleString() || 0}</strong> tokens
        </div>
        <div>
          💎 Purity: <strong>{matrix_state.purity?.toFixed(3) || 'N/A'}</strong>
        </div>
        <div>
          📊 Entropy: <strong>{matrix_state.entropy?.toFixed(2) || 'N/A'}</strong>
        </div>
        <div>
          λ₁: <strong>{matrix_state.eigs?.[0]?.toFixed(3) || 'N/A'}</strong>
        </div>
        {processing_queue.processing > 0 && (
          <div style={{ color: '#1976d2' }}>
            ⚡ Processing: <strong>{processing_queue.processing}</strong>
          </div>
        )}
        {processing_queue.queued > 0 && (
          <div style={{ color: '#7b1fa2' }}>
            📋 Queued: <strong>{processing_queue.queued}</strong>
          </div>
        )}
      </div>
      {meta.book_titles && meta.book_titles.length > 0 && (
        <div style={{ marginTop: 6, fontSize: 12, opacity: 0.8 }}>
          Latest books: {meta.book_titles.slice(-3).join(' • ')}
          {meta.book_titles.length > 3 && ` (+${meta.book_titles.length - 3} more)`}
        </div>
      )}
    </div>
  );
}

// Shared Narrative Context (persists across tabs)
const SharedNarrativeContext = React.createContext();

function SharedNarrativeProvider({ children }) {
  const [sharedNarrative, setSharedNarrative] = useState({
    text: "The brave knight rode through the dark forest, his heart filled with determination and courage. Ancient trees whispered secrets of forgotten adventures, while shadows danced mysteriously around his gleaming armor.",
    extractedAttributes: null,
    customAttributes: [], // User-created attributes
    appliedAdjustments: { persona: 0, namespace: 0, style: 0 },
    transformedText: "",
    rhoStates: {
      base: null,
      adjusted: null
    },
    lastUpdated: Date.now()
  });

  const updateSharedNarrative = (updates) => {
    setSharedNarrative(prev => ({
      ...prev,
      ...updates,
      lastUpdated: Date.now()
    }));
  };

  return (
    <SharedNarrativeContext.Provider value={{ sharedNarrative, updateSharedNarrative }}>
      {children}
    </SharedNarrativeContext.Provider>
  );
}

// Hook to use shared narrative
const useSharedNarrative = () => {
  const context = React.useContext(SharedNarrativeContext);
  if (!context) {
    throw new Error('useSharedNarrative must be used within SharedNarrativeProvider');
  }
  return context;
};

// Main Application
function RhoWorkbench() {
  return (
    <SharedNarrativeProvider>
      <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif' }}>
        <GlobalRhoStatusHeader />
        <div style={{ borderBottom: '1px solid #ddd', background: '#f8f9fa', padding: '0 20px' }}>
          <h1 style={{ margin: 0, padding: '15px 0', fontSize: 22 }}>
            Rho Narrative Humanizer — Unified Quantum Workbench
          </h1>
        </div>
        <UnifiedWorkbench />
      </div>
    </SharedNarrativeProvider>
  );
}

// Attribute Extraction, Removal, and Adjustment Component
function AttributeAdjustmentTab() {
  const { sharedNarrative, updateSharedNarrative } = useSharedNarrative();
  const [extractedAttributes, setExtractedAttributes] = useState(null);
  const [allAttributes, setAllAttributes] = useState(null);
  const [adjustments, setAdjustments] = useState({});
  const [adjustedMatrix, setAdjustedMatrix] = useState(null);
  const [regeneratedText, setRegeneratedText] = useState("");
  const [sentencePreviews, setSentencePreviews] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [baseRhoId, setBaseRhoId] = useState(null);
  const [autoMode, setAutoMode] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('persona');
  const [showSentencePreview, setShowSentencePreview] = useState(false);
  
  // Custom attribute creation
  const [newAttributeName, setNewAttributeName] = useState('');
  const [newAttributeStrength, setNewAttributeStrength] = useState(0);
  const [showAttributeCreator, setShowAttributeCreator] = useState(false);
  
  // Use shared narrative text
  const text = sharedNarrative.text;
  const setText = (newText) => updateSharedNarrative({ text: newText });

  // Load all available attributes
  const loadAllAttributes = async () => {
    try {
      const res = await safeFetch('/attributes/list');
      const data = await res.json();
      setAllAttributes(data);
    } catch (err) {
      console.error('Failed to load attributes:', err);
    }
  };

  // Initialize adjustments object when all attributes are loaded
  useEffect(() => {
    if (allAttributes && allAttributes.categories && Object.keys(adjustments).length === 0) {
      const initialAdjustments = {};
      Object.values(allAttributes.categories).flat().forEach(attr => {
        initialAdjustments[attr.name] = 0;
      });
      setAdjustments(initialAdjustments);
    }
  }, [allAttributes]);

  const extractAttributes = async () => {
    if (!text.trim()) {
      setStatusMsg("Enter narrative text first.");
      return;
    }
    try {
      setLoading(true);
      setStatusMsg("Creating base matrix...");
      
      // First create a base matrix by reading the text
      const initRes = await safeFetch('/rho/init', { method: 'POST' });
      const initData = await initRes.json();
      const rhoId = initData.rho_id;
      
      const readRes = await safeFetch(`/rho/${rhoId}/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: text, alpha: 0.3 })
      });
      await readRes.json(); // Matrix is now in STATE
      setBaseRhoId(rhoId);
      
      setStatusMsg("Extracting attributes...");
      const res = await safeFetch('/attributes/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      setExtractedAttributes(data);
      setStatusMsg("Attributes extracted");
    } catch (err) {
      console.error('Failed to extract attributes:', err);
      setStatusMsg("Extraction failed");
    } finally {
      setLoading(false);
      setTimeout(() => setStatusMsg(""), 1500);
    }
  };

  const previewSentences = async () => {
    if (!text.trim()) {
      setStatusMsg("Enter text first.");
      return;
    }

    // Get only non-zero adjustments
    const nonZeroAdjustments = Object.fromEntries(
      Object.entries(adjustments).filter(([_, value]) => Math.abs(value) > 0.01)
    );

    if (Object.keys(nonZeroAdjustments).length === 0) {
      setStatusMsg("Adjust some attributes first.");
      return;
    }

    try {
      setLoading(true);
      setStatusMsg("Generating sentence previews...");
      const res = await safeFetch('/attributes/preview_sentences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          attribute_adjustments: nonZeroAdjustments,
          preview_sentences: 3
        })
      });
      const data = await res.json();
      setSentencePreviews(data);
      setShowSentencePreview(true);
      setStatusMsg("Preview generated");
    } catch (err) {
      console.error('Failed to preview sentences:', err);
      setStatusMsg("Preview failed");
    } finally {
      setLoading(false);
      setTimeout(() => setStatusMsg(""), 1500);
    }
  };

  const adjustMatrix = async () => {
    if (!extractedAttributes || !baseRhoId) {
      setStatusMsg("Extract attributes first.");
      return;
    }
    try {
      setLoading(true);
      setStatusMsg("Adjusting matrix...");
      const res = await safeFetch('/attributes/adjust_matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          persona_strength: adjustments.persona,
          namespace_strength: adjustments.namespace,
          style_strength: adjustments.style,
          base_rho_id: baseRhoId
        })
      });
      const data = await res.json();
      setAdjustedMatrix(data);
      setStatusMsg("Matrix adjusted");
    } catch (err) {
      console.error('Failed to adjust matrix:', err);
      setStatusMsg("Matrix adjustment failed");
    } finally {
      setLoading(false);
      setTimeout(() => setStatusMsg(""), 1500);
    }
  };

  const regenerateNarrative = async () => {
    if (!adjustedMatrix || !adjustedMatrix.adjusted_rho_id) {
      setStatusMsg("Adjust matrix first.");
      return;
    }
    try {
      setLoading(true);
      setStatusMsg("Applying rho transformations...");
      
      // Determine the most appropriate transformation based on extracted attributes
      let transformationName = "earth_to_mars"; // Default transformation
      let strength = 1.0;
      
      // Select transformation based on dominant attributes
      if (extractedAttributes) {
        const scores = extractedAttributes.scores || {};
        
        // Find the highest scoring attribute categories to determine transformation
        const mysticalScore = scores.mystical || 0;
        const historicalScore = scores.historical || 0;
        const psychologicalScore = scores.psychological || 0;
        const playfulnessScore = scores.playfulness || 0;
        
        if (mysticalScore > 0.02) {
          transformationName = "realistic_to_fantasy";
          strength = Math.min(mysticalScore * 50, 2.0);
        } else if (historicalScore > 0.02) {
          transformationName = "modern_to_historical";
          strength = Math.min(historicalScore * 50, 2.0);
        } else if (playfulnessScore > 0.03) {
          transformationName = "serious_to_humorous";
          strength = Math.min(playfulnessScore * 30, 1.5);
        } else if (psychologicalScore > 0.03) {
          transformationName = "first_to_third_person";
          strength = Math.min(psychologicalScore * 30, 1.5);
        }
      }
      
      setStatusMsg(`Applying ${transformationName} transformation...`);
      
      const res = await safeFetch('/transformations/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          transformation_name: transformationName,
          strength: strength,
          library_name: "narrative_transformations"
        })
      });
      
      const data = await res.json();
      if (data.transformed_text) {
        setRegeneratedText(data.transformed_text);
        setStatusMsg(`Applied ${transformationName} (strength: ${strength.toFixed(2)})`);
      } else {
        setRegeneratedText("No transformation result received");
        setStatusMsg("Transformation failed");
      }
    } catch (err) {
      console.error('Failed to apply rho transformation:', err);
      setStatusMsg("Failed to apply transformation");
      // Fallback to original behavior if new system fails
      try {
        setStatusMsg("Falling back to legacy system...");
        const res = await safeFetch('/attributes/regenerate_narrative', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            original_text: text,
            adjusted_rho_id: adjustedMatrix.adjusted_rho_id
          })
        });
        const data = await res.json();
        setRegeneratedText(data.modified_narrative);
        setStatusMsg("Legacy regeneration completed");
      } catch (fallbackErr) {
        console.error('Fallback also failed:', fallbackErr);
        setRegeneratedText("Both transformation systems failed");
        setStatusMsg("All transformation methods failed");
      }
    } finally {
      setLoading(false);
      setTimeout(() => setStatusMsg(""), 3000);
    }
  };

  const resetAdjustments = () => {
    if (allAttributes) {
      const resetAdjustments = {};
      Object.values(allAttributes.categories).flat().forEach(attr => {
        resetAdjustments[attr.name] = 0;
      });
      setAdjustments(resetAdjustments);
    }
    setAdjustedMatrix(null);
    setRegeneratedText("");
    setSentencePreviews(null);
    setShowSentencePreview(false);
  };

  const addAttributeToConversation = (attributeName, description) => {
    // Add the attribute name and description to the text area
    const attributeInfo = `\n\n[ATTRIBUTE: ${attributeName}]\n${description}\n`;
    const currentText = text || "";  // Ensure text is a string
    setText(currentText + attributeInfo);
    setStatusMsg(`Added ${attributeName} to conversation`);
    setTimeout(() => setStatusMsg(""), 2000);
  };

  // Load attributes on component mount
  useEffect(() => {
    loadAllAttributes();
  }, []);

  // Auto-extract attributes when text changes (with debounce)
  useEffect(() => {
    if (!autoMode || !text.trim()) return;
    
    const timeoutId = setTimeout(() => {
      extractAttributes();
    }, 1000); // 1 second debounce
    
    return () => clearTimeout(timeoutId);
  }, [text, autoMode]); // Dependencies: text and autoMode

  // Auto-adjust matrix when sliders change (with debounce)
  useEffect(() => {
    if (!autoMode || !extractedAttributes || !baseRhoId) return;
    
    const timeoutId = setTimeout(() => {
      adjustMatrix();
    }, 300); // 300ms debounce for responsive sliders
    
    return () => clearTimeout(timeoutId);
  }, [adjustments, autoMode, extractedAttributes, baseRhoId]); // Added missing dependencies

  // Auto-regenerate text when matrix changes
  useEffect(() => {
    if (!autoMode || !adjustedMatrix?.adjusted_rho_id) return;
    
    const timeoutId = setTimeout(() => {
      regenerateNarrative();
    }, 200); // Quick regeneration
    
    return () => clearTimeout(timeoutId);
  }, [adjustedMatrix, autoMode, text]); // Added text dependency

  // Initialize with demo text extraction on component mount
  useEffect(() => {
    if (autoMode && text.trim() && !extractedAttributes) {
      setTimeout(() => {
        extractAttributes();
      }, 500); // Initial extraction after component settles
    }
  }, []); // Run once on mount

  const createCustomAttribute = async () => {
    if (!newAttributeName.trim()) {
      alert('Please enter an attribute name');
      return;
    }
    
    try {
      setLoading(true);
      const res = await safeFetch('/attributes/create_custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newAttributeName.trim(),
          strength: newAttributeStrength,
          text: text
        })
      });
      const data = await res.json();
      
      // Add to custom attributes
      const newCustomAttrs = [...sharedNarrative.customAttributes, {
        name: newAttributeName.trim(),
        strength: newAttributeStrength,
        povm_components: data.povm_components,
        created_at: Date.now()
      }];
      updateSharedNarrative({ customAttributes: newCustomAttrs });
      
      // Reset creator
      setNewAttributeName('');
      setNewAttributeStrength(0);
      setShowAttributeCreator(false);
      setStatusMsg('Custom attribute created!');
    } catch (err) {
      console.error('Failed to create custom attribute:', err);
      setStatusMsg('Custom attribute creation failed');
    } finally {
      setLoading(false);
      setTimeout(() => setStatusMsg(''), 2000);
    }
  };
  
  const removeCustomAttribute = (index) => {
    const newCustomAttrs = sharedNarrative.customAttributes.filter((_, i) => i !== index);
    updateSharedNarrative({ customAttributes: newCustomAttrs });
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Attribute Extraction, Removal & Adjustment</h2>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <p style={{ color: '#666', margin: 0, marginBottom: 5 }}>
            Extract Persona, Namespace, and Style attributes, then adjust their strengths using POVM basis vector modifications.
          </p>
          <div style={{ fontSize: 12, color: '#888' }}>
            📊 Shared across tabs • Last updated: {new Date(sharedNarrative.lastUpdated).toLocaleTimeString()}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoMode}
              onChange={(e) => setAutoMode(e.target.checked)}
              style={{ transform: 'scale(1.2)' }}
            />
            <span style={{ fontWeight: 600, color: autoMode ? '#2e7d32' : '#666' }}>
              🤖 Real-time Mode
            </span>
          </label>
          {statusMsg && (
            <div style={{ 
              background: loading ? '#e3f2fd' : '#e8f5e8', 
              color: loading ? '#1565c0' : '#2e7d32',
              padding: '4px 12px', 
              borderRadius: 16, 
              fontSize: 12,
              fontWeight: 500
            }}>
              {loading ? '⚡ Processing...' : `✓ ${statusMsg}`}
            </div>
          )}
        </div>
      </div>

      {/* Input Section */}
      <div style={{ marginBottom: 20 }}>
        <label style={{ display: 'block', marginBottom: 5, fontWeight: 600 }}>Narrative Text:</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter your narrative text here..."
          style={{ width: '100%', minHeight: 120, padding: 12, borderRadius: 6, border: '1px solid #ddd', fontSize: 14 }}
        />
        <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
          <button 
            onClick={extractAttributes}
            disabled={loading || !text.trim()}
            style={{ padding: '8px 16px', background: '#2196f3', color: 'white', border: 'none', borderRadius: 4 }}
          >
            Extract Attributes
          </button>
          <div style={{ fontSize: 13, color: '#666', alignSelf: 'center' }}>
            {statusMsg || (loading ? "Working..." : "Ready")}
          </div>
        </div>
      </div>

      {/* Extracted Attributes Display */}
      {extractedAttributes && (
        <div style={{ marginBottom: 20, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>Extracted Attributes</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 15 }}>
            {Object.entries(extractedAttributes.attributes).map(([attr, data]) => (
              <div key={attr} style={{ background: '#f8f9fa', padding: 12, borderRadius: 6 }}>
                <div style={{ fontWeight: 600, textTransform: 'capitalize', marginBottom: 5 }}>{attr}</div>
                <div style={{ fontSize: 14, marginBottom: 8 }}>
                  <strong>Strength:</strong> {formatNumber(data.strength, 3)}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>
                  <strong>POVM Vectors:</strong> [{data.components.map(c => c.basis_index).join(', ')}]
                </div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
                  <strong>Measurements:</strong> {data.components.map(c => formatNumber(c.probability, 3)).join(', ')}
                </div>
              </div>
            ))}
          </div>
          
          {/* Original Matrix State */}
          <div style={{ marginTop: 15 }}>
            <h4>Original Matrix State</h4>
            <MatrixVisualization 
              title="Before Adjustment"
              eigs={extractedAttributes.matrix_diagnostics.eigs}
              purity={extractedAttributes.matrix_diagnostics.purity}
              entropy={extractedAttributes.matrix_diagnostics.entropy}
              size={250}
            />
          </div>
        </div>
      )}

      {/* Adjustment Controls */}
      {allAttributes && allAttributes.categories && allAttributes.category_counts && (
        <div style={{ marginBottom: 20, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>Comprehensive Attribute Adjustments</h3>
          <p style={{ color: '#666', fontSize: 14, marginBottom: 15 }}>
            {allAttributes.total_attributes || 0} attributes across {Object.keys(allAttributes.categories).length} categories. 
            Adjust strengths from -1 (weaker) to +1 (stronger) using rho/POVM theory.
          </p>
          
          {/* Category Selector */}
          <div style={{ marginBottom: 15 }}>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>Category:</label>
            <select 
              value={selectedCategory} 
              onChange={(e) => setSelectedCategory(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc', minWidth: 150 }}
            >
              {Object.entries(allAttributes.category_counts).map(([category, count]) => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)} ({count})
                </option>
              ))}
            </select>
          </div>
          
          {/* Attributes in Selected Category */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 15 }}>
            {allAttributes.categories[selectedCategory]?.map(attr => (
              <div key={attr.name} style={{ 
                background: adjustments[attr.name] !== 0 ? '#fff3e0' : '#f8f9fa', 
                padding: 12, 
                borderRadius: 6,
                border: adjustments[attr.name] !== 0 ? '2px solid #ff9800' : '1px solid #e0e0e0'
              }}>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, textTransform: 'capitalize' }}>
                  {attr.name}: {adjustments[attr.name]?.toFixed(2) || '0.00'}
                </label>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.05"
                  value={adjustments[attr.name] || 0}
                  onChange={(e) => setAdjustments(prev => ({ ...prev, [attr.name]: parseFloat(e.target.value) }))}
                  style={{ width: '100%', marginBottom: 8 }}
                />
                <div style={{ fontSize: 11, color: '#666', marginBottom: 5 }}>
                  {attr.description}
                </div>
                <div style={{ fontSize: 10, color: '#888', marginBottom: 8 }}>
                  POVM dims: [{attr.dimension_count}] | 
                  {adjustments[attr.name] > 0 ? ' Strengthen' : adjustments[attr.name] < 0 ? ' Weaken' : ' Unchanged'}
                </div>
                <button
                  onClick={() => addAttributeToConversation(attr.name, attr.description)}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    background: '#e3f2fd',
                    color: '#1976d2',
                    border: '1px solid #bbdefb',
                    borderRadius: 4,
                    cursor: 'pointer',
                    width: '100%'
                  }}
                  title={`Add ${attr.name} definition to conversation`}
                >
                  + Add to Conversation
                </button>
              </div>
            )) || []}
          </div>
          
          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 10, marginTop: 15, flexWrap: 'wrap' }}>
            <button 
              onClick={previewSentences}
              disabled={loading || Object.values(adjustments).every(v => Math.abs(v) < 0.01)}
              style={{ padding: '8px 16px', background: '#2196F3', color: 'white', border: 'none', borderRadius: 4 }}
            >
              📋 Preview Sentences
            </button>
            <button 
              onClick={adjustMatrix}
              disabled={loading}
              style={{ padding: '8px 16px', background: '#ff9800', color: 'white', border: 'none', borderRadius: 4 }}
            >
              🎛️ Adjust Matrix
            </button>
            <button 
              onClick={resetAdjustments}
              style={{ padding: '8px 16px', background: '#6c757d', color: 'white', border: 'none', borderRadius: 4 }}
            >
              🔄 Reset All
            </button>
          </div>
          
          {/* Active Adjustments Summary */}
          {Object.entries(adjustments).filter(([_, val]) => Math.abs(val) > 0.01).length > 0 && (
            <div style={{ marginTop: 15, padding: 10, background: '#e3f2fd', borderRadius: 6 }}>
              <strong>Active Adjustments ({Object.entries(adjustments).filter(([_, val]) => Math.abs(val) > 0.01).length}):</strong>
              <div style={{ fontSize: 12, marginTop: 5, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {Object.entries(adjustments)
                  .filter(([_, val]) => Math.abs(val) > 0.01)
                  .map(([attr, val]) => (
                    <span key={attr} style={{ 
                      background: val > 0 ? '#c8e6c9' : '#ffcdd2', 
                      padding: '2px 8px', 
                      borderRadius: 12,
                      fontSize: 11
                    }}>
                      {attr}: {val > 0 ? '+' : ''}{val.toFixed(2)}
                    </span>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Loading state for attributes */}
      {!allAttributes && (
        <div style={{ marginBottom: 20, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>Comprehensive Attribute Adjustments</h3>
          <p style={{ color: '#666', fontSize: 14 }}>Loading attribute library...</p>
        </div>
      )}

      {/* Sentence Preview Panel */}
      {showSentencePreview && sentencePreviews && (
        <div style={{ marginBottom: 20, border: '2px solid #2196F3', borderRadius: 8, padding: 15 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 }}>
            <h3 style={{ margin: 0, color: '#1976D2' }}>📋 Sentence-Level Preview</h3>
            <button
              onClick={() => setShowSentencePreview(false)}
              style={{ background: '#f44336', color: 'white', border: 'none', borderRadius: 4, padding: '6px 12px' }}
            >
              ✕ Close
            </button>
          </div>
          
          <div style={{ marginBottom: 15, fontSize: 14, color: '#666' }}>
            Previewing {sentencePreviews.previewed_sentences} of {sentencePreviews.total_sentences} sentences with {Object.keys(sentencePreviews.adjustments_applied).length} active attributes
          </div>
          
          {sentencePreviews.preview_results.map((result, index) => (
            <div key={index} style={{ 
              marginBottom: 20, 
              border: '1px solid #e0e0e0', 
              borderRadius: 6, 
              overflow: 'hidden'
            }}>
              <div style={{ background: '#f5f5f5', padding: 10, borderBottom: '1px solid #e0e0e0' }}>
                <strong>Sentence {result.sentence_index + 1}</strong>
                {result.transformation_success && <span style={{ color: '#4caf50', marginLeft: 10 }}>✓ Transformed</span>}
              </div>
              
              <div style={{ padding: 15 }}>
                <div style={{ marginBottom: 15 }}>
                  <div style={{ fontWeight: 600, marginBottom: 8, color: '#555' }}>Original:</div>
                  <div style={{ padding: 10, background: '#fafafa', borderRadius: 4, fontSize: 14 }}>
                    "{result.original_sentence}"
                  </div>
                </div>
                
                {result.transformation_success && (
                  <div style={{ marginBottom: 15 }}>
                    <div style={{ fontWeight: 600, marginBottom: 8, color: '#1976D2' }}>Modified:</div>
                    <div style={{ padding: 10, background: '#e3f2fd', borderRadius: 4, fontSize: 14 }}>
                      "{result.modified_sentence}"
                    </div>
                  </div>
                )}
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
                  {Object.entries(result.attribute_changes).map(([attr, change]) => (
                    <div key={attr} style={{ 
                      padding: 6, 
                      background: change > 0.01 ? '#e8f5e9' : change < -0.01 ? '#ffebee' : '#f5f5f5',
                      borderRadius: 4,
                      fontSize: 12,
                      textAlign: 'center'
                    }}>
                      <div style={{ fontWeight: 600, textTransform: 'capitalize' }}>{attr}</div>
                      <div style={{ color: change > 0.01 ? '#4caf50' : change < -0.01 ? '#f44336' : '#666' }}>
                        {change > 0 ? '+' : ''}{change.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Adjusted Matrix Display */}
      {adjustedMatrix && (
        <div style={{ marginBottom: 20, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>Matrix Adjustment Results</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 15 }}>
            <MatrixVisualization 
              title="Original Matrix"
              eigs={adjustedMatrix.base_diagnostics.eigs}
              purity={adjustedMatrix.base_diagnostics.purity}
              entropy={adjustedMatrix.base_diagnostics.entropy}
              size={200}
            />
            <MatrixVisualization 
              title="Adjusted Matrix"
              eigs={adjustedMatrix.adjusted_diagnostics.eigs}
              purity={adjustedMatrix.adjusted_diagnostics.purity}
              entropy={adjustedMatrix.adjusted_diagnostics.entropy}
              size={200}
            />
          </div>
          
          <div style={{ background: '#f8f9fa', padding: 12, borderRadius: 6 }}>
            <h4>Adjustment Summary</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, fontSize: 14 }}>
              <div><strong>Frobenius Norm Δ:</strong> {formatNumber(adjustedMatrix.difference_magnitude, 4)}</div>
              <div><strong>Purity Change:</strong> {formatNumber(adjustedMatrix.adjusted_diagnostics.purity - adjustedMatrix.base_diagnostics.purity, 4)}</div>
              <div><strong>Entropy Change:</strong> {formatNumber(adjustedMatrix.adjusted_diagnostics.entropy - adjustedMatrix.base_diagnostics.entropy, 4)}</div>
            </div>
            
            <div style={{ marginTop: 10 }}>
              <strong>Applied Adjustments:</strong>
              <div style={{ fontSize: 12, marginTop: 5 }}>
                {Object.entries(adjustments)
                  .filter(([_, val]) => val !== 0)
                  .map(([attr, val]) => `${attr}: ${val > 0 ? '+' : ''}${val.toFixed(2)}`)
                  .join(' | ') || 'No adjustments applied'}
              </div>
            </div>
          </div>
          
          <button 
            onClick={regenerateNarrative}
            disabled={loading}
            style={{ padding: '10px 20px', background: '#4caf50', color: 'white', border: 'none', borderRadius: 4, marginTop: 15 }}
          >
            Generate Modified Narrative
          </button>
        </div>
      )}

      {/* Custom Attribute Management */}
      <div style={{ marginBottom: 20, border: '1px solid #9c27b0', borderRadius: 8, padding: 15 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 }}>
          <h3 style={{ margin: 0 }}>🎨 Custom Attribute Stockpile</h3>
          <button
            onClick={() => setShowAttributeCreator(!showAttributeCreator)}
            style={{ padding: '6px 12px', background: '#9c27b0', color: 'white', border: 'none', borderRadius: 4 }}
          >
            {showAttributeCreator ? 'Cancel' : '+ Create Attribute'}
          </button>
        </div>
        
        {/* Attribute Creator */}
        {showAttributeCreator && (
          <div style={{ background: '#f3e5f5', padding: 15, borderRadius: 6, marginBottom: 15 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr auto', gap: 10, alignItems: 'end' }}>
              <div>
                <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 600 }}>Attribute Name:</label>
                <div style={{ display: 'flex', gap: 5 }}>
                  <input
                    type="text"
                    value={newAttributeName}
                    onChange={(e) => setNewAttributeName(e.target.value)}
                    placeholder="Leave empty for auto-generation, or type name"
                    style={{ flex: 1, padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc' }}
                  />
                  <button
                    onClick={() => setNewAttributeName('auto')}
                    style={{ 
                      padding: '8px 12px', 
                      background: '#ff9800', 
                      color: 'white', 
                      border: 'none', 
                      borderRadius: 4,
                      fontSize: '12px'
                    }}
                    title="Generate descriptive name automatically"
                  >
                    🤖 Auto
                  </button>
                </div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>
                  Tip: Use "auto" or leave empty for AI-generated descriptive names
                </div>
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 600 }}>Strength: {newAttributeStrength.toFixed(2)}</label>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.1"
                  value={newAttributeStrength}
                  onChange={(e) => setNewAttributeStrength(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <button
                onClick={createCustomAttribute}
                disabled={loading || !newAttributeName.trim()}
                style={{ padding: '8px 16px', background: '#4caf50', color: 'white', border: 'none', borderRadius: 4 }}
              >
                Create
              </button>
            </div>
          </div>
        )}
        
        {/* Existing Custom Attributes */}
        {sharedNarrative.customAttributes.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
            {sharedNarrative.customAttributes.map((attr, index) => (
              <div key={index} style={{ 
                background: '#f8f8ff', 
                border: '1px solid #9c27b0', 
                borderRadius: 6, 
                padding: 10,
                position: 'relative'
              }}>
                <button
                  onClick={() => removeCustomAttribute(index)}
                  style={{ 
                    position: 'absolute', 
                    top: 5, 
                    right: 5, 
                    background: '#f44336', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '50%', 
                    width: 20, 
                    height: 20, 
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                >
                  ×
                </button>
                <div style={{ fontWeight: 600, marginBottom: 5, textTransform: 'capitalize' }}>{attr.name}</div>
                <div style={{ fontSize: 12, color: '#666' }}>Strength: {attr.strength.toFixed(2)}</div>
                <div style={{ fontSize: 11, color: '#888', marginTop: 5 }}>Created: {new Date(attr.created_at).toLocaleTimeString()}</div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center', color: '#666', fontStyle: 'italic', padding: 20 }}>
            No custom attributes yet. Create your first attribute to build your narrative toolkit!
          </div>
        )}
      </div>

      {/* Real-time Narrative Transformation Results */}
      {regeneratedText && (
        <div style={{ 
          border: '2px solid #4caf50', 
          borderRadius: 12, 
          padding: 20, 
          marginBottom: 25,
          background: 'linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%)'
        }}>
          <h3 style={{ color: '#2e7d32', marginBottom: 15, display: 'flex', alignItems: 'center', gap: 8 }}>
            ✨ Real-time Transformation Results
            {loading && <div style={{ fontSize: 12, color: '#666' }}>⚡ Updating...</div>}
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 25 }}>
            <div>
              <h4 style={{ color: '#666', marginBottom: 8 }}>📝 Original Narrative</h4>
              <div style={{ background: 'white', padding: 15, borderRadius: 8, minHeight: 120, fontSize: 14, lineHeight: 1.5, border: '1px solid #e0e0e0' }}>
                {text}
              </div>
            </div>
            
            <div>
              <h4 style={{ color: '#2e7d32', marginBottom: 8 }}>🎭 Transformed Narrative</h4>
              <div style={{ 
                background: 'white', 
                padding: 15, 
                borderRadius: 8, 
                minHeight: 120, 
                fontSize: 14, 
                lineHeight: 1.5,
                border: '2px solid #4caf50',
                boxShadow: '0 2px 8px rgba(76, 175, 80, 0.1)'
              }}>
                {regeneratedText}
              </div>
            </div>
          </div>
          
          <div style={{ marginTop: 15, fontSize: 14, color: '#666' }}>
            <strong>Applied Modifications:</strong> {Object.entries(adjustments)
              .filter(([_, val]) => val !== 0)
              .map(([attr, val]) => `${attr} ${val > 0 ? 'strengthened' : 'weakened'} by ${Math.abs(val).toFixed(2)}`)
              .join(', ') || 'No modifications'}
          </div>
        </div>
      )}
    </div>
  );
}

// Book Reader Tab - Project Gutenberg Integration
function MeaningFlowVisualization({ progress, matrixState, chunksProcessed }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrixState) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 200;
    ctx.clearRect(0, 0, 400, 200);
    
    // Draw flow of meaning accumulation
    const flowPoints = 50;
    const centerY = 100;
    
    ctx.strokeStyle = '#2196f3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i <= flowPoints; i++) {
      const x = (i / flowPoints) * 380 + 10;
      const progressAtPoint = (i / flowPoints) * progress;
      
      // Create a flowing wave based on eigenvalue accumulation
      const waveHeight = Math.sin(progressAtPoint * Math.PI * 2) * 20 * progressAtPoint;
      const entropyInfluence = (matrixState.entropy / 4) * 10;
      const y = centerY + waveHeight + entropyInfluence * Math.sin(i * 0.3);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    ctx.stroke();
    
    // Add meaning "particles" at eigenvalue positions
    const topEigs = matrixState.eigs.slice(0, 5);
    topEigs.forEach((eig, idx) => {
      const x = 10 + (progress * 380);
      const y = centerY + (idx - 2) * 30;
      const radius = Math.max(2, eig * 50);
      
      ctx.fillStyle = `rgba(76, 175, 80, ${eig * 2})`;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '12px monospace';
    ctx.fillText('Meaning Flow →', 10, 20);
    ctx.fillText(`Chunks: ${chunksProcessed}`, 10, 190);
    ctx.fillText(`Entropy: ${matrixState.entropy.toFixed(2)}`, 300, 190);
    
  }, [progress, matrixState, chunksProcessed]);
  
  return (
    <div>
      <canvas 
        ref={canvasRef}
        style={{ border: '1px solid #eee', borderRadius: 4, width: '100%', height: 200 }}
      />
      <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
        Visualization shows how narrative meaning accumulates and flows as eigenvalues evolve
      </div>
    </div>
  );
}

// Eigenvalue River - shows the dominant meanings as flowing streams
function EigenvalueRiverVisualization({ matrixState, progress }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrixState) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 200;
    ctx.clearRect(0, 0, 400, 200);
    
    // Draw eigenvalue "rivers" - top 8 eigenvalues as flowing streams
    const topEigs = matrixState.eigs.slice(0, 8);
    const riverY = 180;
    
    topEigs.forEach((eig, idx) => {
      const width = Math.max(2, eig * 100);
      const x = (idx / 8) * 380 + 10;
      const flow = Math.sin(progress * Math.PI * 4 + idx) * 5;
      
      // River bed
      ctx.fillStyle = `rgba(33, 150, 243, 0.3)`;
      ctx.fillRect(x, riverY - width/2, 40, width);
      
      // Flowing water
      ctx.fillStyle = `rgba(33, 150, 243, ${eig * 2})`;
      for (let i = 0; i < 5; i++) {
        const flowX = x + (i * 8) + flow;
        const flowWidth = width * (0.5 + 0.5 * Math.sin(progress * Math.PI * 2 + i));
        ctx.fillRect(flowX, riverY - flowWidth/2, 6, flowWidth);
      }
    });
    
    // Labels
    ctx.fillStyle = '#333';
    ctx.font = '12px monospace';
    ctx.fillText('Eigenvalue Rivers', 10, 20);
    ctx.fillText('← Dominant meanings flow like streams →', 10, 40);
    
  }, [matrixState, progress]);
  
  return (
    <div>
      <canvas 
        ref={canvasRef}
        style={{ border: '1px solid #eee', borderRadius: 4, width: '100%', height: 200 }}
      />
      <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
        Each river represents a dominant meaning (eigenvalue) with flow intensity showing strength
      </div>
    </div>
  );
}

// Entropy Journey - shows the complexity journey of understanding
function EntropyJourneyVisualization({ entropy, purity, progress }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 200;
    ctx.clearRect(0, 0, 400, 200);
    
    // Draw entropy journey as a landscape
    const points = 100;
    ctx.strokeStyle = '#ff9800';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    for (let i = 0; i <= points; i++) {
      const x = (i / points) * 380 + 10;
      const progressPoint = (i / points) * progress;
      
      // Create landscape based on entropy - higher entropy = more mountainous
      const baseHeight = 180 - (entropy / 5) * 100;
      const complexity = Math.sin(progressPoint * Math.PI * 6) * (entropy / 2) * 20;
      const y = Math.max(20, baseHeight + complexity);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    ctx.stroke();
    
    // Fill the area under the curve
    ctx.lineTo(390, 190);
    ctx.lineTo(10, 190);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 152, 0, 0.3)';
    ctx.fill();
    
    // Purity indicator
    const purityY = 50 + (1 - purity) * 100;
    ctx.fillStyle = 'rgba(156, 39, 176, 0.7)';
    ctx.beginPath();
    ctx.arc(350, purityY, 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Labels
    ctx.fillStyle = '#333';
    ctx.font = '12px monospace';
    ctx.fillText('Understanding Journey', 10, 20);
    ctx.fillText(`Entropy: ${entropy.toFixed(2)}`, 10, 190);
    ctx.fillText(`Purity: ${purity.toFixed(3)}`, 250, 30);
    
  }, [entropy, purity, progress]);
  
  return (
    <div>
      <canvas 
        ref={canvasRef}
        style={{ border: '1px solid #eee', borderRadius: 4, width: '100%', height: 200 }}
      />
      <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
        Landscape shows complexity of understanding - higher peaks = more complex meaning
      </div>
    </div>
  );
}

// Batch Processing Queue Tab
function BatchQueueTab() {
  const [queries, setQueries] = useState(['classic literature', 'adventure stories']);
  const [instructions, setInstructions] = useState('Focus on narrative structure and character development for comparative analysis');
  const [queueStatus, setQueueStatus] = useState(null);
  const [submittedJobs, setSubmittedJobs] = useState([]);
  const [loading, setLoading] = useState(false);

  const addQuery = () => {
    setQueries([...queries, '']);
  };

  const updateQuery = (index, value) => {
    const newQueries = [...queries];
    newQueries[index] = value;
    setQueries(newQueries);
  };

  const removeQuery = (index) => {
    setQueries(queries.filter((_, i) => i !== index));
  };

  const submitBatchJob = async () => {
    const validQueries = queries.filter(q => q.trim());
    if (validQueries.length === 0) {
      alert('Please add at least one search query');
      return;
    }

    try {
      setLoading(true);
      const res = await safeFetch('/queue/batch_submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          search_queries: validQueries,
          instructions: instructions,
          chunk_size: 350,
          reading_alpha: 0.2,
          priority: 'medium',
          max_books_per_query: 2,
          auto_finalize: true
        })
      });
      const data = await res.json();
      setSubmittedJobs(data.jobs);
      await refreshQueueStatus();
    } catch (err) {
      console.error('Failed to submit batch job:', err);
    } finally {
      setLoading(false);
    }
  };

  const refreshQueueStatus = async () => {
    try {
      const res = await safeFetch('/queue/status');
      const data = await res.json();
      setQueueStatus(data);
    } catch (err) {
      console.error('Failed to get queue status:', err);
    }
  };

  const cancelJob = async (jobId) => {
    try {
      await safeFetch(`/queue/cancel/${jobId}`, { method: 'POST' });
      await refreshQueueStatus();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  useEffect(() => {
    refreshQueueStatus();
    const interval = setInterval(refreshQueueStatus, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>Batch Processing Queue</h2>
      <p style={{ color: '#666', marginBottom: 20 }}>
        Submit multiple search queries to the intelligent agent scheduler for automated batch processing.
      </p>

      {/* Job Submission Interface */}
      <div style={{ marginBottom: 30, border: '1px solid #ddd', borderRadius: 8, padding: 20 }}>
        <h3>📋 Submit New Batch Job</h3>
        
        {/* Search Queries */}
        <div style={{ marginBottom: 15 }}>
          <label style={{ display: 'block', marginBottom: 5, fontWeight: 600 }}>Search Queries:</label>
          {queries.map((query, index) => (
            <div key={index} style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
              <input
                type="text"
                value={query}
                onChange={(e) => updateQuery(index, e.target.value)}
                placeholder={`Search query ${index + 1} (e.g., "dickens", "adventure", "gothic")`}
                style={{ flex: 1, padding: '8px 12px', borderRadius: 4, border: '1px solid #ccc' }}
              />
              <button 
                onClick={() => removeQuery(index)}
                disabled={queries.length === 1}
                style={{ padding: '8px 12px', background: '#f44336', color: 'white', border: 'none', borderRadius: 4 }}
              >
                Remove
              </button>
            </div>
          ))}
          <button 
            onClick={addQuery}
            style={{ padding: '6px 12px', background: '#2196f3', color: 'white', border: 'none', borderRadius: 4 }}
          >
            + Add Query
          </button>
        </div>

        {/* Instructions */}
        <div style={{ marginBottom: 15 }}>
          <label style={{ display: 'block', marginBottom: 5, fontWeight: 600 }}>Agent Instructions:</label>
          <textarea
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="Provide instructions for how the agent should process these books..."
            style={{ width: '100%', minHeight: 80, padding: 12, borderRadius: 4, border: '1px solid #ccc' }}
          />
        </div>

        {/* Submit Button */}
        <button
          onClick={submitBatchJob}
          disabled={loading}
          style={{ padding: '12px 24px', background: '#4caf50', color: 'white', border: 'none', borderRadius: 4, fontWeight: 600 }}
        >
          {loading ? 'Submitting...' : '🚀 Submit to Agent Scheduler'}
        </button>
      </div>

      {/* Queue Status */}
      {queueStatus && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 30 }}>
          {/* Queue Overview */}
          <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
            <h3>🎯 Queue Overview</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 14 }}>
              <div style={{ background: '#e3f2fd', padding: 10, borderRadius: 4, textAlign: 'center' }}>
                <div style={{ fontWeight: 600, fontSize: 18 }}>{queueStatus.queue_summary.queued}</div>
                <div>Queued</div>
              </div>
              <div style={{ background: '#fff3e0', padding: 10, borderRadius: 4, textAlign: 'center' }}>
                <div style={{ fontWeight: 600, fontSize: 18 }}>{queueStatus.queue_summary.processing}</div>
                <div>Processing</div>
              </div>
              <div style={{ background: '#e8f5e8', padding: 10, borderRadius: 4, textAlign: 'center' }}>
                <div style={{ fontWeight: 600, fontSize: 18 }}>{queueStatus.queue_summary.completed}</div>
                <div>Completed</div>
              </div>
              <div style={{ background: '#ffebee', padding: 10, borderRadius: 4, textAlign: 'center' }}>
                <div style={{ fontWeight: 600, fontSize: 18 }}>{queueStatus.queue_summary.failed}</div>
                <div>Failed</div>
              </div>
            </div>
            
            <div style={{ marginTop: 15, fontSize: 12, color: '#666' }}>
              Background Processor: {queueStatus.processor_status.background_processor_running ? 
                '✅ Running' : '❌ Stopped'}
            </div>
          </div>

          {/* Currently Processing */}
          <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
            <h3>⚡ Currently Processing</h3>
            {queueStatus.current_jobs.processing.length > 0 ? (
              queueStatus.current_jobs.processing.map(job => (
                <div key={job.job_id} style={{ 
                  border: '1px solid #ff9800', 
                  borderRadius: 4, 
                  padding: 10, 
                  marginBottom: 10,
                  background: '#fff3e0'
                }}>
                  <div style={{ fontWeight: 600, fontSize: 14 }}>{job.book_title}</div>
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
                    Started: {new Date(job.started_at).toLocaleTimeString()}
                  </div>
                  
                  {/* Progress Bar */}
                  <div style={{ background: '#f0f0f0', borderRadius: 10, height: 8, overflow: 'hidden' }}>
                    <div style={{ 
                      background: '#ff9800', 
                      height: '100%', 
                      width: `${job.progress * 100}%`,
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <div style={{ fontSize: 12, marginTop: 5 }}>
                    {job.chunks_processed} / {job.total_chunks} chunks ({Math.round(job.progress * 100)}%)
                  </div>
                </div>
              ))
            ) : (
              <div style={{ textAlign: 'center', color: '#666', fontStyle: 'italic' }}>
                No jobs currently processing
              </div>
            )}
          </div>
        </div>
      )}

      {/* Queued Jobs */}
      {queueStatus && queueStatus.current_jobs.queued.length > 0 && (
        <div style={{ marginBottom: 30, border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>📚 Queued Jobs</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 15 }}>
            {queueStatus.current_jobs.queued.map(job => (
              <div key={job.job_id} style={{ 
                border: '1px solid #2196f3', 
                borderRadius: 6, 
                padding: 12,
                background: '#e3f2fd'
              }}>
                <div style={{ fontWeight: 600, marginBottom: 5 }}>{job.book_title}</div>
                <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
                  Priority: <span style={{ 
                    background: job.priority === 'high' ? '#ff9800' : job.priority === 'medium' ? '#2196f3' : '#9e9e9e',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: 3,
                    fontSize: 10,
                    textTransform: 'uppercase'
                  }}>{job.priority}</span>
                </div>
                <div style={{ fontSize: 12, marginBottom: 8 }}>
                  {job.agent_notes}
                </div>
                <div style={{ fontSize: 11, color: '#888' }}>
                  Queued: {new Date(job.created_at).toLocaleString()}
                </div>
                
                <button
                  onClick={() => cancelJob(job.job_id)}
                  style={{ 
                    marginTop: 8, 
                    padding: '4px 8px', 
                    background: '#f44336', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: 3,
                    fontSize: 11
                  }}
                >
                  Cancel
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Last Submitted Jobs */}
      {submittedJobs.length > 0 && (
        <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 15 }}>
          <h3>✅ Last Submitted Jobs</h3>
          <div style={{ fontSize: 14, color: '#666', marginBottom: 10 }}>
            The intelligent agent selected these books from your queries:
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 10 }}>
            {submittedJobs.map(job => (
              <div key={job.job_id} style={{ 
                border: '1px solid #4caf50', 
                borderRadius: 4, 
                padding: 10,
                background: '#e8f5e8'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{job.book_title}</div>
                <div style={{ fontSize: 11, color: '#666' }}>by {job.book_author}</div>
                <div style={{ fontSize: 11, marginTop: 5, color: '#555' }}>
                  {job.agent_notes}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Render the app
const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<RhoWorkbench />);
} else {
  console.error("Root element not found: create a <div id='root'></div> in index.html");
}