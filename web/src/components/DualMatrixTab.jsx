import React, { useState } from 'react';

// Matrix Visualization - shows actual eigenvalues as heatmap
function MatrixVisualization({ title, eigs = [], purity = 0, entropy = 0, size = 300 }) {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
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

  const formatNumber = (n, digits = 4) => {
    if (!Number.isFinite(n)) return "NaN";
    return Number(n).toFixed(digits);
  };

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

// Dual Matrix Management Tab
export function DualMatrixTab() {
  const [compositeState, setCompositeState] = useState(null);
  const [previewState, setPreviewState] = useState(null);
  const [previewText, setPreviewText] = useState('');
  const [alpha, setAlpha] = useState(0.2);
  const [previewResult, setPreviewResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const safeFetch = async (path, options = {}) => {
    const baseUrl = 'http://localhost:8192';
    const fullUrl = path.startsWith('http') ? path : `${baseUrl}${path}`;
    const response = await fetch(fullUrl, {
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

  const formatNumber = (n, digits = 4) => {
    if (!Number.isFinite(n)) return "NaN";
    return Number(n).toFixed(digits);
  };

  const initDualMatrices = async () => {
    try {
      setLoading(true);
      const res = await safeFetch('/matrix/dual/init', { method: 'POST' });
      const data = await res.json();
      setCompositeState(data.composite);
      setPreviewState(data.preview);
    } catch (err) {
      console.error('Failed to initialize dual matrices:', err);
    } finally {
      setLoading(false);
    }
  };

  const previewNarrative = async () => {
    if (!previewText.trim()) return;
    
    try {
      setLoading(true);
      const res = await safeFetch('/matrix/dual/preview_narrative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: previewText, alpha })
      });
      const data = await res.json();
      setPreviewResult(data);
    } catch (err) {
      console.error('Failed to preview narrative:', err);
    } finally {
      setLoading(false);
    }
  };

  const applyPreview = async () => {
    try {
      setLoading(true);
      const res = await safeFetch('/matrix/dual/apply_preview', { method: 'POST' });
      const data = await res.json();
      setCompositeState(data.composite);
      setPreviewText('');
      setPreviewResult(null);
    } catch (err) {
      console.error('Failed to apply preview:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Dual Matrix Management</h2>
      <p style={{ color: '#666', marginBottom: 20 }}>
        Manage composite and preview matrices. Test narrative effects before applying them.
      </p>

      <div style={{ marginBottom: 20 }}>
        <button 
          onClick={initDualMatrices} 
          disabled={loading}
          style={{ padding: '8px 16px', background: '#2196f3', color: 'white', border: 'none', borderRadius: 4 }}
        >
          Initialize Dual Matrices
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Composite Matrix */}
        <div>
          <h3>Composite Matrix</h3>
          {compositeState ? (
            <MatrixVisualization 
              title="Composite State"
              eigs={compositeState.eigs}
              purity={compositeState.purity}
              entropy={compositeState.entropy}
              size={250}
            />
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>
              Initialize matrices to begin
            </div>
          )}
        </div>

        {/* Preview Matrix */}
        <div>
          <h3>Preview Matrix</h3>
          {previewState || previewResult ? (
            <MatrixVisualization 
              title="Preview State"
              eigs={previewResult?.preview?.eigs || previewState?.eigs}
              purity={previewResult?.preview?.purity || previewState?.purity}
              entropy={previewResult?.preview?.entropy || previewState?.entropy}
              size={250}
            />
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>
              Preview changes will appear here
            </div>
          )}
        </div>
      </div>

      {/* Narrative Preview Interface */}
      {compositeState && (
        <div style={{ marginTop: 30, border: '1px solid #ddd', borderRadius: 8, padding: 20 }}>
          <h3>Preview Narrative Effect</h3>
          
          <div style={{ marginBottom: 15 }}>
            <textarea
              value={previewText}
              onChange={(e) => setPreviewText(e.target.value)}
              placeholder="Enter narrative text to preview its effect..."
              style={{ width: '100%', minHeight: 100, padding: 10, borderRadius: 4, border: '1px solid #ccc' }}
            />
          </div>

          <div style={{ display: 'flex', gap: 15, alignItems: 'center', marginBottom: 15 }}>
            <label>α (blend): {alpha.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
          </div>

          <div style={{ display: 'flex', gap: 10 }}>
            <button 
              onClick={previewNarrative}
              disabled={loading || !previewText.trim()}
              style={{ padding: '8px 16px', background: '#ff9800', color: 'white', border: 'none', borderRadius: 4 }}
            >
              Preview Effect
            </button>
            
            {previewResult && (
              <button 
                onClick={applyPreview}
                disabled={loading}
                style={{ padding: '8px 16px', background: '#4caf50', color: 'white', border: 'none', borderRadius: 4 }}
              >
                Apply to Composite
              </button>
            )}
          </div>

          {previewResult && (
            <div style={{ marginTop: 15, background: '#f8f9fa', padding: 15, borderRadius: 4 }}>
              <h4>Preview Results</h4>
              <div><strong>Difference Magnitude:</strong> {formatNumber(previewResult.difference_magnitude, 4)}</div>
              <div><strong>Eigenvalue Changes:</strong></div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 5 }}>
                <div>
                  <strong>Before:</strong><br/>
                  {previewResult.eigenvalue_changes.before.map((val, idx) => (
                    <div key={idx}>λ{idx+1}: {formatNumber(val, 4)}</div>
                  ))}
                </div>
                <div>
                  <strong>After:</strong><br/>
                  {previewResult.eigenvalue_changes.after.map((val, idx) => (
                    <div key={idx}>λ{idx+1}: {formatNumber(val, 4)}</div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}