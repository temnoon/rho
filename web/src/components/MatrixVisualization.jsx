/**
 * MatrixVisualization - Visual representation of quantum density matrices.
 * 
 * Provides an intuitive heatmap visualization of eigenvalue distributions
 * and quantum state properties for real-time feedback.
 */

import React, { useRef, useEffect } from 'react';
import { formatNumber } from '../utils/api.js';

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
    
    // Normalize eigenvalues for color mapping
    const maxEig = Math.max(...eigs, 0.001);
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const idx = i * gridSize + j;
        const eigVal = eigs[idx] || 0;
        
        // Color intensity based on normalized eigenvalue
        const intensity = Math.min(255, Math.floor((eigVal / maxEig) * 255));
        const r = intensity;
        const g = Math.floor(intensity * 0.7);
        const b = Math.floor(intensity * 0.3);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(i * cellSize, j * cellSize, cellSize - 1, cellSize - 1);
        
        // Add eigenvalue text for top components
        if (idx < 8 && eigVal > maxEig * 0.1) {
          ctx.fillStyle = eigVal > maxEig * 0.5 ? 'white' : 'black';
          ctx.font = '8px monospace';
          ctx.fillText(eigVal.toFixed(3), i * cellSize + 2, j * cellSize + 10);
        }
      }
    }
    
    // Add border
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, size, size);
    
  }, [eigs, size]);

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
      <div style={{ fontWeight: 700, marginBottom: 8, fontSize: 14 }}>{title}</div>
      <canvas 
        ref={canvasRef} 
        width={size} 
        height={size}
        style={{ border: '1px solid #ccc', borderRadius: 4 }}
      />
      <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Purity: {formatNumber(purity, 3)}</span>
          <span>Entropy: {formatNumber(entropy, 3)}</span>
        </div>
        <div style={{ marginTop: 4 }}>
          Top λ: {eigs[0] ? formatNumber(eigs[0], 3) : 'N/A'}
        </div>
        <div style={{ fontSize: 10, color: '#999', marginTop: 4 }}>
          Eigenvalue heatmap (64D → 8×8 grid)
        </div>
      </div>
    </div>
  );
}

export default MatrixVisualization;