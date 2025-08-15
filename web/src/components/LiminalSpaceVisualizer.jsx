import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiUrl } from '../utils/api.js';

/**
 * Liminal Space Visualizer - Post-Lexical Grammatological Laboratory
 * 
 * Visualizes the "liminal space between observable words" through various
 * quantum-inspired representations of text evolution and channel dynamics.
 * 
 * Visualization Modes:
 * - Eigenstate Flow: Evolution of quantum eigenvalues through text
 * - Channel Phase Space: Complex phase relationships in channel operations
 * - Embedding Trajectory: Path through high-dimensional semantic space
 * - Residue Landscape: Topological view of narrative loops and residues
 * - Integrability Manifold: Visualization of path independence
 */
export function LiminalSpaceVisualizer({ rhoId, width = 800, height = 600 }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  const [visualizationMode, setVisualizationMode] = useState('eigenstate_flow');
  const [isAnimating, setIsAnimating] = useState(false);
  const [quantumData, setQuantumData] = useState(null);
  const [embeddingTrajectory, setEmbeddingTrajectory] = useState([]);
  const [channelHistory, setChannelHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [timeStep, setTimeStep] = useState(0);

  // Visualization modes configuration
  const visualizationModes = {
    eigenstate_flow: {
      name: 'Eigenstate Flow',
      description: 'Evolution of quantum eigenvalues through narrative time',
      icon: '🌊',
      color: '#2196F3'
    },
    channel_phase_space: {
      name: 'Channel Phase Space',
      description: 'Complex phase relationships in quantum channel operations',
      icon: '🌀',
      color: '#9C27B0'
    },
    embedding_trajectory: {
      name: 'Embedding Trajectory',
      description: 'Path through high-dimensional semantic space',
      icon: '🛤️',
      color: '#FF9800'
    },
    residue_landscape: {
      name: 'Residue Landscape',
      description: 'Topological view of narrative loops and residues',
      icon: '🏔️',
      color: '#4CAF50'
    },
    integrability_manifold: {
      name: 'Integrability Manifold',
      description: 'Visualization of path independence and segmentation effects',
      icon: '🕸️',
      color: '#F44336'
    }
  };

  // Fetch quantum data for visualization
  const fetchQuantumData = useCallback(async () => {
    if (!rhoId) return;
    
    setLoading(true);
    try {
      // Get current quantum state
      const rhoResponse = await fetch(apiUrl(`/rho/${rhoId}`));
      if (rhoResponse.ok) {
        const rhoData = await rhoResponse.json();
        setQuantumData(rhoData);
      }

      // Get POVM measurements for visualization data
      try {
        const povmResponse = await fetch(apiUrl(`/packs/measure/${rhoId}`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pack_id: 'advanced_narrative_pack' })
        });
        if (povmResponse.ok) {
          const povmData = await povmResponse.json();
          setQuantumData(prev => ({ ...prev, povm: povmData }));
        }
      } catch (e) {
        console.log('No POVM data available');
      }

      // Get channel audit data if available
      try {
        const auditResponse = await fetch(apiUrl(`/audit/channel_health/${rhoId}`));
        if (auditResponse.ok) {
          const auditData = await auditResponse.json();
          setQuantumData(prev => ({ ...prev, audit: auditData }));
        }
      } catch (e) {
        console.log('No audit data available');
      }

    } catch (error) {
      console.error('Failed to fetch quantum data:', error);
    } finally {
      setLoading(false);
    }
  }, [rhoId]);

  // Generate embedding trajectory data
  const generateEmbeddingTrajectory = useCallback(() => {
    if (!quantumData) return;

    // Use both eigenvalues and POVM measurements for richer embedding
    const eigs = quantumData.diagnostics?.eigs || [];
    const measurements = quantumData.povm?.measurements || {};
    const trajectory = [];
    
    // Convert measurements to trajectory points
    const measurementKeys = Object.keys(measurements);
    const numPoints = Math.max(Math.min(eigs.length, 50), measurementKeys.length);
    
    for (let i = 0; i < numPoints; i++) {
      const t = i / numPoints;
      const eigenValue = eigs[i] || 0;
      
      // If we have measurements, use them to influence position
      let measurementInfluence = 0;
      if (measurementKeys.length > 0) {
        const measurementKey = measurementKeys[i % measurementKeys.length];
        measurementInfluence = measurements[measurementKey] || 0;
      }
      
      // Create more complex trajectory using both eigenvalues and measurements
      const radius = (eigenValue + measurementInfluence) * 200;
      const angle = t * 2 * Math.PI + measurementInfluence * Math.PI;
      
      const x = Math.cos(angle) * radius + width / 2;
      const y = Math.sin(angle) * radius + height / 2;
      const z = eigenValue + measurementInfluence; // Combined "height"
      
      trajectory.push({
        x: x,
        y: y,
        z: z,
        eigenValue: eigenValue,
        measurement: measurementInfluence,
        measurementKey: measurementKeys[i % measurementKeys.length] || null,
        index: i,
        t: t
      });
    }
    
    setEmbeddingTrajectory(trajectory);
  }, [quantumData, width, height]);

  // Generate channel history for phase space visualization
  const generateChannelHistory = useCallback(() => {
    if (!quantumData) return;

    const history = [];
    const eigs = quantumData.diagnostics?.eigs || [];
    
    // Create simulated channel evolution
    for (let i = 0; i < Math.min(eigs.length, 30); i++) {
      const eigenVal = eigs[i] || 0;
      const phase = i * 0.2;
      
      history.push({
        real: eigenVal * Math.cos(phase),
        imag: eigenVal * Math.sin(phase),
        magnitude: eigenVal,
        phase: phase,
        step: i
      });
    }
    
    setChannelHistory(history);
  }, [quantumData]);

  // Animation loop
  const animate = useCallback(() => {
    if (!isAnimating) return;
    
    setTimeStep(prev => (prev + 0.02) % (2 * Math.PI));
    animationRef.current = requestAnimationFrame(animate);
  }, [isAnimating]);

  // Start/stop animation
  const toggleAnimation = useCallback(() => {
    setIsAnimating(prev => {
      if (!prev) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        cancelAnimationFrame(animationRef.current);
      }
      return !prev;
    });
  }, [animate]);

  // Render eigenstate flow visualization
  const renderEigenstateFlow = useCallback((ctx) => {
    if (!quantumData) return;

    const eigs = quantumData.diagnostics?.eigs || [];
    const centerX = width / 2;
    const centerY = height / 2;

    // Clear canvas
    ctx.fillStyle = '#000015';
    ctx.fillRect(0, 0, width, height);

    // Draw eigenvalue flows
    for (let i = 0; i < Math.min(eigs.length, 64); i++) {
      const eigenValue = eigs[i] || 0;
      if (eigenValue < 1e-6) continue;

      const angle = (i / 64) * 2 * Math.PI;
      const radius = eigenValue * 200;
      const animatedRadius = radius * (1 + 0.1 * Math.sin(timeStep + i * 0.1));
      
      const x = centerX + Math.cos(angle) * animatedRadius;
      const y = centerY + Math.sin(angle) * animatedRadius;
      
      // Color based on eigenvalue magnitude
      const intensity = Math.min(eigenValue * 1000, 1);
      const hue = (i / 64) * 360;
      
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, eigenValue * 10), 0, 2 * Math.PI);
      ctx.fillStyle = `hsla(${hue}, 80%, ${50 + intensity * 30}%, ${0.3 + intensity * 0.7})`;
      ctx.fill();
      
      // Draw connecting lines for flow
      if (i > 0) {
        const prevAngle = ((i - 1) / 64) * 2 * Math.PI;
        const prevRadius = (eigs[i - 1] || 0) * 200;
        const prevX = centerX + Math.cos(prevAngle) * prevRadius;
        const prevY = centerY + Math.sin(prevAngle) * prevRadius;
        
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = `hsla(${hue}, 60%, 50%, 0.2)`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Draw entropy indicator
    const entropy = quantumData.diagnostics?.entropy || 0;
    ctx.fillStyle = `rgba(255, 255, 255, ${0.1 + entropy * 0.1})`;
    ctx.beginPath();
    ctx.arc(centerX, centerY, entropy * 50, 0, 2 * Math.PI);
    ctx.fill();

  }, [quantumData, width, height, timeStep]);

  // Render channel phase space
  const renderChannelPhaseSpace = useCallback((ctx) => {
    if (channelHistory.length === 0) return;

    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) * 0.3;

    // Clear canvas
    ctx.fillStyle = '#0a0a20';
    ctx.fillRect(0, 0, width, height);

    // Draw phase space trajectory
    ctx.beginPath();
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;

    for (let i = 0; i < channelHistory.length; i++) {
      const point = channelHistory[i];
      const x = centerX + point.real * scale;
      const y = centerY + point.imag * scale;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw phase space points with animation
    for (let i = 0; i < channelHistory.length; i++) {
      const point = channelHistory[i];
      const x = centerX + point.real * scale;
      const y = centerY + point.imag * scale;
      
      const animationPhase = timeStep + i * 0.3;
      const radius = 3 + 2 * Math.sin(animationPhase);
      const alpha = 0.3 + 0.7 * (1 - i / channelHistory.length);
      
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = `hsla(${120 + i * 5}, 80%, 60%, ${alpha})`;
      ctx.fill();
    }

    // Draw coordinate axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

  }, [channelHistory, width, height, timeStep]);

  // Render embedding trajectory
  const renderEmbeddingTrajectory = useCallback((ctx) => {
    if (embeddingTrajectory.length === 0) return;

    // Clear canvas with gradient background
    const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, Math.max(width, height));
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#0f0f23');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Draw trajectory path
    ctx.beginPath();
    ctx.strokeStyle = '#FF9800';
    ctx.lineWidth = 2;

    for (let i = 0; i < embeddingTrajectory.length; i++) {
      const point = embeddingTrajectory[i];
      
      if (i === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    }
    ctx.stroke();

    // Draw trajectory points with varying sizes based on eigenvalues
    for (let i = 0; i < embeddingTrajectory.length; i++) {
      const point = embeddingTrajectory[i];
      const radius = 2 + point.eigenValue * 8;
      const animatedRadius = radius * (1 + 0.2 * Math.sin(timeStep + i * 0.15));
      
      const alpha = 0.4 + 0.6 * point.eigenValue;
      
      ctx.beginPath();
      ctx.arc(point.x, point.y, animatedRadius, 0, 2 * Math.PI);
      ctx.fillStyle = `hsla(${30 + i * 3}, 90%, 60%, ${alpha})`;
      ctx.fill();
      
      // Add glow effect for high eigenvalues
      if (point.eigenValue > 0.1) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, animatedRadius * 2, 0, 2 * Math.PI);
        ctx.fillStyle = `hsla(${30 + i * 3}, 90%, 80%, ${alpha * 0.2})`;
        ctx.fill();
      }
    }

  }, [embeddingTrajectory, width, height, timeStep]);

  // Render residue landscape
  const renderResidueLandscape = useCallback((ctx) => {
    if (!quantumData) return;

    const eigs = quantumData.diagnostics?.eigs || [];
    
    // Clear canvas
    ctx.fillStyle = '#0d1421';
    ctx.fillRect(0, 0, width, height);

    // Create landscape based on eigenvalue distribution
    const gridSize = 32;
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const eigIndex = (i * gridSize + j) % eigs.length;
        const eigenValue = eigs[eigIndex] || 0;
        
        // Create landscape height based on eigenvalue
        const height_val = eigenValue;
        const animatedHeight = height_val * (1 + 0.3 * Math.sin(timeStep + i * 0.1 + j * 0.1));
        
        const x = i * cellWidth;
        const y = j * cellHeight;
        
        // Color based on height and position
        const hue = 220 + animatedHeight * 100;
        const saturation = 60 + animatedHeight * 40;
        const lightness = 20 + animatedHeight * 60;
        
        ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.8)`;
        ctx.fillRect(x, y, cellWidth, cellHeight);
        
        // Add residue indicators for significant eigenvalues
        if (eigenValue > 0.05) {
          ctx.beginPath();
          ctx.arc(x + cellWidth/2, y + cellHeight/2, cellWidth * 0.3, 0, 2 * Math.PI);
          ctx.fillStyle = `hsla(${hue + 180}, 90%, 70%, 0.6)`;
          ctx.fill();
        }
      }
    }

  }, [quantumData, width, height, timeStep]);

  // Render integrability manifold
  const renderIntegrabilityManifold = useCallback((ctx) => {
    if (!quantumData) return;

    const eigs = quantumData.diagnostics?.eigs || [];
    
    // Clear canvas
    ctx.fillStyle = '#1a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Draw manifold mesh
    const meshResolution = 20;
    const stepX = width / meshResolution;
    const stepY = height / meshResolution;

    ctx.strokeStyle = '#F44336';
    ctx.lineWidth = 1;

    // Horizontal lines
    for (let i = 0; i <= meshResolution; i++) {
      const y = i * stepY;
      const eigenIndex = Math.floor((i / meshResolution) * eigs.length);
      const eigenValue = eigs[eigenIndex] || 0;
      
      ctx.beginPath();
      for (let j = 0; j <= meshResolution; j++) {
        const x = j * stepX;
        const distortion = eigenValue * 20 * Math.sin(timeStep + x * 0.01 + y * 0.01);
        
        if (j === 0) {
          ctx.moveTo(x, y + distortion);
        } else {
          ctx.lineTo(x, y + distortion);
        }
      }
      ctx.globalAlpha = 0.3 + eigenValue * 0.7;
      ctx.stroke();
    }

    // Vertical lines
    for (let j = 0; j <= meshResolution; j++) {
      const x = j * stepX;
      
      ctx.beginPath();
      for (let i = 0; i <= meshResolution; i++) {
        const y = i * stepY;
        const eigenIndex = Math.floor((i / meshResolution) * eigs.length);
        const eigenValue = eigs[eigenIndex] || 0;
        const distortion = eigenValue * 20 * Math.sin(timeStep + x * 0.01 + y * 0.01);
        
        if (i === 0) {
          ctx.moveTo(x + distortion, y);
        } else {
          ctx.lineTo(x + distortion, y);
        }
      }
      ctx.globalAlpha = 0.2;
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    // Draw path independence indicators
    const numPaths = 8;
    for (let p = 0; p < numPaths; p++) {
      const pathAngle = (p / numPaths) * 2 * Math.PI;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = 100;
      
      const pathX = centerX + Math.cos(pathAngle + timeStep) * radius;
      const pathY = centerY + Math.sin(pathAngle + timeStep) * radius;
      
      ctx.beginPath();
      ctx.arc(pathX, pathY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = `hsla(${p * 45}, 80%, 60%, 0.8)`;
      ctx.fill();
    }

  }, [quantumData, width, height, timeStep]);

  // Main render function
  const renderVisualization = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    switch (visualizationMode) {
      case 'eigenstate_flow':
        renderEigenstateFlow(ctx);
        break;
      case 'channel_phase_space':
        renderChannelPhaseSpace(ctx);
        break;
      case 'embedding_trajectory':
        renderEmbeddingTrajectory(ctx);
        break;
      case 'residue_landscape':
        renderResidueLandscape(ctx);
        break;
      case 'integrability_manifold':
        renderIntegrabilityManifold(ctx);
        break;
      default:
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
    }
  }, [visualizationMode, renderEigenstateFlow, renderChannelPhaseSpace, 
      renderEmbeddingTrajectory, renderResidueLandscape, renderIntegrabilityManifold]);

  // Effects
  useEffect(() => {
    fetchQuantumData();
  }, [fetchQuantumData]);

  useEffect(() => {
    generateEmbeddingTrajectory();
    generateChannelHistory();
  }, [generateEmbeddingTrajectory, generateChannelHistory]);

  useEffect(() => {
    renderVisualization();
  }, [renderVisualization, timeStep]);

  useEffect(() => {
    if (isAnimating) {
      animate();
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isAnimating, animate]);

  if (!rhoId) {
    return (
      <div style={{ 
        padding: 20,
        textAlign: 'center',
        color: '#666',
        border: '1px dashed #ddd',
        borderRadius: 8
      }}>
        <div style={{ fontSize: 16, marginBottom: 8 }}>🌌 Liminal Space Visualizer</div>
        <div style={{ fontSize: 12 }}>Select a quantum state to visualize the space between words</div>
      </div>
    );
  }

  return (
    <div style={{ padding: 15 }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: 20,
        borderBottom: '2px solid #eee',
        paddingBottom: 10
      }}>
        <h3 style={{ margin: 0, fontSize: 18, color: '#333' }}>
          🌌 Liminal Space Visualizer
        </h3>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            onClick={toggleAnimation}
            style={{
              padding: '6px 12px',
              border: `1px solid ${isAnimating ? '#f44336' : '#4CAF50'}`,
              borderRadius: 4,
              backgroundColor: isAnimating ? '#f44336' : '#4CAF50',
              color: 'white',
              cursor: 'pointer',
              fontSize: 12
            }}
          >
            {isAnimating ? '⏸️ Pause' : '▶️ Animate'}
          </button>
          <button
            onClick={fetchQuantumData}
            disabled={loading}
            style={{
              padding: '6px 12px',
              border: '1px solid #2196F3',
              borderRadius: 4,
              backgroundColor: '#2196F3',
              color: 'white',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: 12,
              opacity: loading ? 0.6 : 1
            }}
          >
            🔄 Refresh Data
          </button>
        </div>
      </div>

      {/* Visualization Mode Selector */}
      <div style={{ marginBottom: 20 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Visualization Mode</h4>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(visualizationModes).map(([mode, config]) => (
            <button
              key={mode}
              onClick={() => setVisualizationMode(mode)}
              style={{
                padding: '8px 12px',
                border: `2px solid ${visualizationMode === mode ? config.color : '#ddd'}`,
                borderRadius: 8,
                backgroundColor: visualizationMode === mode ? config.color : 'white',
                color: visualizationMode === mode ? 'white' : '#333',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: visualizationMode === mode ? 'bold' : 'normal',
                transition: 'all 0.2s'
              }}
            >
              {config.icon} {config.name}
            </button>
          ))}
        </div>
        <div style={{ 
          marginTop: 8, 
          padding: 8, 
          backgroundColor: '#f8f9fa', 
          borderRadius: 4,
          fontSize: 12,
          color: '#666'
        }}>
          {visualizationModes[visualizationMode].description}
        </div>
      </div>

      {/* Canvas */}
      <div style={{ 
        border: '1px solid #ddd',
        borderRadius: 8,
        overflow: 'hidden',
        backgroundColor: '#000'
      }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ display: 'block' }}
        />
      </div>

      {/* Status Information */}
      <div style={{ 
        marginTop: 15,
        padding: 10,
        backgroundColor: '#f8f9fa',
        borderRadius: 4,
        fontSize: 12
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
          <div>
            <strong>Quantum State:</strong> {rhoId.substring(0, 8)}...
          </div>
          <div>
            <strong>Visualization:</strong> {visualizationModes[visualizationMode].name}
          </div>
          <div>
            <strong>Animation:</strong> {isAnimating ? 'Running' : 'Paused'}
          </div>
          {quantumData && (
            <div>
              <strong>Entropy:</strong> {quantumData.diagnostics?.entropy?.toFixed(3) || 'N/A'}
            </div>
          )}
          {quantumData && (
            <div>
              <strong>Purity:</strong> {quantumData.diagnostics?.purity?.toFixed(3) || 'N/A'}
            </div>
          )}
          <div>
            <strong>Data Points:</strong> {embeddingTrajectory.length}
          </div>
          {quantumData?.povm && (
            <div>
              <strong>POVM Measurements:</strong> {Object.keys(quantumData.povm.measurements || {}).length}
            </div>
          )}
        </div>
      </div>

      {/* Description */}
      <div style={{ 
        marginTop: 15,
        padding: 15,
        backgroundColor: '#e8f4f8',
        borderRadius: 8,
        fontSize: 12,
        fontStyle: 'italic',
        color: '#555'
      }}>
        <strong>Liminal Space Visualization:</strong> This interface reveals the quantum mechanical
        underpinnings of narrative consciousness, showing how meaning emerges from the probabilistic
        superposition of semantic states. Each visualization mode exposes different aspects of the
        "space between words" where quantum coherence mediates the transition from textual input
        to conscious understanding.
      </div>
    </div>
  );
}

export default LiminalSpaceVisualizer;