import React, { useState, useEffect, useCallback } from 'react';
import { LiminalSpaceVisualizer } from './LiminalSpaceVisualizer.jsx';
import { apiUrl } from '../utils/api.js';

/**
 * Liminal Space Tab - Post-Lexical Grammatological Laboratory
 * 
 * Comprehensive interface for exploring the quantum-mechanical underpinnings
 * of narrative consciousness through various visualization modes.
 */
export function LiminalSpaceTab() {
  const [availableMatrices, setAvailableMatrices] = useState([]);
  const [selectedMatrix, setSelectedMatrix] = useState(null);
  const [loading, setLoading] = useState(false);
  const [quantumAnalysis, setQuantumAnalysis] = useState(null);
  const [embeddingData, setEmbeddingData] = useState(null);

  // Fetch available matrices
  const fetchAvailableMatrices = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(apiUrl('/rho/global/status'));
      if (response.ok) {
        const data = await response.json();
        setAvailableMatrices(data.matrices || []);
        
        // Auto-select first matrix if none selected
        if (!selectedMatrix && data.matrices.length > 0) {
          setSelectedMatrix(data.matrices[0].rho_id);
        }
      }
    } catch (error) {
      console.error('Failed to fetch available matrices:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedMatrix]);

  // Fetch quantum analysis for selected matrix
  const fetchQuantumAnalysis = useCallback(async () => {
    if (!selectedMatrix) return;
    
    try {
      // Get matrix diagnostics
      const rhoResponse = await fetch(apiUrl(`/rho/${selectedMatrix}`));
      if (rhoResponse.ok) {
        const rhoData = await rhoResponse.json();
        
        // Get channel health if available
        let healthData = null;
        try {
          const healthResponse = await fetch(apiUrl(`/audit/channel_health/${selectedMatrix}`));
          if (healthResponse.ok) {
            healthData = await healthResponse.json();
          }
        } catch (e) {
          console.log('No health data available');
        }
        
        // Get POVM measurements to understand matrix formation
        let povmData = null;
        try {
          const povmResponse = await fetch(apiUrl(`/packs/measure/${selectedMatrix}`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pack_id: 'advanced_narrative_pack' })
          });
          if (povmResponse.ok) {
            povmData = await povmResponse.json();
          }
        } catch (e) {
          console.log('No POVM data available');
        }
        
        // Get integrability metrics if available
        let integrabilityData = null;
        try {
          const integrabilityResponse = await fetch(apiUrl('/integrability/metrics'));
          if (integrabilityResponse.ok) {
            integrabilityData = await integrabilityResponse.json();
          }
        } catch (e) {
          console.log('No integrability data available');
        }
        
        setQuantumAnalysis({
          matrix: rhoData,
          health: healthData,
          integrability: integrabilityData,
          povm: povmData
        });
      }
    } catch (error) {
      console.error('Failed to fetch quantum analysis:', error);
    }
  }, [selectedMatrix]);

  // Generate embedding visualization data
  const generateEmbeddingData = useCallback(async () => {
    if (!selectedMatrix || !quantumAnalysis) return;
    
    try {
      // Use eigenvalues to create embedding-like visualization
      const eigs = quantumAnalysis.matrix.diagnostics?.eigs || [];
      const data = [];
      
      for (let i = 0; i < Math.min(eigs.length, 100); i++) {
        const eigenValue = eigs[i] || 0;
        const angle = (i / 100) * 2 * Math.PI;
        const radius = eigenValue * 150;
        
        data.push({
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: eigenValue,
          magnitude: eigenValue,
          index: i,
          normalized: eigenValue / (eigs[0] || 1)
        });
      }
      
      setEmbeddingData(data);
    } catch (error) {
      console.error('Failed to generate embedding data:', error);
    }
  }, [selectedMatrix, quantumAnalysis]);

  // Load matrices on component mount
  useEffect(() => {
    fetchAvailableMatrices();
  }, [fetchAvailableMatrices]);

  // Fetch analysis when matrix changes
  useEffect(() => {
    fetchQuantumAnalysis();
  }, [fetchQuantumAnalysis]);

  // Generate embedding data when analysis changes
  useEffect(() => {
    generateEmbeddingData();
  }, [generateEmbeddingData]);

  // Render matrix selector
  const renderMatrixSelector = () => (
    <div style={{ 
      marginBottom: 20,
      padding: 15,
      border: '1px solid #ddd',
      borderRadius: 8,
      backgroundColor: '#f8f9fa'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Select Quantum State for Liminal Analysis</h4>
      
      {loading ? (
        <div style={{ fontSize: 12, color: '#666' }}>Loading matrices...</div>
      ) : availableMatrices.length === 0 ? (
        <div style={{ fontSize: 12, color: '#666' }}>
          No quantum states available. Create a matrix first using the Book Reader or Live Narrative tabs.
        </div>
      ) : (
        <div>
          <select
            value={selectedMatrix || ''}
            onChange={(e) => setSelectedMatrix(e.target.value)}
            style={{
              width: '100%',
              padding: 8,
              border: '1px solid #ddd',
              borderRadius: 4,
              fontSize: 12,
              marginBottom: 10
            }}
          >
            <option value="">-- Select a quantum state --</option>
            {availableMatrices.map(matrix => (
              <option key={matrix.rho_id} value={matrix.rho_id}>
                {matrix.label} ({matrix.narratives_count} narratives)
              </option>
            ))}
          </select>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
            {availableMatrices.slice(0, 6).map(matrix => (
              <div 
                key={matrix.rho_id}
                onClick={() => setSelectedMatrix(matrix.rho_id)}
                style={{
                  padding: 10,
                  border: `2px solid ${selectedMatrix === matrix.rho_id ? '#9C27B0' : '#ddd'}`,
                  borderRadius: 6,
                  cursor: 'pointer',
                  backgroundColor: selectedMatrix === matrix.rho_id ? '#f3e5f5' : 'white',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: 12, marginBottom: 4 }}>
                  {matrix.label}
                </div>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 2 }}>
                  {matrix.narratives_count} narratives
                </div>
                <div style={{ fontSize: 11, color: '#666' }}>
                  {matrix.operations_count} operations
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <button
        onClick={fetchAvailableMatrices}
        style={{
          marginTop: 10,
          padding: '6px 12px',
          border: '1px solid #666',
          borderRadius: 4,
          backgroundColor: 'white',
          cursor: 'pointer',
          fontSize: 12
        }}
      >
        🔄 Refresh States
      </button>
    </div>
  );

  // Render quantum analysis panel
  const renderQuantumAnalysis = () => {
    if (!quantumAnalysis) return null;

    const matrix = quantumAnalysis.matrix;
    const health = quantumAnalysis.health;

    return (
      <div style={{ 
        marginBottom: 20,
        padding: 15,
        border: '1px solid #ddd',
        borderRadius: 8,
        backgroundColor: '#f8fff8'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Quantum State Analysis</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
          <div style={{ fontSize: 11 }}>
            <strong>Trace:</strong><br/>
            {matrix.diagnostics?.trace?.toFixed(6) || 'N/A'}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Purity:</strong><br/>
            {matrix.diagnostics?.purity?.toFixed(4) || 'N/A'}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Entropy:</strong><br/>
            {matrix.diagnostics?.entropy?.toFixed(4) || 'N/A'}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Eigenvalues:</strong><br/>
            {matrix.diagnostics?.eigs?.length || 0} computed
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Max Eigenvalue:</strong><br/>
            {matrix.diagnostics?.eigs?.[0]?.toExponential(3) || 'N/A'}
          </div>
          {health && (
            <div style={{ fontSize: 11 }}>
              <strong>Health Status:</strong><br/>
              <span style={{ 
                color: health.is_healthy ? '#4CAF50' : '#f44336',
                fontWeight: 'bold'
              }}>
                {health.is_healthy ? '✓ Healthy' : '✗ Issues'}
              </span>
            </div>
          )}
        </div>

        {matrix.diagnostics?.eigs && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 'bold', marginBottom: 5 }}>
              Top Eigenvalues (Liminal Amplitudes):
            </div>
            <div style={{ 
              display: 'flex', 
              gap: 4, 
              flexWrap: 'wrap',
              fontSize: 10,
              fontFamily: 'monospace'
            }}>
              {matrix.diagnostics.eigs.slice(0, 16).map((eig, i) => (
                <span 
                  key={i}
                  style={{ 
                    padding: '2px 4px',
                    backgroundColor: `hsla(${i * 22.5}, 60%, 80%, 0.6)`,
                    borderRadius: 3,
                    border: '1px solid #ddd'
                  }}
                >
                  λ{i}: {eig.toExponential(2)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render liminal space insights
  const renderLiminalInsights = () => {
    if (!quantumAnalysis) return null;

    const matrix = quantumAnalysis.matrix;
    const entropy = matrix.diagnostics?.entropy || 0;
    const purity = matrix.diagnostics?.purity || 0;
    const eigs = matrix.diagnostics?.eigs || [];

    // Calculate liminal metrics
    const coherenceIndex = purity;
    const entanglementDegree = entropy / Math.log2(64); // Normalized to max possible entropy
    const quantumness = 1 - purity; // How "quantum" vs classical the state is
    const dimensionality = eigs.filter(e => e > 1e-6).length; // Effective dimension

    return (
      <div style={{ 
        marginBottom: 20,
        padding: 15,
        border: '1px solid #9C27B0',
        borderRadius: 8,
        backgroundColor: '#faf5ff'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14, color: '#9C27B0' }}>
          🌌 Liminal Space Insights
        </h4>
        
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 12, marginBottom: 8 }}>
            <strong>Narrative Consciousness Metrics:</strong>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
            <div style={{ 
              padding: 8,
              backgroundColor: 'white',
              borderRadius: 4,
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: 11, fontWeight: 'bold', color: '#9C27B0' }}>
                Coherence Index
              </div>
              <div style={{ fontSize: 14, fontWeight: 'bold' }}>
                {(coherenceIndex * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 10, color: '#666' }}>
                Classical vs quantum superposition
              </div>
            </div>
            
            <div style={{ 
              padding: 8,
              backgroundColor: 'white',
              borderRadius: 4,
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: 11, fontWeight: 'bold', color: '#9C27B0' }}>
                Entanglement Degree
              </div>
              <div style={{ fontSize: 14, fontWeight: 'bold' }}>
                {(entanglementDegree * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 10, color: '#666' }}>
                Semantic interconnectedness
              </div>
            </div>
            
            <div style={{ 
              padding: 8,
              backgroundColor: 'white',
              borderRadius: 4,
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: 11, fontWeight: 'bold', color: '#9C27B0' }}>
                Quantumness
              </div>
              <div style={{ fontSize: 14, fontWeight: 'bold' }}>
                {(quantumness * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 10, color: '#666' }}>
                Non-classical narrative effects
              </div>
            </div>
            
            <div style={{ 
              padding: 8,
              backgroundColor: 'white',
              borderRadius: 4,
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: 11, fontWeight: 'bold', color: '#9C27B0' }}>
                Effective Dimensionality
              </div>
              <div style={{ fontSize: 14, fontWeight: 'bold' }}>
                {dimensionality}D
              </div>
              <div style={{ fontSize: 10, color: '#666' }}>
                Active semantic dimensions
              </div>
            </div>
          </div>
        </div>

        <div style={{ 
          fontSize: 11,
          fontStyle: 'italic',
          color: '#666',
          padding: 8,
          backgroundColor: '#f0f0f0',
          borderRadius: 4
        }}>
          <strong>Interpretation:</strong> This quantum state represents {
            coherenceIndex > 0.8 ? 'a highly coherent narrative with classical-like properties' :
            coherenceIndex > 0.5 ? 'a balanced superposition of narrative possibilities' :
            'a deeply quantum narrative with significant superposition effects'
          }. The {entanglementDegree > 0.7 ? 'high' : entanglementDegree > 0.4 ? 'moderate' : 'low'} entanglement
          suggests {entanglementDegree > 0.7 ? 'complex semantic interconnections' : 
          entanglementDegree > 0.4 ? 'moderate thematic coherence' : 'relatively independent narrative elements'}.
        </div>
      </div>
    );
  };

  // Render matrix formation analysis
  const renderMatrixFormationAnalysis = () => {
    if (!quantumAnalysis?.povm) return null;

    const measurements = quantumAnalysis.povm.measurements;
    
    // Analyze narrative characteristics from POVM data
    const formationProfile = {
      narrative_type: measurements.narrative_concerns_narrative > 0.5 ? 'Narrative-focused' : 'Information-focused',
      complexity: measurements.cognitive_load_complex > 0.5 ? 'Complex' : 'Simple',
      formality: measurements.tenor_formality_formal > 0.5 ? 'Formal' : 'Informal',
      temporal_orientation: measurements.temporal_perspective_retrospective > measurements.temporal_perspective_prospective ? 'Retrospective' : 'Prospective',
      engagement_style: measurements.reader_engagement_engaging > 0.5 ? 'Engaging' : 'Passive',
      discourse_coherence: measurements.discourse_coherence_tight > 0.5 ? 'Cohesive' : 'Loose'
    };

    const significantMeasurements = Object.entries(measurements)
      .filter(([key, value]) => value > 0.02) // Show measurements above threshold
      .sort(([,a], [,b]) => b - a) // Sort by magnitude
      .slice(0, 8); // Top 8 measurements

    return (
      <div style={{ 
        marginBottom: 20,
        padding: 15,
        border: '1px solid #FF9800',
        borderRadius: 8,
        backgroundColor: '#fff8e1'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14, color: '#F57C00' }}>
          🔬 Matrix Formation Analysis
        </h4>
        
        <div style={{ marginBottom: 15 }}>
          <div style={{ fontSize: 12, marginBottom: 8 }}>
            <strong>Narrative Profile:</strong>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
            {Object.entries(formationProfile).map(([aspect, value]) => (
              <div key={aspect} style={{ 
                padding: 6,
                backgroundColor: 'white',
                borderRadius: 4,
                border: '1px solid #ddd',
                fontSize: 11
              }}>
                <div style={{ fontWeight: 'bold', color: '#F57C00' }}>
                  {aspect.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 10, color: '#666' }}>{value}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 'bold', marginBottom: 8 }}>
            Dominant Formation Vectors:
          </div>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: 6,
            fontSize: 10,
            fontFamily: 'monospace'
          }}>
            {significantMeasurements.map(([key, value], index) => (
              <div key={key} style={{ 
                padding: 4,
                backgroundColor: `hsla(${30 + index * 20}, 70%, 95%, 0.8)`,
                borderRadius: 3,
                border: '1px solid #ddd'
              }}>
                <span style={{ fontWeight: 'bold' }}>
                  {key.replace(/_/g, ' ')}: 
                </span>
                <span style={{ color: '#F57C00', fontWeight: 'bold' }}>
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: 20 }}>
      <div style={{ 
        marginBottom: 20,
        borderBottom: '2px solid #eee',
        paddingBottom: 15
      }}>
        <h2 style={{ margin: 0, fontSize: 20, color: '#333' }}>
          🌌 Liminal Space Explorer
        </h2>
        <div style={{ fontSize: 14, color: '#666', marginTop: 5 }}>
          Visualize the quantum-mechanical underpinnings of narrative consciousness
        </div>
      </div>

      {renderMatrixSelector()}
      {renderQuantumAnalysis()}
      {renderMatrixFormationAnalysis()}
      {renderLiminalInsights()}
      
      <LiminalSpaceVisualizer 
        rhoId={selectedMatrix}
        width={800}
        height={600}
      />
    </div>
  );
}

export default LiminalSpaceTab;