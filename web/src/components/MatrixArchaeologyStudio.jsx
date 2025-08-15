import React, { useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../utils/api.js';

/**
 * Matrix Archaeology Studio
 * 
 * A sophisticated interface for exploring, analyzing, and synthesizing your
 * accumulated density matrices. Transform byproduct matrices into tools for
 * creative discovery and synthesis.
 * 
 * Features:
 * - Matrix library management and registration
 * - Quality assessment and best work identification  
 * - Similarity analysis and clustering visualization
 * - Creative synthesis recommendations and execution
 * - Temporal exploration of creative evolution
 */
export function MatrixArchaeologyStudio() {
  // Core state
  const [matrices, setMatrices] = useState([]);
  const [selectedMatrices, setSelectedMatrices] = useState(new Set());
  const [activeView, setActiveView] = useState('library'); // library, analysis, synthesis, timeline
  
  // Analysis state
  const [collectionAnalysis, setCollectionAnalysis] = useState(null);
  const [qualityAssessments, setQualityAssessments] = useState({});
  const [bestWork, setBestWork] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [selectedCriteria, setSelectedCriteria] = useState('overall');
  const [selectedSynthesisMethod, setSelectedSynthesisMethod] = useState('convex_combination');
  const [synthesisWeights, setSynthesisWeights] = useState([]);
  
  // Load available matrices
  const loadMatrices = useCallback(async () => {
    try {
      const response = await fetch(apiUrl('/rho/global/status'));
      if (response.ok) {
        const data = await response.json();
        setMatrices(data.matrices || []);
      }
    } catch (error) {
      console.error('Failed to load matrices:', error);
    }
  }, []);
  
  // Register matrix in library (placeholder - not implemented yet)
  const registerMatrix = useCallback(async (matrix) => {
    alert('Matrix library registration coming soon! This will allow you to tag and categorize matrices for better organization.');
    console.log('Register matrix:', matrix);
  }, []);
  
  // Load library status (placeholder)
  const loadLibraryStatus = useCallback(async () => {
    // Placeholder for future library functionality
    console.log('Library status check - feature coming soon');
  }, []);
  
  // Analyze collection (placeholder)
  const analyzeCollection = useCallback(async () => {
    setLoading(true);
    setTimeout(() => {
      alert('Advanced collection analysis coming soon! This will use machine learning to find patterns and clusters in your matrix collection.');
      setLoading(false);
    }, 1000);
  }, []);
  
  // Find best work (placeholder)
  const findBestWork = useCallback(async (criteria = 'overall') => {
    alert(`Best work finder coming soon! This will analyze your matrices by ${criteria} criteria and rank them.`);
  }, []);
  
  // Get synthesis recommendations (placeholder)
  const getSynthesisRecommendations = useCallback(async () => {
    alert('AI-powered synthesis recommendations coming soon! This will suggest creative combinations of your matrices.');
  }, []);
  
  // Synthesize matrices (placeholder)
  const synthesizeMatrices = useCallback(async (matrixIds, method, weights = null) => {
    setLoading(true);
    setTimeout(() => {
      alert(`Matrix synthesis coming soon! This will combine ${matrixIds.length} matrices using ${method} method.`);
      setLoading(false);
    }, 1000);
  }, []);
  
  // Quality assessment (placeholder)
  const assessQuality = useCallback(async (matrixId) => {
    alert('Quality assessment coming soon! This will analyze matrix complexity, novelty, and narrative depth.');
  }, []);
  
  // Initialize
  useEffect(() => {
    loadMatrices();
    loadLibraryStatus();
  }, [loadMatrices, loadLibraryStatus]);
  
  // Render matrix library view
  const renderLibraryView = () => (
    <div style={{ padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h3 style={{ margin: 0 }}>📚 Matrix Library ({matrices.length} matrices)</h3>
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={loadMatrices}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer'
            }}
          >
            🔄 Refresh
          </button>
          <button
            onClick={analyzeCollection}
            disabled={matrices.length < 2}
            style={{
              padding: '8px 16px',
              backgroundColor: matrices.length >= 2 ? '#FF9800' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: matrices.length >= 2 ? 'pointer' : 'not-allowed'
            }}
          >
            🔬 Analyze Collection
          </button>
        </div>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 15 }}>
        {matrices.map(matrix => (
          <div
            key={matrix.rho_id}
            style={{
              border: `2px solid ${selectedMatrices.has(matrix.rho_id) ? '#9C27B0' : '#ddd'}`,
              borderRadius: 8,
              padding: 15,
              backgroundColor: selectedMatrices.has(matrix.rho_id) ? '#f3e5f5' : 'white',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onClick={() => {
              const newSelection = new Set(selectedMatrices);
              if (newSelection.has(matrix.rho_id)) {
                newSelection.delete(matrix.rho_id);
              } else {
                newSelection.add(matrix.rho_id);
              }
              setSelectedMatrices(newSelection);
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <h4 style={{ margin: 0, fontSize: 14, fontWeight: 'bold' }}>
                {matrix.label}
              </h4>
              <div style={{ fontSize: 11, color: '#666' }}>
                Purity: {matrix.purity?.toFixed(3)}
              </div>
            </div>
            
            <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
              Narratives: {matrix.narratives_count} | Operations: {matrix.operations_count}
            </div>
            
            <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
              Entropy: {matrix.entropy?.toFixed(3)} | Top λ: {matrix.eigenvals?.[0]?.toFixed(3)}
            </div>
            
            <div style={{ display: 'flex', gap: 5, marginBottom: 10, flexWrap: 'wrap' }}>
              <span style={{
                padding: '2px 6px',
                backgroundColor: '#e3f2fd',
                color: '#1565C0',
                borderRadius: 4,
                fontSize: 10
              }}>
                {matrix.rho_id.substring(0, 8)}
              </span>
              {matrix.dual_pair && (
                <span style={{
                  padding: '2px 6px',
                  backgroundColor: '#f3e5f5',
                  color: '#7B1FA2',
                  borderRadius: 4,
                  fontSize: 10
                }}>
                  dual-paired
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', gap: 5 }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  registerMatrix(matrix);
                }}
                style={{
                  padding: '4px 8px',
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  fontSize: 11,
                  cursor: 'pointer'
                }}
              >
                📋 Register
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  assessQuality(matrix.rho_id);
                }}
                style={{
                  padding: '4px 8px',
                  backgroundColor: '#FF9800',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  fontSize: 11,
                  cursor: 'pointer'
                }}
              >
                ⭐ Assess
              </button>
            </div>
            
            {qualityAssessments[matrix.rho_id] && (
              <div style={{ marginTop: 10, padding: 8, backgroundColor: '#f5f5f5', borderRadius: 4 }}>
                <div style={{ fontSize: 11, fontWeight: 'bold' }}>
                  Quality Score: {(qualityAssessments[matrix.rho_id].overall_score * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 10, color: '#666' }}>
                  {qualityAssessments[matrix.rho_id].assessment_rationale}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      
      {selectedMatrices.size > 0 && (
        <div style={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          padding: 15,
          backgroundColor: '#9C27B0',
          color: 'white',
          borderRadius: 8,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        }}>
          <div style={{ marginBottom: 10 }}>
            {selectedMatrices.size} matrices selected
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => setActiveView('synthesis')}
              style={{
                padding: '8px 12px',
                backgroundColor: 'white',
                color: '#9C27B0',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12
              }}
            >
              ⚗️ Synthesize
            </button>
            <button
              onClick={() => setSelectedMatrices(new Set())}
              style={{
                padding: '8px 12px',
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                border: '1px solid white',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12
              }}
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
  
  // Render analysis view
  const renderAnalysisView = () => (
    <div style={{ padding: 20 }}>
      <h3 style={{ margin: '0 0 20px 0' }}>📊 Collection Analysis</h3>
      
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <select
          value={selectedCriteria}
          onChange={(e) => setSelectedCriteria(e.target.value)}
          style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd' }}
        >
          <option value="overall">Overall Quality</option>
          <option value="complexity">Complexity</option>
          <option value="novelty">Novelty</option>
          <option value="depth">Depth</option>
          <option value="resonance">Emotional Resonance</option>
        </select>
        <button
          onClick={() => findBestWork(selectedCriteria)}
          style={{
            padding: '8px 16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer'
          }}
        >
          🏆 Find Best Work
        </button>
        <button
          onClick={getSynthesisRecommendations}
          style={{
            padding: '8px 16px',
            backgroundColor: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer'
          }}
        >
          💡 Get Recommendations
        </button>
      </div>
      
      {bestWork.length > 0 && (
        <div style={{ marginBottom: 30 }}>
          <h4>🏆 Best Work ({selectedCriteria})</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 10 }}>
            {bestWork.map((work, index) => (
              <div key={work.rho_id} style={{
                padding: 12,
                border: '1px solid #ddd',
                borderRadius: 6,
                backgroundColor: index < 3 ? '#fff3e0' : 'white'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 'bold', fontSize: 13 }}>
                    #{index + 1} {work.label}
                  </span>
                  <span style={{ 
                    padding: '2px 6px',
                    backgroundColor: index < 3 ? '#ff9800' : '#2196F3',
                    color: 'white',
                    borderRadius: 3,
                    fontSize: 11
                  }}>
                    {(work.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 5 }}>
                  {work.source_type} • {work.content_preview.substring(0, 50)}...
                </div>
                <div style={{ fontSize: 10, color: '#999', marginTop: 5 }}>
                  {work.tags.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {recommendations.length > 0 && (
        <div style={{ marginBottom: 30 }}>
          <h4>💡 Synthesis Recommendations</h4>
          <div style={{ display: 'grid', gap: 15 }}>
            {recommendations.map((rec, index) => (
              <div key={index} style={{
                padding: 15,
                border: '1px solid #ddd',
                borderRadius: 8,
                backgroundColor: 'white'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                  <div>
                    <h5 style={{ margin: 0, color: '#333' }}>{rec.type.replace('_', ' ').toUpperCase()}</h5>
                    <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>
                      {rec.rationale}
                    </div>
                  </div>
                  <button
                    onClick={() => synthesizeMatrices(rec.matrices, rec.method, rec.weights)}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#9C27B0',
                      color: 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontSize: 11
                    }}
                  >
                    🚀 Execute
                  </button>
                </div>
                
                <div style={{ fontSize: 12, marginBottom: 8 }}>
                  <strong>Method:</strong> {rec.method} | <strong>Expected:</strong> {rec.expected_outcome}
                </div>
                
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {rec.matrices_info?.map(info => (
                    <span key={info.rho_id} style={{
                      padding: '4px 8px',
                      backgroundColor: '#e3f2fd',
                      borderRadius: 4,
                      fontSize: 11,
                      color: '#1565C0'
                    }}>
                      {info.label} ({info.source_type})
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {collectionAnalysis && (
        <div>
          <h4>🔬 Collection Analysis Results</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
            <div style={{ padding: 15, border: '1px solid #ddd', borderRadius: 8 }}>
              <h5>Similarity Matrix</h5>
              <div style={{ fontSize: 12, color: '#666' }}>
                {collectionAnalysis.matrix_ids?.length || 0} matrices analyzed
              </div>
              {/* Simplified similarity visualization */}
              <div style={{ marginTop: 10, fontSize: 11 }}>
                Cluster analysis available for {collectionAnalysis.matrix_ids?.length || 0} works
              </div>
            </div>
            
            <div style={{ padding: 15, border: '1px solid #ddd', borderRadius: 8 }}>
              <h5>Clusters Found</h5>
              {Object.entries(collectionAnalysis.clusters || {}).map(([clusterName, clusters]) => (
                <div key={clusterName} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 'bold' }}>{clusterName}</div>
                  {Object.entries(clusters).map(([clusterId, matrices]) => (
                    <div key={clusterId} style={{ fontSize: 11, color: '#666', marginLeft: 10 }}>
                      {clusterId}: {matrices.length} matrices
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  // Render synthesis view
  const renderSynthesisView = () => (
    <div style={{ padding: 20 }}>
      <h3 style={{ margin: '0 0 20px 0' }}>⚗️ Matrix Synthesis Laboratory</h3>
      
      {selectedMatrices.size > 0 && (
        <div style={{ marginBottom: 30 }}>
          <h4>Selected Matrices ({selectedMatrices.size})</h4>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
            {Array.from(selectedMatrices).map(matrixId => {
              const matrix = matrices.find(m => m.rho_id === matrixId);
              return (
                <span key={matrixId} style={{
                  padding: '6px 12px',
                  backgroundColor: '#e3f2fd',
                  borderRadius: 6,
                  fontSize: 12,
                  color: '#1565C0'
                }}>
                  {matrix?.label || matrixId.substring(0, 8)}
                </span>
              );
            })}
          </div>
          
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, fontSize: 14, fontWeight: 'bold' }}>
              Synthesis Method:
            </label>
            <select
              value={selectedSynthesisMethod}
              onChange={(e) => setSelectedSynthesisMethod(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', width: 300 }}
            >
              <option value="convex_combination">Convex Combination (Balanced Blend)</option>
              <option value="geometric_mean">Geometric Mean (Deep Fusion)</option>
              <option value="coherent_superposition">Coherent Superposition (Quantum Interference)</option>
              <option value="interference_pattern">Interference Pattern (Creative Collision)</option>
            </select>
          </div>
          
          <button
            onClick={() => synthesizeMatrices(Array.from(selectedMatrices), selectedSynthesisMethod)}
            disabled={loading}
            style={{
              padding: '12px 24px',
              backgroundColor: loading ? '#ccc' : '#9C27B0',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 'bold',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? '⚗️ Synthesizing...' : '🚀 Create Synthesis'}
          </button>
        </div>
      )}
      
      <div style={{ fontSize: 14, color: '#666' }}>
        Select matrices from the Library view to begin synthesis, or return to Analysis for recommendations.
      </div>
    </div>
  );
  
  // Main render
  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
      <div style={{
        textAlign: 'center',
        marginBottom: 30,
        borderBottom: '2px solid #eee',
        paddingBottom: 20
      }}>
        <h1 style={{
          margin: 0,
          fontSize: 24,
          color: '#333',
          background: 'linear-gradient(45deg, #9C27B0, #FF9800)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          🏛️ Matrix Archaeology Studio
        </h1>
        <div style={{ fontSize: 14, color: '#666', marginTop: 8 }}>
          Transform your accumulated density matrices into tools for creative synthesis and discovery
        </div>
      </div>
      
      {/* Navigation */}
      <div style={{ display: 'flex', gap: 5, marginBottom: 30, justifyContent: 'center' }}>
        {[
          { id: 'library', label: '📚 Library', desc: 'Manage matrices' },
          { id: 'analysis', label: '📊 Analysis', desc: 'Find patterns' },
          { id: 'synthesis', label: '⚗️ Synthesis', desc: 'Create new work' }
        ].map(view => (
          <button
            key={view.id}
            onClick={() => setActiveView(view.id)}
            style={{
              padding: '12px 20px',
              backgroundColor: activeView === view.id ? '#9C27B0' : 'white',
              color: activeView === view.id ? 'white' : '#333',
              border: `2px solid ${activeView === view.id ? '#9C27B0' : '#ddd'}`,
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 'bold',
              transition: 'all 0.2s'
            }}
          >
            <div>{view.label}</div>
            <div style={{ fontSize: 10, opacity: 0.8 }}>{view.desc}</div>
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div style={{
        backgroundColor: 'white',
        borderRadius: 12,
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        minHeight: 500
      }}>
        {activeView === 'library' && renderLibraryView()}
        {activeView === 'analysis' && renderAnalysisView()}
        {activeView === 'synthesis' && renderSynthesisView()}
      </div>
    </div>
  );
}

export default MatrixArchaeologyStudio;