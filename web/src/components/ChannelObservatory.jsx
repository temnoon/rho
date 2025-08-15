import React, { useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../utils/api.js';

/**
 * Channel Observatory - Post-Lexical Grammatological Laboratory
 * 
 * Provides comprehensive monitoring and interaction with the quantum channel system.
 * Implements the "Analytic Post-Lexical Grammatology" framework for observing
 * the liminal space between observable words through proper quantum channel theory.
 */
export function ChannelObservatory({ rhoId, onChannelApplied }) {
  const [channelAudit, setChannelAudit] = useState(null);
  const [channelHealth, setChannelHealth] = useState(null);
  const [channelType, setChannelType] = useState('rank_one_update');
  const [testSegments, setTestSegments] = useState(['']);
  const [integrabilityResult, setIntegrabilityResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [auditHistory, setAuditHistory] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Channel type configurations
  const channelTypes = {
    rank_one_update: {
      name: 'Rank-One Update',
      description: 'Factual/expository text with CPTP guarantees',
      color: '#4CAF50',
      icon: '📝'
    },
    coherent_rotation: {
      name: 'Coherent Rotation', 
      description: 'Perspective shifts (unitary, entropy-preserving)',
      color: '#2196F3',
      icon: '🔄'
    },
    dephasing_mixture: {
      name: 'Dephasing Mixture',
      description: 'Ambiguous text with multiple interpretations',
      color: '#FF9800',
      icon: '🌀'
    }
  };

  // Health check thresholds
  const healthThresholds = {
    trace_error: 1e-8,
    psd_violation: -1e-10,
    integrability_tolerance: 1e-6,
    max_commutator: 0.1
  };

  // Fetch channel health status
  const fetchChannelHealth = useCallback(async () => {
    if (!rhoId) return;
    
    try {
      const response = await fetch(apiUrl(`/audit/channel_health/${rhoId}`));
      if (response.ok) {
        const health = await response.json();
        setChannelHealth(health);
      }
    } catch (error) {
      console.error('Failed to fetch channel health:', error);
    }
  }, [rhoId]);

  // Run comprehensive channel audit
  const runChannelAudit = useCallback(async () => {
    if (!rhoId) return;
    
    setLoading(true);
    try {
      const auditRequest = {
        rho_id: rhoId,
        test_segments: testSegments.filter(s => s.trim()),
        check_integrability: true,
        check_commutativity: true
      };

      const response = await fetch(apiUrl(`/audit/sanity_check/${rhoId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(auditRequest)
      });

      if (response.ok) {
        const audit = await response.json();
        setChannelAudit(audit);
        
        // Add to audit history
        setAuditHistory(prev => [{
          timestamp: new Date().toISOString(),
          ...audit
        }, ...prev.slice(0, 4)]); // Keep last 5 audits
      }
    } catch (error) {
      console.error('Channel audit failed:', error);
    } finally {
      setLoading(false);
    }
  }, [rhoId, testSegments]);

  // Repair matrix if failing sanity checks
  const repairMatrix = useCallback(async () => {
    if (!rhoId) return;
    
    setLoading(true);
    try {
      const response = await fetch(apiUrl(`/audit/repair_matrix/${rhoId}`), {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Matrix repaired:', result);
        
        // Refresh health and audit after repair
        await fetchChannelHealth();
        await runChannelAudit();
        
        if (onChannelApplied) {
          onChannelApplied('repair', result);
        }
      }
    } catch (error) {
      console.error('Matrix repair failed:', error);
    } finally {
      setLoading(false);
    }
  }, [rhoId, fetchChannelHealth, runChannelAudit, onChannelApplied]);

  // Test integrability with different segmentations
  const testIntegrability = useCallback(async () => {
    if (!testSegments.filter(s => s.trim()).length) return;
    
    setLoading(true);
    try {
      const segments = testSegments.filter(s => s.trim());
      
      if (segments.length < 2) {
        setIntegrabilityResult({
          segments_tested: segments.length,
          passes_test: false,
          bures_distance: 0,
          recommendations: ['Add at least 2 segments to test integrability']
        });
        return;
      }
      
      // Split segments into two groups for integrability testing
      const midpoint = Math.ceil(segments.length / 2);
      const segments_a = segments.slice(0, midpoint);
      const segments_b = segments.slice(midpoint);
      
      const response = await fetch(apiUrl('/integrability/test_segmentations'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segments_a: segments_a,
          segments_b: segments_b,
          channel_type: channelType,
          alpha: 0.3,
          tolerance: 1e-6
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setIntegrabilityResult({
          segments_tested: result.segments_a_count + result.segments_b_count,
          passes_test: result.passes_test,
          bures_distance: result.bures_distance,
          trace_distance: result.trace_distance,
          fidelity: result.fidelity,
          recommendations: result.recommendations,
          tolerance: result.tolerance,
          quantum_metrics: result.quantum_metrics
        });
      } else {
        throw new Error('Integrability test request failed');
      }
    } catch (error) {
      console.error('Integrability test failed:', error);
      setIntegrabilityResult({
        segments_tested: 0,
        passes_test: false,
        bures_distance: Infinity,
        recommendations: [`Test failed: ${error.message}`]
      });
    } finally {
      setLoading(false);
    }
  }, [testSegments, channelType]);

  // Auto-refresh health status
  useEffect(() => {
    fetchChannelHealth();
    const interval = setInterval(fetchChannelHealth, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, [fetchChannelHealth]);

  // Render health indicator
  const renderHealthIndicator = (value, threshold, invert = false) => {
    const isHealthy = invert ? value <= threshold : value >= threshold;
    return (
      <span style={{
        padding: '2px 6px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 'bold',
        color: 'white',
        backgroundColor: isHealthy ? '#4CAF50' : '#f44336'
      }}>
        {isHealthy ? '✓' : '✗'}
      </span>
    );
  };

  // Render channel type selector
  const renderChannelTypeSelector = () => (
    <div style={{ marginBottom: 15 }}>
      <h4 style={{ margin: '0 0 8px 0', fontSize: 14 }}>Channel Type Selection</h4>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {Object.entries(channelTypes).map(([type, config]) => (
          <button
            key={type}
            onClick={() => setChannelType(type)}
            style={{
              padding: '8px 12px',
              border: `2px solid ${channelType === type ? config.color : '#ddd'}`,
              borderRadius: 8,
              backgroundColor: channelType === type ? config.color : 'white',
              color: channelType === type ? 'white' : '#333',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: channelType === type ? 'bold' : 'normal',
              transition: 'all 0.2s'
            }}
          >
            {config.icon} {config.name}
          </button>
        ))}
      </div>
      {channelType && (
        <div style={{ 
          marginTop: 8, 
          padding: 8, 
          backgroundColor: '#f8f9fa', 
          borderRadius: 4,
          fontSize: 12,
          color: '#666'
        }}>
          {channelTypes[channelType].description}
        </div>
      )}
    </div>
  );

  // Render health dashboard
  const renderHealthDashboard = () => (
    <div style={{ 
      marginBottom: 15,
      padding: 15,
      border: '1px solid #ddd',
      borderRadius: 8,
      backgroundColor: channelHealth?.is_healthy ? '#f8fff8' : '#fff8f8'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
        Channel Health Monitor
        <span style={{
          padding: '2px 8px',
          borderRadius: 12,
          fontSize: 10,
          fontWeight: 'bold',
          backgroundColor: channelHealth?.is_healthy ? '#4CAF50' : '#f44336',
          color: 'white'
        }}>
          {channelHealth?.is_healthy ? 'HEALTHY' : 'NEEDS ATTENTION'}
        </span>
      </h4>
      
      {channelHealth && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
          <div style={{ fontSize: 11 }}>
            <strong>Trace Error:</strong><br/>
            {channelHealth.trace_error.toExponential(2)} {renderHealthIndicator(channelHealth.trace_error, healthThresholds.trace_error, true)}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Min Eigenvalue:</strong><br/>
            {channelHealth.min_eigenvalue.toExponential(2)} {renderHealthIndicator(channelHealth.min_eigenvalue, healthThresholds.psd_violation)}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Purity:</strong><br/>
            {channelHealth.purity.toFixed(4)}
          </div>
          <div style={{ fontSize: 11 }}>
            <strong>Entropy:</strong><br/>
            {channelHealth.entropy.toFixed(4)}
          </div>
        </div>
      )}
      
      <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
        <button 
          onClick={runChannelAudit}
          disabled={loading || !rhoId}
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
          {loading ? 'Running...' : '🔍 Full Audit'}
        </button>
        
        {channelHealth && !channelHealth.is_healthy && (
          <button 
            onClick={repairMatrix}
            disabled={loading}
            style={{
              padding: '6px 12px',
              border: '1px solid #FF9800',
              borderRadius: 4,
              backgroundColor: '#FF9800',
              color: 'white',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: 12,
              opacity: loading ? 0.6 : 1
            }}
          >
            🔧 Repair Matrix
          </button>
        )}
      </div>
    </div>
  );

  // Render audit results
  const renderAuditResults = () => {
    if (!channelAudit) return null;

    return (
      <div style={{ 
        marginBottom: 15,
        padding: 15,
        border: '1px solid #ddd',
        borderRadius: 8,
        backgroundColor: channelAudit.passes_sanity_check ? '#f8fff8' : '#fff8f8'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
          Audit Results
          <span style={{
            padding: '2px 8px',
            borderRadius: 12,
            fontSize: 10,
            fontWeight: 'bold',
            backgroundColor: channelAudit.passes_sanity_check ? '#4CAF50' : '#f44336',
            color: 'white'
          }}>
            {channelAudit.passes_sanity_check ? 'PASS' : 'FAIL'}
          </span>
        </h4>

        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, marginBottom: 5 }}>
            <strong>Trace Preservation:</strong> {channelAudit.trace_preservation_error.toExponential(2)}
          </div>
          <div style={{ fontSize: 11, marginBottom: 5 }}>
            <strong>PSD Violation:</strong> {channelAudit.psd_violation.toExponential(2)}
          </div>
          <div style={{ fontSize: 11, marginBottom: 5 }}>
            <strong>Integrability Error:</strong> {channelAudit.integrability_error.toExponential(2)}
          </div>
        </div>

        {channelAudit.recommendations.length > 0 && (
          <div>
            <strong style={{ fontSize: 12 }}>Recommendations:</strong>
            {channelAudit.recommendations.map((rec, i) => (
              <div key={i} style={{ 
                fontSize: 11, 
                color: rec.startsWith('✓') ? '#4CAF50' : '#666',
                marginLeft: 10,
                marginTop: 2
              }}>
                {rec}
              </div>
            ))}
          </div>
        )}

        {Object.keys(channelAudit.commutator_norms).length > 0 && (
          <div style={{ marginTop: 10 }}>
            <strong style={{ fontSize: 12 }}>Commutator Analysis:</strong>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 4, marginTop: 4 }}>
              {Object.entries(channelAudit.commutator_norms).map(([pair, norm]) => (
                <div key={pair} style={{ fontSize: 10, padding: 2 }}>
                  <strong>{pair}:</strong> {norm.toFixed(4)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render integrability testing
  const renderIntegrabilityTesting = () => (
    <div style={{ 
      marginBottom: 15,
      padding: 15,
      border: '1px solid #ddd',
      borderRadius: 8,
      backgroundColor: '#f8f9fa'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Integrability Testing</h4>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 10 }}>
        Test if different text segmentations yield equivalent quantum states
      </div>
      
      <div style={{ marginBottom: 10 }}>
        <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
          Test Segments (one per line):
        </label>
        <textarea
          value={testSegments.join('\n')}
          onChange={(e) => setTestSegments(e.target.value.split('\n'))}
          placeholder="Enter text segments to test integrability..."
          style={{
            width: '100%',
            height: 80,
            padding: 8,
            border: '1px solid #ddd',
            borderRadius: 4,
            fontSize: 12,
            fontFamily: 'monospace'
          }}
        />
      </div>

      <button 
        onClick={testIntegrability}
        disabled={loading || testSegments.filter(s => s.trim()).length < 2}
        style={{
          padding: '6px 12px',
          border: '1px solid #9C27B0',
          borderRadius: 4,
          backgroundColor: '#9C27B0',
          color: 'white',
          cursor: (loading || testSegments.filter(s => s.trim()).length < 2) ? 'not-allowed' : 'pointer',
          fontSize: 12,
          opacity: (loading || testSegments.filter(s => s.trim()).length < 2) ? 0.6 : 1
        }}
      >
        🧪 Test Integrability
      </button>

      {integrabilityResult && (
        <div style={{ 
          marginTop: 10,
          padding: 10,
          backgroundColor: integrabilityResult.passes_test ? '#f8fff8' : '#fff8f8',
          borderRadius: 4,
          border: `1px solid ${integrabilityResult.passes_test ? '#4CAF50' : '#f44336'}`
        }}>
          <div style={{ fontSize: 12, fontWeight: 'bold', marginBottom: 5 }}>
            Integrability Test {integrabilityResult.passes_test ? 'PASSED' : 'FAILED'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 8 }}>
            <div style={{ fontSize: 11 }}>
              <strong>Segments:</strong> {integrabilityResult.segments_tested}
            </div>
            <div style={{ fontSize: 11 }}>
              <strong>Bures Distance:</strong><br/>
              {integrabilityResult.bures_distance.toExponential(3)}
            </div>
            {integrabilityResult.trace_distance !== undefined && (
              <div style={{ fontSize: 11 }}>
                <strong>Trace Distance:</strong><br/>
                {integrabilityResult.trace_distance.toExponential(3)}
              </div>
            )}
            {integrabilityResult.fidelity !== undefined && (
              <div style={{ fontSize: 11 }}>
                <strong>Fidelity:</strong><br/>
                {integrabilityResult.fidelity.toFixed(4)}
              </div>
            )}
            {integrabilityResult.tolerance !== undefined && (
              <div style={{ fontSize: 11 }}>
                <strong>Tolerance:</strong><br/>
                {integrabilityResult.tolerance.toExponential(1)}
              </div>
            )}
          </div>
          
          {integrabilityResult.quantum_metrics && (
            <div style={{ 
              marginBottom: 8,
              padding: 6,
              backgroundColor: '#f0f0f0',
              borderRadius: 3,
              fontSize: 10
            }}>
              <strong>Quantum Metrics:</strong>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: 4, marginTop: 4 }}>
                <div>Trace A: {integrabilityResult.quantum_metrics.final_state_a_trace?.toFixed(4)}</div>
                <div>Trace B: {integrabilityResult.quantum_metrics.final_state_b_trace?.toFixed(4)}</div>
                <div>Purity A: {integrabilityResult.quantum_metrics.final_state_a_purity?.toFixed(4)}</div>
                <div>Purity B: {integrabilityResult.quantum_metrics.final_state_b_purity?.toFixed(4)}</div>
              </div>
            </div>
          )}
          
          <div style={{ fontSize: 11, fontWeight: 'bold', marginBottom: 3 }}>Recommendations:</div>
          {integrabilityResult.recommendations.map((rec, i) => (
            <div key={i} style={{ 
              fontSize: 10, 
              color: rec.startsWith('✓') ? '#4CAF50' : rec.startsWith('⚠️') || rec.startsWith('🚨') ? '#f44336' : '#666',
              marginLeft: 8,
              marginBottom: 1
            }}>
              {rec}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // Render audit history
  const renderAuditHistory = () => {
    if (auditHistory.length === 0) return null;

    return (
      <div style={{ 
        marginBottom: 15,
        padding: 15,
        border: '1px solid #ddd',
        borderRadius: 8,
        backgroundColor: '#f8f9fa'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Recent Audit History</h4>
        <div style={{ maxHeight: 200, overflowY: 'auto' }}>
          {auditHistory.map((audit, i) => (
            <div key={i} style={{ 
              padding: 8,
              marginBottom: 8,
              backgroundColor: 'white',
              borderRadius: 4,
              border: '1px solid #eee',
              fontSize: 11
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontWeight: 'bold' }}>
                  {new Date(audit.timestamp).toLocaleTimeString()}
                </span>
                <span style={{ 
                  color: audit.passes_sanity_check ? '#4CAF50' : '#f44336',
                  fontWeight: 'bold'
                }}>
                  {audit.passes_sanity_check ? 'PASS' : 'FAIL'}
                </span>
              </div>
              <div style={{ color: '#666' }}>
                Trace: {audit.trace_preservation_error.toExponential(2)} | 
                PSD: {audit.psd_violation.toExponential(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (!rhoId) {
    return (
      <div style={{ 
        padding: 20,
        textAlign: 'center',
        color: '#666',
        border: '1px dashed #ddd',
        borderRadius: 8
      }}>
        <div style={{ fontSize: 16, marginBottom: 8 }}>🔭 Channel Observatory</div>
        <div style={{ fontSize: 12 }}>Select a quantum state to begin channel monitoring</div>
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
          🔭 Channel Observatory
        </h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            padding: '6px 12px',
            border: '1px solid #666',
            borderRadius: 4,
            backgroundColor: 'white',
            cursor: 'pointer',
            fontSize: 12
          }}
        >
          {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
        </button>
      </div>

      <div style={{ fontSize: 12, color: '#666', marginBottom: 15, fontStyle: 'italic' }}>
        Monitoring quantum state: <strong>{rhoId}</strong>
      </div>

      {renderChannelTypeSelector()}
      {renderHealthDashboard()}
      {renderAuditResults()}
      
      {showAdvanced && (
        <>
          {renderIntegrabilityTesting()}
          {renderAuditHistory()}
        </>
      )}
    </div>
  );
}

export default ChannelObservatory;