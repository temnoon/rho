import React from 'react';

const AuditTrail = ({
  auditTrail,
  showAuditTrail,
  setShowAuditTrail
}) => {
  if (!auditTrail) return null;

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '30px auto 0 auto',
      background: 'rgba(255,255,255,0.95)',
      borderRadius: '20px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      overflow: 'hidden'
    }}>
      {/* Toggle Button */}
      <div style={{
        padding: '20px 30px',
        borderBottom: '1px solid #e9ecef',
        background: 'white'
      }}>
        <button
          onClick={() => setShowAuditTrail(!showAuditTrail)}
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
          <span>{showAuditTrail ? '🔼' : '🔽'}</span>
          View Quantum Process
        </button>
      </div>

      {/* Audit Trail Content */}
      {showAuditTrail && (
        <div style={{ padding: '30px', borderBottom: '1px solid #e9ecef' }}>
          <h3 style={{ 
            margin: '0 0 20px 0', 
            color: '#495057',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            🔬 Quantum Transformation Process
            <span style={{
              fontSize: '12px',
              background: '#e3f2fd',
              color: '#1976d2',
              padding: '4px 8px',
              borderRadius: '6px'
            }}>
              Live Audit
            </span>
          </h3>
          
          {/* Performance Overview */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
            gap: '15px', 
            marginBottom: '25px' 
          }}>
            <div style={{ 
              padding: '15px', 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              color: 'white',
              borderRadius: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700' }}>
                {(auditTrail.performance?.total_duration * 1000 || 0).toFixed(0)}ms
              </div>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>Total Time</div>
            </div>
            <div style={{ 
              padding: '15px', 
              background: 'linear-gradient(135deg, #28a745 0%, #20c997 100%)', 
              color: 'white',
              borderRadius: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700' }}>
                {(auditTrail.performance?.quantum_operations_time * 1000 || 0).toFixed(0)}ms
              </div>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>Quantum Ops</div>
            </div>
            <div style={{ 
              padding: '15px', 
              background: 'linear-gradient(135deg, #fd7e14 0%, #ffc107 100%)', 
              color: 'white',
              borderRadius: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700' }}>
                {(auditTrail.performance?.llm_time * 1000 || 0).toFixed(0)}ms
              </div>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>LLM Generation</div>
            </div>
            <div style={{ 
              padding: '15px', 
              background: 'linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%)', 
              color: 'white',
              borderRadius: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '16px', fontWeight: '700' }}>
                {auditTrail.llm_interaction?.model?.split('/').pop() || 'Unknown'}
              </div>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>Model Used</div>
            </div>
          </div>

          {/* Quantum Steps Timeline */}
          {auditTrail.quantum_steps?.length > 0 && (
            <div style={{ marginBottom: '25px' }}>
              <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
                ⚛️ Quantum Operations Timeline
              </h4>
              <div style={{ background: '#f8f9fa', borderRadius: '12px', padding: '20px' }}>
                {auditTrail.quantum_steps.map((step, idx) => (
                  <div key={idx} style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '12px 0',
                    borderBottom: idx < auditTrail.quantum_steps.length - 1 ? '1px solid #e9ecef' : 'none'
                  }}>
                    <div style={{
                      width: '30px',
                      height: '30px',
                      borderRadius: '50%',
                      background: step.success !== false ? '#28a745' : '#dc3545',
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '12px',
                      fontWeight: '600',
                      marginRight: '15px'
                    }}>
                      {step.step}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: '600', color: '#495057' }}>
                        {step.name?.replace(/_/g, ' ') || 'Quantum Operation'}
                      </div>
                      <div style={{ fontSize: '14px', color: '#6c757d', marginTop: '2px' }}>
                        {step.description}
                      </div>
                    </div>
                    <div style={{
                      padding: '4px 8px',
                      background: 'white',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#495057',
                      border: '1px solid #e9ecef'
                    }}>
                      {(step.duration * 1000).toFixed(0)}ms
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* LLM Interaction Details */}
          {auditTrail.llm_interaction && (
            <div>
              <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
                🤖 LLM Interaction Details
              </h4>
              <div style={{ 
                background: '#f8f9fa', 
                borderRadius: '12px', 
                padding: '20px',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '15px'
              }}>
                <div>
                  <strong>Provider:</strong><br />
                  <span style={{ color: '#6c757d' }}>
                    {auditTrail.llm_interaction.provider || 'Unknown'}
                  </span>
                </div>
                <div>
                  <strong>Response Length:</strong><br />
                  <span style={{ color: '#6c757d' }}>
                    {auditTrail.llm_interaction.response_length || 0} chars
                  </span>
                </div>
                <div>
                  <strong>Cleaning Applied:</strong><br />
                  <span style={{ color: auditTrail.llm_interaction.cleaning_applied ? '#dc3545' : '#28a745' }}>
                    {auditTrail.llm_interaction.cleaning_applied ? 'Yes' : 'No'}
                  </span>
                </div>
                <div>
                  <strong>Success Rate:</strong><br />
                  <span style={{ color: '#28a745' }}>
                    {auditTrail.llm_interaction.success ? '100%' : '0%'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AuditTrail;