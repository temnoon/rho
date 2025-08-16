import React from 'react';

const TransformationHistory = ({
  transformHistory
}) => {
  if (transformHistory.length === 0) return null;

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '30px auto 0 auto',
      background: 'rgba(255,255,255,0.95)',
      borderRadius: '20px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
      overflow: 'hidden'
    }}>
      {/* Transformation History */}
      <div style={{ padding: '30px' }}>
        <h3 style={{ 
          margin: '0 0 20px 0', 
          color: '#495057',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          📊 Transformation History
          <span style={{
            fontSize: '12px',
            background: '#e3f2fd',
            color: '#1976d2',
            padding: '4px 8px',
            borderRadius: '6px'
          }}>
            Last {transformHistory.length}
          </span>
        </h3>
        
        <div style={{ 
          display: 'grid', 
          gap: '12px',
          maxHeight: '400px',
          overflowY: 'auto'
        }}>
          {transformHistory.map((entry, idx) => (
            <div key={entry.id} style={{
              background: idx === 0 ? '#e8f5e8' : '#f8f9fa',
              border: `1px solid ${idx === 0 ? '#28a745' : '#e9ecef'}`,
              borderRadius: '12px',
              padding: '15px',
              display: 'grid',
              gridTemplateColumns: '1fr auto auto auto',
              gap: '15px',
              alignItems: 'center'
            }}>
              <div>
                <div style={{ fontWeight: '600', color: '#495057', marginBottom: '5px' }}>
                  "{entry.prompt}"
                </div>
                <div style={{ fontSize: '14px', color: '#6c757d' }}>
                  {entry.originalText} → {entry.transformedText}
                </div>
                {entry.transformationType === 'compass_povm' && entry.povmDetails && (
                  <div style={{ 
                    fontSize: '12px', 
                    color: '#667eea', 
                    marginTop: '4px',
                    fontWeight: '500'
                  }}>
                    POVM: {entry.povmDetails.operator} • {entry.povmDetails.direction} • Mag: {entry.povmDetails.magnitude}
                  </div>
                )}
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Distance</div>
                <div style={{ fontSize: '14px', fontWeight: '600' }}>
                  {entry.quantumDistance.toExponential(1)}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Time</div>
                <div style={{ fontSize: '14px', fontWeight: '600' }}>
                  {(entry.duration * 1000).toFixed(0)}ms
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Model</div>
                <div style={{ fontSize: '12px', fontWeight: '600' }}>
                  {entry.model.split('/').pop()}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Performance Statistics */}
        {transformHistory.length > 1 && (
          <div style={{ 
            marginTop: '25px',
            padding: '20px',
            background: '#f8f9fa',
            borderRadius: '12px',
            border: '1px solid #e9ecef'
          }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>
              📈 Performance Statistics
            </h4>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
              gap: '15px' 
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#667eea' }}>
                  {(transformHistory.reduce((sum, entry) => sum + entry.duration, 0) * 1000).toFixed(0)}ms
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Total Processing Time</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#28a745' }}>
                  {(transformHistory.reduce((sum, entry) => sum + entry.duration, 0) * 1000 / transformHistory.length).toFixed(0)}ms
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Average Time</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#fd7e14' }}>
                  {transformHistory.reduce((sum, entry) => sum + entry.quantumDistance, 0).toExponential(1)}
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Total Distance</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#6f42c1' }}>
                  {(transformHistory.filter(entry => entry.success).length / transformHistory.length * 100).toFixed(0)}%
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d' }}>Success Rate</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TransformationHistory;