import React from 'react';

/**
 * ProgressIndicator - Reusable component for showing progress and status
 * 
 * Extracted from NarrativeDistillationStudio progress patterns
 */
export function ProgressIndicator({ 
  progress = 0, // 0-100
  status = 'idle', // idle, loading, success, error
  message = '',
  showPercentage = true,
  showSpinner = true,
  height = 20,
  animated = true,
  style = {}
}) {
  const getStatusColor = () => {
    switch (status) {
      case 'loading': return '#FF9800';
      case 'success': return '#4CAF50';
      case 'error': return '#f44336';
      default: return '#2196F3';
    }
  };

  const getStatusIcon = () => {
    if (!showSpinner) return null;
    
    switch (status) {
      case 'loading': return '⚡';
      case 'success': return '✅';
      case 'error': return '❌';
      default: return null;
    }
  };

  return (
    <div style={{
      width: '100%',
      ...style
    }}>
      {message && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '10px',
          fontSize: '14px',
          color: '#666'
        }}>
          {getStatusIcon()}
          <span>{message}</span>
          {showPercentage && status !== 'idle' && (
            <span style={{ 
              marginLeft: 'auto',
              fontWeight: 600,
              color: getStatusColor()
            }}>
              {Math.round(progress)}%
            </span>
          )}
        </div>
      )}
      
      <div style={{
        width: '100%',
        height: height,
        backgroundColor: '#f0f0f0',
        borderRadius: height / 2,
        overflow: 'hidden',
        position: 'relative'
      }}>
        <div 
          style={{
            width: `${Math.max(0, Math.min(100, progress))}%`,
            height: '100%',
            backgroundColor: getStatusColor(),
            transition: animated ? 'width 0.3s ease' : 'none',
            borderRadius: height / 2,
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* Animated shine effect for loading */}
          {status === 'loading' && animated && (
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
              animation: 'shine 2s infinite',
              borderRadius: height / 2
            }} />
          )}
        </div>
      </div>
      
    </div>
  );
}

/**
 * StepProgressIndicator - Multi-step progress indicator
 */
export function StepProgressIndicator({
  steps = [],
  currentStep = 0,
  completedSteps = [],
  orientation = 'horizontal', // horizontal, vertical
  showLabels = true,
  style = {}
}) {
  const isStepCompleted = (stepIndex) => {
    return completedSteps.includes(stepIndex) || stepIndex < currentStep;
  };

  const isStepActive = (stepIndex) => {
    return stepIndex === currentStep;
  };

  const renderHorizontalSteps = () => (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      width: '100%'
    }}>
      {steps.map((step, index) => (
        <React.Fragment key={index}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            flex: 1
          }}>
            <div style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: isStepCompleted(index) ? '#4CAF50' : 
                         isStepActive(index) ? '#2196F3' : '#e0e0e0',
              color: isStepCompleted(index) || isStepActive(index) ? 'white' : '#666',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '14px',
              fontWeight: 600,
              marginBottom: showLabels ? '8px' : 0
            }}>
              {isStepCompleted(index) ? '✓' : index + 1}
            </div>
            {showLabels && (
              <div style={{
                fontSize: '12px',
                color: isStepActive(index) ? '#2196F3' : '#666',
                textAlign: 'center',
                maxWidth: '80px'
              }}>
                {step.label || step}
              </div>
            )}
          </div>
          
          {index < steps.length - 1 && (
            <div style={{
              flex: 1,
              height: 2,
              background: isStepCompleted(index) ? '#4CAF50' : '#e0e0e0',
              marginBottom: showLabels ? '20px' : 0
            }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );

  const renderVerticalSteps = () => (
    <div style={{
      display: 'flex',
      flexDirection: 'column'
    }}>
      {steps.map((step, index) => (
        <div key={index} style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: index < steps.length - 1 ? '20px' : 0
        }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: '50%',
            background: isStepCompleted(index) ? '#4CAF50' : 
                       isStepActive(index) ? '#2196F3' : '#e0e0e0',
            color: isStepCompleted(index) || isStepActive(index) ? 'white' : '#666',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '14px',
            fontWeight: 600,
            marginRight: '12px'
          }}>
            {isStepCompleted(index) ? '✓' : index + 1}
          </div>
          {showLabels && (
            <div style={{
              fontSize: '14px',
              color: isStepActive(index) ? '#2196F3' : '#666'
            }}>
              {step.label || step}
            </div>
          )}
        </div>
      ))}
    </div>
  );

  return (
    <div style={style}>
      {orientation === 'horizontal' ? renderHorizontalSteps() : renderVerticalSteps()}
    </div>
  );
}

export default ProgressIndicator;