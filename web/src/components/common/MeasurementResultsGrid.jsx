import React from 'react';

/**
 * MeasurementResultsGrid - Reusable component for displaying POVM measurement results
 * 
 * Extracted from NarrativeDistillationStudio and NarrativeExplorer patterns
 */
export function MeasurementResultsGrid({ 
  measurements = {},
  title = "Measurement Results",
  maxDisplayed = 12,
  showBars = true,
  sortBy = 'value', // value, name
  layout = 'grid', // grid, list
  onAttributeClick = null,
  style = {}
}) {
  if (!measurements || Object.keys(measurements).length === 0) {
    return (
      <div style={{
        background: '#f8f9fa',
        border: '1px solid #dee2e6',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center',
        color: '#6c757d',
        ...style
      }}>
        No measurements available
      </div>
    );
  }

  // Process and sort measurements
  const processedMeasurements = Object.entries(measurements)
    .map(([attribute, value]) => ({
      attribute,
      value: typeof value === 'number' ? value : 0,
      displayName: attribute.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      percentage: typeof value === 'number' ? (value * 100).toFixed(1) : '0.0'
    }))
    .sort((a, b) => {
      if (sortBy === 'value') {
        return b.value - a.value;
      } else {
        return a.displayName.localeCompare(b.displayName);
      }
    })
    .slice(0, maxDisplayed);

  const renderMeasurementBar = (value) => {
    if (!showBars) return null;
    
    const barLength = Math.round(value * 20);
    const bar = '█'.repeat(barLength) + '░'.repeat(20 - barLength);
    
    return (
      <div style={{
        fontFamily: 'monospace',
        fontSize: '10px',
        color: '#666',
        marginTop: '2px',
        letterSpacing: '1px'
      }}>
        {bar}
      </div>
    );
  };

  const renderGridLayout = () => (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '15px'
    }}>
      {processedMeasurements.map(({ attribute, value, displayName, percentage }) => (
        <div 
          key={attribute}
          onClick={onAttributeClick ? () => onAttributeClick(attribute, value) : undefined}
          style={{
            background: 'white',
            padding: '12px',
            borderRadius: '6px',
            border: '1px solid #e0e0e0',
            cursor: onAttributeClick ? 'pointer' : 'default',
            transition: 'all 0.2s ease',
            ':hover': onAttributeClick ? {
              borderColor: '#9C27B0',
              boxShadow: '0 2px 8px rgba(156,39,176,0.1)'
            } : {}
          }}
        >
          <div style={{ 
            fontWeight: 600, 
            marginBottom: '5px', 
            fontSize: '13px',
            color: '#333'
          }}>
            {displayName}
          </div>
          <div style={{ 
            fontSize: '18px', 
            color: '#9C27B0', 
            fontWeight: 600,
            marginBottom: '2px'
          }}>
            {percentage}%
          </div>
          {renderMeasurementBar(value)}
        </div>
      ))}
    </div>
  );

  const renderListLayout = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {processedMeasurements.map(({ attribute, value, displayName, percentage }) => (
        <div 
          key={attribute}
          onClick={onAttributeClick ? () => onAttributeClick(attribute, value) : undefined}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '10px 15px',
            background: 'white',
            border: '1px solid #e0e0e0',
            borderRadius: '6px',
            cursor: onAttributeClick ? 'pointer' : 'default',
            transition: 'all 0.2s ease'
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, fontSize: '14px', color: '#333' }}>
              {displayName}
            </div>
            {showBars && renderMeasurementBar(value)}
          </div>
          <div style={{ 
            fontSize: '16px', 
            color: '#9C27B0', 
            fontWeight: 600,
            minWidth: '60px',
            textAlign: 'right'
          }}>
            {percentage}%
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{
      background: '#f3e5f5',
      border: '1px solid #9C27B0',
      borderRadius: '8px',
      padding: '20px',
      ...style
    }}>
      <h3 style={{ 
        marginBottom: '15px', 
        color: '#7B1FA2',
        fontSize: '16px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        ⚡ {title}
        <span style={{ 
          fontSize: '12px', 
          color: '#666', 
          fontWeight: 'normal' 
        }}>
          ({processedMeasurements.length} attributes)
        </span>
      </h3>
      
      {layout === 'grid' ? renderGridLayout() : renderListLayout()}
      
      {Object.keys(measurements).length > maxDisplayed && (
        <div style={{
          marginTop: '15px',
          padding: '10px',
          background: 'rgba(156,39,176,0.1)',
          borderRadius: '4px',
          fontSize: '12px',
          color: '#7B1FA2',
          textAlign: 'center'
        }}>
          Showing top {maxDisplayed} of {Object.keys(measurements).length} measurements
        </div>
      )}
    </div>
  );
}

export default MeasurementResultsGrid;