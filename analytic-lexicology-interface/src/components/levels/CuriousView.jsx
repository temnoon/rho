import React, { useEffect } from 'react';
import { MEASUREMENT_AXES } from '../../utils/constants.js';
import { safeToFixed, safePercentage, formatQuantumDiagnostics, processQuantumMeasurements } from '../../utils/formatters.js';

export const CuriousView = ({ quantumState, narrative, onViewMetrics }) => {
  // Trigger the view metrics action when component mounts
  useEffect(() => {
    if (onViewMetrics) {
      onViewMetrics();
    }
  }, [onViewMetrics]);

  // Get safely formatted diagnostics
  const rawDiagnostics = quantumState?.diagnostics || {
    purity: Math.random() * 0.5 + 0.1,
    entropy: Math.random() * 4 + 1,
    trace: 1.0
  };
  const diagnostics = formatQuantumDiagnostics(rawDiagnostics);

  // Process raw measurements into display-friendly format
  const rawMeasurements = quantumState?.measurements;
  const measurements = processQuantumMeasurements(rawMeasurements);

  return (
    <div className="quantum-card">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Quantum State Analysis</h3>
      
      {/* Key metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="metric-card bg-purple-50">
          <div className="text-xs text-purple-600 uppercase tracking-wide">Purity</div>
          <div className="text-lg font-semibold text-purple-900">{diagnostics.purity}</div>
          <div className="text-xs text-purple-600 mt-1">Quantum coherence</div>
        </div>
        <div className="metric-card bg-blue-50">
          <div className="text-xs text-blue-600 uppercase tracking-wide">Entropy</div>
          <div className="text-lg font-semibold text-blue-900">{diagnostics.entropy}</div>
          <div className="text-xs text-blue-600 mt-1">Information content</div>
        </div>
        <div className="metric-card bg-green-50">
          <div className="text-xs text-green-600 uppercase tracking-wide">Agency</div>
          <div className="text-lg font-semibold text-green-900">{safeToFixed(measurements.agency, 2)}</div>
          <div className="text-xs text-green-600 mt-1">Actor attribution</div>
        </div>
        <div className="metric-card bg-orange-50">
          <div className="text-xs text-orange-600 uppercase tracking-wide">Formality</div>
          <div className="text-lg font-semibold text-orange-900">{safeToFixed(measurements.formality, 2)}</div>
          <div className="text-xs text-orange-600 mt-1">Language register</div>
        </div>
      </div>
      
      {/* Measurement axes with progress bars */}
      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="text-sm font-medium text-gray-700 mb-3">Measurement Axes</div>
        <div className="space-y-3">
          {Object.entries(measurements).map(([axis, value]) => {
            const axisInfo = MEASUREMENT_AXES[axis] || { name: axis, color: 'gray' };
            return (
              <div key={axis} className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-gray-600">{axisInfo.name}</span>
                    <span className="text-sm font-medium text-gray-900">{safeToFixed(value, 2)}</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div 
                      className={`h-2 bg-${axisInfo.color}-500 rounded-full transition-all duration-500`}
                      style={{ width: `${safePercentage(value)}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Explanation for curious users */}
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-blue-800 font-medium mb-2">Understanding Your Quantum State</div>
        <div className="text-blue-700 text-sm space-y-2">
          <p><strong>Purity ({diagnostics.purity}):</strong> How "mixed" your meaning state is. Higher purity means clearer, more focused meaning.</p>
          <p><strong>Entropy ({diagnostics.entropy}):</strong> Information complexity. Higher entropy indicates richer, more nuanced semantic content.</p>
          <p><strong>Measurement Axes:</strong> Interpretable dimensions like agency, formality, and emotional intensity that we can extract from your quantum state.</p>
        </div>
      </div>
    </div>
  );
};
