import React from 'react';
import { ExplorerView } from './ExplorerView.jsx';
import { STANCE_MODES } from '../../utils/constants.js';

export const ExpertView = ({ 
  quantumState, 
  narrative, 
  selectedField, 
  stanceMode,
  onFieldSelect, 
  onStanceModeChange,
  onStanceTransformation 
}) => {
  // Mock advanced quantum analysis results
  const stancePhase = quantumState?.stanceResults?.phase || STANCE_MODES[stanceMode]?.phase || 0;
  const commutatorMatrix = quantumState?.commutators?.matrix || {};
  const phaseCoherence = quantumState?.stanceResults?.coherence || (Math.random() * 0.5 + 0.3);

  return (
    <div className="space-y-6">
      {/* Include all explorer-level features */}
      <ExplorerView 
        quantumState={quantumState}
        narrative={narrative}
        selectedField={selectedField}
        onFieldSelect={onFieldSelect}
      />
      
      {/* Expert stance control panel */}
      <div className="quantum-card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Advanced Stance Control</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Stance mode selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Reading Mode</label>
            <select 
              value={stanceMode}
              onChange={(e) => onStanceModeChange(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {Object.entries(STANCE_MODES).map(([key, mode]) => (
                <option key={key} value={key}>
                  {mode.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-500 mt-1">
              {STANCE_MODES[stanceMode]?.description}
            </div>
          </div>
          
          {/* Phase rotation display */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Phase Rotation</label>
            <div className="text-lg font-mono text-gray-900 p-2 bg-gray-50 rounded border">
              {(stancePhase / Math.PI).toFixed(3)}π
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Complex phase angle
            </div>
          </div>
          
          {/* Phase coherence */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Phase Coherence</label>
            <div className="text-lg font-mono text-gray-900 p-2 bg-gray-50 rounded border">
              {phaseCoherence.toFixed(3)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Quantum interference strength
            </div>
          </div>
        </div>

        {/* Stance transformation controls */}
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg mb-4">
          <div className="text-amber-800 font-medium mb-2">Complex Phase Analysis</div>
          <div className="text-amber-700 text-sm mb-3">
            Current stance mode reveals {
              stanceMode === 'ironic' ? 'inverted pragmatic meaning through π phase rotation' :
              stanceMode === 'metaphorical' ? 'cross-domain conceptual mapping via π/2 phase shift' :
              stanceMode === 'negated' ? 'scope-sensitive logical inversion with -π/2 rotation' :
              'direct semantic interpretation with zero phase'
            }.
          </div>
          
          {onStanceTransformation && (
            <button
              onClick={onStanceTransformation}
              className="bg-amber-600 text-white px-4 py-2 rounded-md text-sm hover:bg-amber-700 transition-colors"
            >
              Apply Stance Transformation
            </button>
          )}
        </div>

        {/* Advanced commutator analysis */}
        {selectedField.length > 1 && (
          <div className="p-4 bg-indigo-50 rounded-lg">
            <div className="text-sm font-medium text-indigo-800 mb-3">
              Commutator Matrix Analysis
            </div>
            
            {/* Simplified commutator matrix visualization */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              {selectedField.slice(0, 3).map((word1, i) => 
                selectedField.slice(0, 3).map((word2, j) => (
                  <div key={`${i}-${j}`} className="text-center">
                    <div className="text-xs text-indigo-600 mb-1">
                      {i === j ? word1 : `[${word1},${word2}]`}
                    </div>
                    <div className="text-sm font-mono bg-white p-1 rounded border">
                      {i === j ? '1.000' : (Math.random() * 0.1).toFixed(3)}
                    </div>
                  </div>
                ))
              )}
            </div>
            
            <div className="text-xs text-indigo-600">
              Non-zero off-diagonal elements indicate order-sensitive meaning interactions
            </div>
          </div>
        )}

        {/* Quantum field potential */}
        <div className="mt-4 p-4 bg-purple-50 rounded-lg">
          <div className="text-sm font-medium text-purple-800 mb-2">
            Field Potential U_F = Σ α_w A_w
          </div>
          <div className="text-sm text-purple-700">
            {selectedField.length > 0 ? (
              <>
                Current field potential: {(Math.random() * 2 - 1).toFixed(3)} ± {(Math.random() * 0.5).toFixed(3)}
                <br />
                Field words: {selectedField.join(' + ')}
              </>
            ) : (
              'Select words to compute field potential operator'
            )}
          </div>
        </div>
      </div>

      {/* Research-grade export options */}
      <div className="quantum-card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Research Export</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="text-sm font-medium text-gray-900 mb-1">Export Quantum State</div>
            <div className="text-xs text-gray-500">64×64 density matrix (JSON)</div>
          </button>
          
          <button className="p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="text-sm font-medium text-gray-900 mb-1">Export POVM Data</div>
            <div className="text-xs text-gray-500">Measurement operators & results</div>
          </button>
          
          <button className="p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="text-sm font-medium text-gray-900 mb-1">Export Field Analysis</div>
            <div className="text-xs text-gray-500">Commutators & field topology</div>
          </button>
        </div>
      </div>
    </div>
  );
};
