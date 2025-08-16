import React from 'react';
import { CuriousView } from './CuriousView.jsx';
import { WordCloud } from '../WordCloud.jsx';

export const ExplorerView = ({ 
  quantumState, 
  narrative, 
  selectedField, 
  onFieldSelect, 
  onFieldAnalysis 
}) => {
  const handleWordSelect = (word) => {
    onFieldSelect(prev => 
      prev.includes(word) 
        ? prev.filter(w => w !== word)
        : [...prev, word]
    );
  };

  // Mock field analysis results
  const fieldTension = quantumState?.fieldAnalysis?.tension || (Math.random() * 0.5 + 0.2);
  const commutatorNorm = quantumState?.commutators?.average_norm || (Math.random() * 0.1);

  return (
    <div className="space-y-6">
      {/* Include all curious-level features */}
      <CuriousView quantumState={quantumState} narrative={narrative} />
      
      {/* Field analysis section */}
      <div className="quantum-card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Lexical Field Analysis</h3>
        
        <div className="mb-6">
          <WordCloud
            text={narrative}
            selectedWords={selectedField}
            onWordSelect={handleWordSelect}
            width={600}
            height={400}
            maxWords={35}
          />
        </div>

        {selectedField.length > 1 && (
          <div className="space-y-4">
            {/* Field analysis results */}
            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="text-sm font-medium text-purple-800 mb-2">
                Field Tension: {fieldTension.toFixed(3)}
              </div>
              <div className="text-sm text-purple-700 mb-3">
                Selected field: [{selectedField.join(', ')}] shows {fieldTension > 0.4 ? 'high' : 'moderate'} co-variation in meaning space.
              </div>
              
              {onFieldAnalysis && (
                <button
                  onClick={onFieldAnalysis}
                  className="bg-purple-600 text-white px-4 py-2 rounded-md text-sm hover:bg-purple-700 transition-colors"
                >
                  Analyze Field Relationships
                </button>
              )}
            </div>

            {/* Commutator analysis preview */}
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="text-sm font-medium text-green-800 mb-2">
                Order Sensitivity Analysis
              </div>
              <div className="text-sm text-green-700 mb-2">
                Average commutator norm: {commutatorNorm.toFixed(4)}
              </div>
              <div className="text-xs text-green-600">
                {commutatorNorm > 0.05 ? 
                  'High order sensitivity - word order significantly affects meaning' :
                  'Low order sensitivity - relatively stable meaning regardless of order'
                }
              </div>
            </div>

            {/* Field topology visualization placeholder */}
            <div className="p-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center text-gray-500">
                <div className="text-sm font-medium mb-1">Field Topology Visualization</div>
                <div className="text-xs">Interactive word relationship network would appear here</div>
              </div>
            </div>
          </div>
        )}

        {selectedField.length === 0 && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-blue-800 text-sm">
              <strong>Lexical Fields:</strong> Groups of words whose meanings co-constrain each other. 
              Select words above to see how their meanings interact in quantum space.
            </div>
          </div>
        )}

        {selectedField.length === 1 && (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="text-yellow-800 text-sm">
              Select at least 2 words to form a lexical field and analyze their relationships.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
