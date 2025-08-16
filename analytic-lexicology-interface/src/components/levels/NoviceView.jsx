import React from 'react';

export const NoviceView = ({ quantumState, narrative }) => {
  return (
    <div className="quantum-card">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Transformed</h3>
      <div className="space-y-3">
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="text-green-800 font-medium mb-1">Enhanced (170%)</div>
          <div className="text-green-700">
            {quantumState?.enhanced?.transformed_text || 
             `"The algorithm *profusely* apologized to its user, expressing deep regret, but the *devastating* damage was already irreversibly done."`}
          </div>
        </div>
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="text-blue-800 font-medium mb-1">Subdued (70%)</div>
          <div className="text-blue-700">
            {quantumState?.subdued?.transformed_text ||
             `"The algorithm noted an error to its user, though the issue had occurred."`}
          </div>
        </div>
      </div>
    </div>
  );
};
