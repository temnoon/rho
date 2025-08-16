import React, { createContext, useContext } from 'react';
import { useMasteryLevel } from './hooks/useMasteryLevel.js';
import { useQuantumAPI } from './hooks/useQuantumAPI.js';

// Create context for the Analytic Lexicology application state
const AnalyticLexicologyContext = createContext();

// Provider component that wraps the entire application
export const AnalyticLexicologyProvider = ({ children }) => {
  const masteryHook = useMasteryLevel();
  const quantumHook = useQuantumAPI(masteryHook.masteryLevel);

  const value = {
    // Mastery level management
    ...masteryHook,
    
    // Quantum API operations
    ...quantumHook,
    
    // Combined convenience methods
    isReady: !quantumHook.isLoading && !quantumHook.error,
    canShowAdvancedFeatures: masteryHook.canAccess('curious'),
    canShowFieldAnalysis: masteryHook.canAccess('explorer'),
    canShowStanceControls: masteryHook.canAccess('expert')
  };

  return (
    <AnalyticLexicologyContext.Provider value={value}>
      {children}
    </AnalyticLexicologyContext.Provider>
  );
};

// Hook to use the Analytic Lexicology context
export const useAnalyticLexicology = () => {
  const context = useContext(AnalyticLexicologyContext);
  
  if (!context) {
    throw new Error('useAnalyticLexicology must be used within an AnalyticLexicologyProvider');
  }
  
  return context;
};

// Component to handle context and render main interface
import { ProgressiveInterface } from './components/core/ProgressiveInterface.jsx';

const App = () => {
  return (
    <AnalyticLexicologyProvider>
      <ProgressiveInterface />
    </AnalyticLexicologyProvider>
  );
};

export default App;
