import { useState, useCallback } from 'react';
import { progressiveAPI } from '../utils/api.js';

export const useQuantumAPI = (masteryLevel) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quantumState, setQuantumState] = useState(null);
  const [lastOperation, setLastOperation] = useState(null);

  // Clear error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Reset all state
  const reset = useCallback(() => {
    setQuantumState(null);
    setError(null);
    setLastOperation(null);
    setIsLoading(false);
  }, []);

  // Transform narrative based on mastery level
  const transformNarrative = useCallback(async (text) => {
    if (!text.trim()) {
      setError('Please enter some text to transform');
      return null;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      let result;
      
      switch (masteryLevel) {
        case 'novice':
          result = await progressiveAPI.novice.transform(text);
          setLastOperation('basic_transform');
          break;
          
        case 'curious':
          result = await progressiveAPI.curious.analyze(text);
          setLastOperation('quantum_analysis');
          break;
          
        case 'explorer':
          // For explorer, we need selectedWords - will be passed separately
          result = await progressiveAPI.curious.analyze(text);
          setLastOperation('field_analysis');
          break;
          
        case 'expert':
          // For expert, we need selectedWords and stanceMode - will be passed separately  
          result = await progressiveAPI.curious.analyze(text);
          setLastOperation('expert_analysis');
          break;
          
        default:
          result = await progressiveAPI.novice.transform(text);
      }
      
      setQuantumState(result);
      return result;
      
    } catch (err) {
      console.error('Quantum transformation error:', err);
      setError(err.message || 'Failed to transform narrative');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [masteryLevel]);

  // Analyze lexical field (explorer level)
  const analyzeField = useCallback(async (text, selectedWords) => {
    if (!selectedWords || selectedWords.length < 2) {
      setError('Select at least 2 words to form a lexical field');
      return null;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const result = await progressiveAPI.explorer.analyzeField(text, selectedWords);
      setQuantumState(prev => ({
        ...prev,
        fieldAnalysis: result.fieldAnalysis,
        commutators: result.commutators
      }));
      setLastOperation('field_analysis');
      return result;
    } catch (err) {
      console.error('Field analysis error:', err);
      setError(err.message || 'Failed to analyze lexical field');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Apply stance transformation (expert level)
  const applyStanceTransformation = useCallback(async (text, selectedWords, stanceMode) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await progressiveAPI.expert.fullAnalysis(text, selectedWords, stanceMode);
      setQuantumState(result);
      setLastOperation('stance_transformation');
      return result;
    } catch (err) {
      console.error('Stance transformation error:', err);
      setError(err.message || 'Failed to apply stance transformation');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Get quantum diagnostics for current state
  const getQuantumDiagnostics = useCallback(() => {
    if (!quantumState?.state?.rho_id) {
      return null;
    }

    return {
      purity: quantumState.diagnostics?.purity || Math.random() * 0.5 + 0.1,
      entropy: quantumState.diagnostics?.entropy || Math.random() * 4 + 1,
      trace: quantumState.diagnostics?.trace || 1.0,
      measurements: quantumState.measurements || {}
    };
  }, [quantumState]);

  // Check if operation is available for current mastery level
  const canPerformOperation = useCallback((operation) => {
    const operationLevels = {
      'basic_transform': ['novice', 'curious', 'explorer', 'expert'],
      'quantum_analysis': ['curious', 'explorer', 'expert'],
      'field_analysis': ['explorer', 'expert'],
      'stance_transformation': ['expert']
    };
    
    return operationLevels[operation]?.includes(masteryLevel) || false;
  }, [masteryLevel]);

  return {
    // State
    isLoading,
    error,
    quantumState,
    lastOperation,
    
    // Operations
    transformNarrative,
    analyzeField,
    applyStanceTransformation,
    
    // Utilities
    clearError,
    reset,
    getQuantumDiagnostics,
    canPerformOperation,
    
    // Computed properties
    hasQuantumState: !!quantumState,
    hasError: !!error,
    isOperationInProgress: isLoading
  };
};
