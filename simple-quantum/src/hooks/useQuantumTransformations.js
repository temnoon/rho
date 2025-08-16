import { useState, useCallback } from 'react';

export const useQuantumTransformations = () => {
  const [isTransforming, setIsTransforming] = useState(false);
  const [transformedText, setTransformedText] = useState('');
  const [auditTrail, setAuditTrail] = useState(null);
  const [quantumDistance, setQuantumDistance] = useState(0);
  const [currentRhoState, setCurrentRhoState] = useState(null);
  const [transformHistory, setTransformHistory] = useState([]);

  const handleTransform = useCallback(async (inputText, prompt, advancedParams) => {
    if (!inputText.trim()) return;
    
    setIsTransforming(true);
    
    try {
      const response = await fetch('http://localhost:8192/transformations/demo-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          transformation_name: 'agent_request',
          strength: advancedParams.strength,
          creativity_level: advancedParams.creativity,
          preservation_level: advancedParams.preservation,
          complexity_target: advancedParams.complexity,
          tone_target: 'creative',
          agent_request: advancedParams.language ? `${prompt}. ${advancedParams.language}` : prompt
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      setTransformedText(result.transformed_text);
      setQuantumDistance(result.quantum_distance);
      setAuditTrail(result.audit_trail);
      
      // Add to transformation history
      const historyEntry = {
        id: Date.now(),
        timestamp: new Date(),
        prompt: prompt,
        originalText: inputText.substring(0, 100) + (inputText.length > 100 ? '...' : ''),
        transformedText: result.transformed_text.substring(0, 100) + (result.transformed_text.length > 100 ? '...' : ''),
        quantumDistance: result.quantum_distance,
        duration: result.audit_trail?.performance?.total_duration || 0,
        model: result.audit_trail?.llm_interaction?.model || 'Unknown',
        success: true
      };
      setTransformHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10
      
      return result;
    } catch (error) {
      console.error('Transformation failed:', error);
      setTransformedText(`❌ Transformation failed: ${error.message}`);
      throw error;
    } finally {
      setIsTransforming(false);
    }
  }, []);

  const handleCompassTransformation = useCallback(async (inputText, compassConfig, advancedParams) => {
    if (!inputText.trim()) return;
    
    setIsTransforming(true);
    
    try {
      // Create a POVM-specific transformation request
      const povmRequest = {
        text: inputText,
        transformation_name: 'povm_compass_navigation',
        povm_operator: compassConfig.operator,
        direction: compassConfig.direction,
        magnitude: compassConfig.magnitude,
        pass_history: compassConfig.passHistory,
        strength: advancedParams.strength,
        creativity_level: advancedParams.creativity,
        preservation_level: advancedParams.preservation,
        tone_target: 'analytic_lexography'
      };

      const response = await fetch('http://localhost:8192/transformations/demo-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(povmRequest)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setTransformedText(result.transformed_text);
      setAuditTrail(result.audit_trail);
      setQuantumDistance(result.quantum_distance);
      setCurrentRhoState(result.rho_state);

      // Add to transformation history with POVM details
      const historyEntry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        prompt: `${compassConfig.operator} → ${compassConfig.direction} (${compassConfig.magnitude})`,
        originalText: inputText.substring(0, 100) + (inputText.length > 100 ? '...' : ''),
        transformedText: result.transformed_text.substring(0, 100) + (result.transformed_text.length > 100 ? '...' : ''),
        quantumDistance: result.quantum_distance,
        duration: result.audit_trail?.performance?.total_duration || 0,
        model: result.audit_trail?.llm_interaction?.model || 'Unknown',
        success: true,
        transformationType: 'compass_povm',
        povmDetails: compassConfig
      };
      setTransformHistory(prev => [historyEntry, ...prev.slice(0, 9)]);
      
      return result;
    } catch (error) {
      console.error('Compass transformation failed:', error);
      setTransformedText(`❌ Compass transformation failed: ${error.message}`);
      throw error;
    } finally {
      setIsTransforming(false);
    }
  }, []);

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      // Simple visual feedback
      return true;
    } catch (err) {
      console.error('Failed to copy text:', err);
      return false;
    }
  };

  return {
    // State
    isTransforming,
    transformedText,
    auditTrail,
    quantumDistance,
    currentRhoState,
    transformHistory,
    
    // Actions
    handleTransform,
    handleCompassTransformation,
    copyToClipboard,
    
    // Setters for external control
    setTransformedText,
    setAuditTrail,
    setQuantumDistance,
    setCurrentRhoState,
    setTransformHistory
  };
};