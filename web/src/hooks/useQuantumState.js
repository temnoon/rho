/**
 * useQuantumState - Custom hook for quantum density matrix operations.
 * 
 * This hook provides a clean interface for managing quantum state operations,
 * including matrix creation, narrative reading, measurements, and advanced operations.
 */

import { useState, useCallback } from 'react';
import { safeFetch } from '../utils/api.js';

export function useQuantumState() {
  const [currentRho, setCurrentRho] = useState({
    rho_id: null,
    matrix: null,
    diagnostics: null,
    label: null
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const createMatrix = useCallback(async (seedText = null, label = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await safeFetch('/rho/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_text: seedText, label })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Fetch the complete matrix state
        const stateResponse = await safeFetch(`/rho/${result.rho_id}`);
        if (stateResponse.ok) {
          const stateData = await stateResponse.json();
          setCurrentRho({
            rho_id: result.rho_id,
            matrix: stateData.matrix,
            diagnostics: stateData.diagnostics,
            label: stateData.label
          });
          return result.rho_id;
        }
      }
      
      throw new Error('Failed to create matrix');
    } catch (err) {
      setError(err.message);
      console.error('Matrix creation failed:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const readNarrative = useCallback(async (rho_id, text, alpha = 0.2) => {
    if (!rho_id || !text.trim()) return false;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await safeFetch(`/rho/${rho_id}/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: text, alpha })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update current state
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return true;
      }
      
      throw new Error('Failed to read narrative');
    } catch (err) {
      setError(err.message);
      console.error('Narrative reading failed:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const measureMatrix = useCallback(async (rho_id, pack_id = 'advanced_narrative_pack') => {
    if (!rho_id) return null;
    
    try {
      const response = await safeFetch(`/packs/measure/${rho_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pack_id })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update diagnostics
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return result;
      }
      
      throw new Error('Failed to measure matrix');
    } catch (err) {
      setError(err.message);
      console.error('Matrix measurement failed:', err);
      return null;
    }
  }, []);

  const steerMatrix = useCallback(async (rho_id, targetAttributes, pack_id = 'advanced_narrative_pack') => {
    if (!rho_id || !targetAttributes) return false;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await safeFetch(`/advanced/steer/${rho_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_attributes: targetAttributes,
          attribute_pack_id: pack_id,
          max_iterations: 20,
          step_size: 0.1
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update current state
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return result;
      }
      
      throw new Error('Failed to steer matrix');
    } catch (err) {
      setError(err.message);
      console.error('Matrix steering failed:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const applyChannel = useCallback(async (rho_id, channelType, params = {}) => {
    if (!rho_id || !channelType) return false;
    
    setLoading(true);
    setError(null);
    
    try {
      let endpoint = `/advanced/channel/${channelType}/${rho_id}`;
      let body = params;
      
      const response = await safeFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update current state
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return result;
      }
      
      throw new Error(`Failed to apply ${channelType} channel`);
    } catch (err) {
      setError(err.message);
      console.error('Channel application failed:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const projectMaxEntropy = useCallback(async (rho_id, constraints, pack_id = 'advanced_narrative_pack') => {
    if (!rho_id || !constraints) return false;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await safeFetch(`/advanced/project_maxent/${rho_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          constraints,
          attribute_pack_id: pack_id,
          max_iterations: 50
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update current state
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return result;
      }
      
      throw new Error('Failed to apply max-entropy projection');
    } catch (err) {
      setError(err.message);
      console.error('Max-entropy projection failed:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const resetMatrix = useCallback(async (rho_id, seedText = null) => {
    if (!rho_id) return false;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await safeFetch(`/rho/${rho_id}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_text: seedText })
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update current state
        setCurrentRho(prev => ({
          ...prev,
          diagnostics: result.diagnostics
        }));
        
        return true;
      }
      
      throw new Error('Failed to reset matrix');
    } catch (err) {
      setError(err.message);
      console.error('Matrix reset failed:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshState = useCallback(async (rho_id) => {
    if (!rho_id) return;
    
    try {
      const response = await safeFetch(`/rho/${rho_id}`);
      if (response.ok) {
        const stateData = await response.json();
        setCurrentRho({
          rho_id: rho_id,
          matrix: stateData.matrix,
          diagnostics: stateData.diagnostics,
          label: stateData.label
        });
      }
    } catch (err) {
      console.error('Failed to refresh state:', err);
    }
  }, []);

  return {
    // State
    currentRho,
    loading,
    error,
    
    // Basic operations
    createMatrix,
    readNarrative,
    measureMatrix,
    resetMatrix,
    refreshState,
    
    // Advanced operations
    steerMatrix,
    applyChannel,
    projectMaxEntropy,
    
    // Utilities
    clearError: () => setError(null)
  };
}