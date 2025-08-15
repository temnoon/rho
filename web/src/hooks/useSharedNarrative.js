/**
 * useSharedNarrative - Hook for managing shared narrative state across components.
 * 
 * This provides a simple way to share narrative text between different parts
 * of the application while maintaining reactivity.
 */

import { useState, useCallback } from 'react';

// Global narrative state (simple implementation)
let globalNarrative = "In the quantum realm of narrative possibilities, each word carries the weight of infinite interpretations...";
let listeners = new Set();

export function useSharedNarrative() {
  const [narrative, setNarrative] = useState(globalNarrative);

  // Register this component as a listener
  useState(() => {
    const updateLocal = (newNarrative) => {
      setNarrative(newNarrative);
    };
    
    listeners.add(updateLocal);
    
    // Cleanup on unmount
    return () => {
      listeners.delete(updateLocal);
    };
  });

  const updateSharedNarrative = useCallback((newNarrative) => {
    globalNarrative = newNarrative;
    
    // Notify all listeners
    listeners.forEach(listener => {
      listener(newNarrative);
    });
  }, []);

  return {
    sharedNarrative: narrative,
    updateSharedNarrative
  };
}