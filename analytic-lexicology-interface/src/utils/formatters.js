/**
 * Utility functions for safe number formatting in the analytic lexicology interface
 */

/**
 * Safely format a number to a fixed number of decimal places
 * @param {any} value - The value to format (may not be a number)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @param {string} fallback - Fallback string if value is not a number
 * @returns {string} Formatted number string
 */
export const safeToFixed = (value, decimals = 2, fallback = null) => {
  if (typeof value === 'number' && !isNaN(value)) {
    return value.toFixed(decimals);
  }
  
  // If fallback is not provided, generate it based on decimals
  if (fallback === null) {
    fallback = '0.' + '0'.repeat(decimals);
  }
  
  return fallback;
};

/**
 * Safely convert a value to a percentage for progress bars
 * @param {any} value - The value to convert (should be 0-1 range)
 * @param {number} fallback - Fallback percentage (default: 0)
 * @returns {number} Percentage value 0-100
 */
export const safePercentage = (value, fallback = 0) => {
  if (typeof value === 'number' && !isNaN(value)) {
    return Math.max(0, Math.min(100, value * 100));
  }
  return fallback;
};

/**
 * Format quantum diagnostics with appropriate precision
 * @param {any} diagnostics - Diagnostics object from API
 * @returns {object} Safely formatted diagnostics
 */
export const formatQuantumDiagnostics = (diagnostics) => {
  if (!diagnostics || typeof diagnostics !== 'object') {
    return {
      purity: '0.000',
      entropy: '0.00',
      trace: '1.00'
    };
  }
  
  return {
    purity: safeToFixed(diagnostics.purity, 3),
    entropy: safeToFixed(diagnostics.entropy, 2),
    trace: safeToFixed(diagnostics.trace, 2, '1.00')
  };
};

/**
 * Format measurement values with consistent precision
 * @param {object} measurements - Measurements object from API
 * @returns {object} Safely formatted measurements
 */
export const formatMeasurements = (measurements) => {
  if (!measurements || typeof measurements !== 'object') {
    return {};
  }
  
  const formatted = {};
  for (const [key, value] of Object.entries(measurements)) {
    formatted[key] = {
      value: typeof value === 'number' ? value : 0,
      display: safeToFixed(value, 2),
      percentage: safePercentage(value)
    };
  }
  
  return formatted;
};

/**
 * Process complex API measurements into display-friendly categories
 * @param {object} rawMeasurements - Raw measurements from API
 * @returns {object} Processed measurements for display
 */
export const processQuantumMeasurements = (rawMeasurements) => {
  if (!rawMeasurements || typeof rawMeasurements !== 'object') {
    return {
      agency: 0,
      formality: 0,
      emotional_intensity: 0,
      narrative_distance: 0,
      temporal_perspective: 0,
      certainty: 0
    };
  }

  // Aggregate related measurements into display categories
  const processed = {
    // Agency: combine involved_production and narrative_concerns
    agency: Math.max(
      rawMeasurements.involved_production_involved || 0,
      rawMeasurements.narrative_concerns_narrative || 0,
      0
    ),
    
    // Formality: use tenor_formality measurements
    formality: Math.max(
      rawMeasurements.tenor_formality_formal || 0,
      rawMeasurements.field_register_specialized || 0,
      0
    ),
    
    // Emotional intensity: use tenor_affect
    emotional_intensity: Math.max(
      rawMeasurements.tenor_affect_affective || 0,
      rawMeasurements.tenor_affect_neutral || 0,
      0
    ),
    
    // Narrative distance: use narrative_distance measurements
    narrative_distance: Math.max(
      rawMeasurements.narrative_distance_close || 0,
      rawMeasurements.narrative_distance_distant || 0,
      rawMeasurements.focalization_type_internal || 0,
      0
    ),
    
    // Temporal perspective: use temporal_perspective measurements  
    temporal_perspective: Math.max(
      rawMeasurements.temporal_perspective_prospective || 0,
      rawMeasurements.temporal_perspective_retrospective || 0,
      0
    ),
    
    // Certainty: use discourse_coherence and elaborated_reference
    certainty: Math.max(
      rawMeasurements.discourse_coherence_coherent || 0,
      rawMeasurements.elaborated_reference_elaborated || 0,
      0
    )
  };

  return processed;
};

/**
 * Generate mock data with proper numeric types for development
 * @returns {object} Mock quantum state data
 */
export const generateMockQuantumData = () => ({
  diagnostics: {
    purity: Math.random() * 0.5 + 0.1,
    entropy: Math.random() * 4 + 1,
    trace: 1.0
  },
  measurements: {
    agency: Math.random(),
    formality: Math.random(),
    emotional_intensity: Math.random(),
    narrative_distance: Math.random(),
    temporal_perspective: Math.random(),
    certainty: Math.random()
  }
});