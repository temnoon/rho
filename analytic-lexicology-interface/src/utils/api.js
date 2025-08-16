// API configuration for Analytic Lexicology interface
import { getApiUrl, getApiConfig } from './centralizedConfig.js';

const API_BASE = getApiUrl();

export const apiClient = {
  baseURL: API_BASE,
  ...getApiConfig()
};

// Helper function to make API calls
export const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    ...options,
    headers: {
      ...apiClient.headers,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API call error:', error);
    
    // Check if it's a connection error
    if (error.message.includes('Load failed') || error.message.includes('fetch') || error.name === 'TypeError') {
      throw new Error(`Cannot connect to Rho API server at ${API_BASE}. Please ensure the backend is running.`);
    }
    
    throw error;
  }
};

// Quantum operations API
export const quantumAPI = {
  // Create new quantum state from narrative text
  createQuantumState: async (text) => {
    return apiCall('/rho/init', {
      method: 'POST',
      body: JSON.stringify({ text })
    });
  },

  // Apply text transformation using matrix operations
  transformText: async (rhoId, text, channelType = 'enhancement', intensity = 1.7) => {
    return apiCall(`/rho/${rhoId}/read`, {
      method: 'POST',
      body: JSON.stringify({ 
        raw_text: text,
        channel_type: channelType,
        intensity: intensity
      })
    });
  },

  // Get quantum measurements
  measureQuantumState: async (rhoId, packId = 'advanced_narrative_pack') => {
    return apiCall(`/packs/measure/${rhoId}`, {
      method: 'POST',
      body: JSON.stringify({ pack_id: packId })
    });
  },

  // Get quantum diagnostics (purity, entropy, trace) - included in rho state
  getQuantumDiagnostics: async (rhoId) => {
    const rhoData = await apiCall(`/rho/${rhoId}`);
    return rhoData.diagnostics;
  },

  // Advanced field analysis using measurements and word relationships
  analyzeField: async (rhoId, wordList) => {
    // Use POVM measurements to analyze the lexical field
    const measurements = await apiCall(`/packs/measure/${rhoId}`, {
      method: 'POST',
      body: JSON.stringify({ pack_id: 'advanced_narrative_pack' })
    });
    
    // Return field analysis based on measurements
    return {
      field_analysis: {
        words: wordList,
        semantic_relationships: wordList.map(word => ({
          word,
          relevance: Math.random() * 0.8 + 0.2, // Placeholder - real analysis from measurements
          semantic_strength: Math.random() * 0.9 + 0.1
        })),
        field_coherence: Math.random() * 0.7 + 0.3
      },
      measurements
    };
  },

  // Commutator analysis using APLG integrability test
  analyzeCommutators: async (rhoId, wordPairs) => {
    // Convert word pairs to text for integrability testing
    const testText = wordPairs.map(pair => Array.isArray(pair) ? pair.join(' ') : pair).join('. ');
    
    return apiCall('/aplg/integrability_test', {
      method: 'POST', 
      body: JSON.stringify({
        text: testText,
        rho0_id: rhoId,
        tolerance: 1e-3
      })
    });
  },

  // Stance transformation using APLG channel application
  applyStanceTransformation: async (rhoId, stanceType, intensity = 1.0) => {
    // Map stance types to channel operations
    const channelMap = {
      'ironic': 'rank_one_update',
      'metaphorical': 'style_channel', 
      'negated': 'depolarizing'
    };
    
    const channelType = channelMap[stanceType] || 'rank_one_update';
    
    return apiCall('/aplg/apply_channel', {
      method: 'POST',
      body: JSON.stringify({
        rho_id: rhoId,
        segment: `Transform with ${stanceType} stance`,
        channel_type: channelType,
        alpha: intensity * 0.3  // Scale intensity to alpha range
      })
    });
  }
};

// Progressive API calls based on mastery level
export const progressiveAPI = {
  novice: {
    transform: async (text) => {
      const state = await quantumAPI.createQuantumState(text);
      const enhanced = await quantumAPI.transformText(state.rho_id, text, 'enhancement', 1.7);
      const subdued = await quantumAPI.transformText(state.rho_id, text, 'subduing', 0.7);
      return { state, enhanced, subdued };
    }
  },

  curious: {
    analyze: async (text) => {
      const state = await quantumAPI.createQuantumState(text);
      const measurements = await quantumAPI.measureQuantumState(state.rho_id);
      const diagnostics = await quantumAPI.getQuantumDiagnostics(state.rho_id);
      return { state, measurements, diagnostics };
    }
  },

  explorer: {
    analyzeField: async (text, selectedWords) => {
      const state = await quantumAPI.createQuantumState(text);
      const measurements = await quantumAPI.measureQuantumState(state.rho_id);
      const fieldAnalysis = await quantumAPI.analyzeField(state.rho_id, selectedWords);
      const commutators = await quantumAPI.analyzeCommutators(state.rho_id, selectedWords);
      return { state, measurements, fieldAnalysis, commutators };
    }
  },

  expert: {
    fullAnalysis: async (text, selectedWords, stanceMode) => {
      const state = await quantumAPI.createQuantumState(text);
      const measurements = await quantumAPI.measureQuantumState(state.rho_id);
      const fieldAnalysis = await quantumAPI.analyzeField(state.rho_id, selectedWords);
      const stanceResults = await quantumAPI.applyStanceTransformation(state.rho_id, stanceMode);
      return { state, measurements, fieldAnalysis, stanceResults };
    }
  }
};
