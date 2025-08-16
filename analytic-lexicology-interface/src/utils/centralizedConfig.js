/**
 * Centralized Configuration for Analytic Lexicology Interface
 * 
 * This module provides unified access to configuration values,
 * eliminating hardcoded URLs and enabling environment-based overrides.
 */

// Configuration object with defaults
const CONFIG = {
  // Service URLs
  API_BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8192',
  
  // Port configuration (for reference)
  PORTS: {
    API_INTERNAL: 8000,
    API_EXTERNAL: 8192,
    WEB_EXTERNAL: 5173,
    ANALYTIC_LEXICOLOGY: 5174
  },
  
  // API configuration
  API: {
    TIMEOUT: 30000, // 30 seconds for quantum operations
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000, // 1 second
    HEADERS: {
      'Content-Type': 'application/json'
    }
  },
  
  // LLM and quantum parameters
  QUANTUM: {
    RHO_DIMENSION: 64,
    EMBEDDING_DIMENSION: 1536,
    DEFAULT_ALPHA: 0.3,
    MAX_TEXT_LENGTH: 10000,
    MIN_TEXT_LENGTH: 10
  },
  
  // Mastery system configuration
  MASTERY: {
    LEVELS: {
      NOVICE: 'novice',
      CURIOUS: 'curious', 
      EXPLORER: 'explorer',
      EXPERT: 'expert'
    },
    PROGRESSION_CRITERIA: {
      'novice→curious': {
        actionsRequired: ['transform_text'],
        minimumInteractions: 1,
        description: 'Complete your first text transformation'
      },
      'curious→explorer': {
        actionsRequired: ['view_quantum_metrics', 'understand_measurements'],
        minimumInteractions: 3,
        description: 'Explore quantum measurements and understand the data'
      },
      'explorer→expert': {
        actionsRequired: ['analyze_lexical_field', 'examine_commutators'],
        minimumInteractions: 5,
        description: 'Analyze lexical fields and word relationships'
      }
    }
  },
  
  // Sample narratives for testing
  SAMPLES: {
    SIMPLE: "The algorithm apologized to its user, but the damage was already done.",
    COMPLEX: "In the liminal space between dream and waking, consciousness danced with possibility.",
    PHILOSOPHICAL: "Consciousness, like quantum superposition, exists in multiple states until observed.",
    LITERARY: "The old man and the sea danced their eternal dance of struggle and surrender."
  },
  
  // UI configuration
  UI: {
    ANIMATION_DURATION: 300,
    DEBOUNCE_DELAY: 500,
    MAX_DISPLAY_RESULTS: 10,
    LOADING_TIMEOUT: 30000
  },
  
  // Development vs Production settings
  DEV: {
    LOG_LEVEL: import.meta.env.MODE === 'development' ? 'debug' : 'info',
    MOCK_API: import.meta.env.VITE_MOCK_API === 'true',
    ENABLE_ANALYTICS: import.meta.env.MODE === 'production'
  }
};

// Helper functions for accessing configuration

/**
 * Get API base URL with optional path
 */
export const getApiUrl = (path = '') => {
  const base = CONFIG.API_BASE_URL.replace(/\/+$/, '');
  const cleanPath = path.startsWith('/') ? path.substring(1) : path;
  return cleanPath ? `${base}/${cleanPath}` : base;
};

/**
 * Get full API configuration for fetch requests
 */
export const getApiConfig = (overrides = {}) => ({
  timeout: CONFIG.API.TIMEOUT,
  headers: { ...CONFIG.API.HEADERS, ...overrides.headers },
  ...overrides
});

/**
 * Get mastery level configuration
 */
export const getMasteryConfig = (level) => {
  if (!CONFIG.MASTERY.LEVELS[level.toUpperCase()]) {
    throw new Error(`Unknown mastery level: ${level}`);
  }
  return {
    level: CONFIG.MASTERY.LEVELS[level.toUpperCase()],
    ...CONFIG.MASTERY
  };
};

/**
 * Get quantum parameters
 */
export const getQuantumConfig = () => CONFIG.QUANTUM;

/**
 * Get sample narratives
 */
export const getSamples = () => CONFIG.SAMPLES;

/**
 * Get UI configuration
 */
export const getUIConfig = () => CONFIG.UI;

/**
 * Check if running in development mode
 */
export const isDevelopment = () => import.meta.env.MODE === 'development';

/**
 * Get environment-specific configuration
 */
export const getEnvConfig = () => ({
  environment: import.meta.env.MODE,
  apiUrl: CONFIG.API_BASE_URL,
  mockApi: CONFIG.DEV.MOCK_API,
  logLevel: CONFIG.DEV.LOG_LEVEL,
  enableAnalytics: CONFIG.DEV.ENABLE_ANALYTICS
});

/**
 * Validate configuration on startup
 */
export const validateConfig = () => {
  const errors = [];
  
  // Check required environment variables
  if (!CONFIG.API_BASE_URL) {
    errors.push('API_BASE_URL is required');
  }
  
  // Validate URLs
  try {
    new URL(CONFIG.API_BASE_URL);
  } catch (e) {
    errors.push(`Invalid API_BASE_URL: ${CONFIG.API_BASE_URL}`);
  }
  
  // Validate quantum parameters
  if (CONFIG.QUANTUM.RHO_DIMENSION <= 0) {
    errors.push('RHO_DIMENSION must be positive');
  }
  
  if (errors.length > 0) {
    throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
  }
  
  return true;
};

/**
 * Log configuration on startup (development only)
 */
export const logConfig = () => {
  if (isDevelopment()) {
    console.log('🔧 Analytic Lexicology Interface Configuration:', {
      environment: import.meta.env.MODE,
      apiUrl: CONFIG.API_BASE_URL,
      ports: CONFIG.PORTS,
      quantum: CONFIG.QUANTUM
    });
  }
};

// Validate configuration on module load
try {
  validateConfig();
  if (isDevelopment()) {
    logConfig();
  }
} catch (error) {
  console.error('Configuration validation failed:', error);
  throw error;
}

export default CONFIG;