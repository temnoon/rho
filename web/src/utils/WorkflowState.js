/**
 * Unified Workflow State Management
 * 
 * Centralized state management for the Unified Quantum Narrative Workbench.
 * Consolidates patterns from NarrativeExplorer, MatrixArchaeologyStudio, 
 * and NarrativeDistillationStudio into a cohesive system.
 * 
 * Features:
 * - Reactive state updates with observers
 * - Stage-based workflow progression
 * - Persistent state across browser sessions
 * - API integration helpers
 * - Progress tracking and notifications
 */

import React from 'react';
import { apiUrl } from './api.js';

// Workflow stages definition
export const WORKFLOW_STAGES = {
  input: { 
    id: 'input', 
    label: 'Input', 
    icon: '📝', 
    description: 'Enter or import narrative text',
    color: '#4CAF50'
  },
  analysis: { 
    id: 'analysis', 
    label: 'Analysis', 
    icon: '🔬', 
    description: 'POVM measurements and attribute extraction',
    color: '#2196F3'
  },
  transform: { 
    id: 'transform', 
    label: 'Transform', 
    icon: '⚗️', 
    description: 'Edit, sequence, and synthesize',
    color: '#FF9800'
  },
  visualize: { 
    id: 'visualize', 
    label: 'Visualize', 
    icon: '📊', 
    description: 'Bures trajectories and quantum state evolution',
    color: '#9C27B0'
  },
  export: { 
    id: 'export', 
    label: 'Export', 
    icon: '💾', 
    description: 'Save, share, and document results',
    color: '#607D8B'
  }
};

// User complexity modes
export const USER_MODES = {
  beginner: { label: 'Beginner', icon: '🌱', complexity: 1 },
  intermediate: { label: 'Intermediate', icon: '🔬', complexity: 2 },
  expert: { label: 'Expert', icon: '⚗️', complexity: 3 },
  research: { label: 'Research', icon: '🏛️', complexity: 4 }
};

// Default state structure
const DEFAULT_STATE = {
  // Core workflow
  currentStage: 'input',
  userMode: 'beginner',
  progress: 0,
  statusMessage: 'Ready to begin',
  
  // Input stage
  narrativeText: '',
  inputSource: 'manual', // manual, file, gutenberg
  inputMetadata: {
    filename: null,
    wordCount: 0,
    characterCount: 0,
    lastModified: null
  },
  
  // Analysis stage
  currentRhoId: null,
  quantumDiagnostics: null,
  povmMeasurements: null,
  extractedAttributes: null,
  integrabilityResults: null,
  residueAnalysis: null,
  distillationStrategy: 'comprehensive',
  channelType: 'rank_one_update',
  readingAlpha: 0.3,
  
  // Transform stage
  transformations: [],
  invariantEdits: [],
  sequences: [],
  syntheses: [],
  activeTransformation: null,
  transformHistory: [],
  
  // Visualize stage
  trajectoryVisualizations: [],
  stateComparisons: [],
  eigenvalueFlows: [],
  entropyLandscapes: [],
  activeVisualization: null,
  
  // Export stage
  savedResults: [],
  generatedReports: [],
  exportFormats: ['json', 'csv', 'pdf'],
  archiveItems: [],
  
  // Advanced operations
  matrixLibrary: [],
  selectedMatrices: new Set(),
  qualityAssessments: {},
  synthesisRecommendations: [],
  
  // UI state
  loading: false,
  notifications: [],
  contextualSuggestions: [],
  advancedSidebarOpen: false,
  activePanel: null,
  
  // Session metadata
  sessionId: null,
  lastUpdated: null,
  autoSaveEnabled: true,
  version: '1.0.0'
};

/**
 * WorkflowState Class
 * 
 * Manages unified state with reactive updates, persistence, and API integration
 */
export class WorkflowState {
  constructor(initialState = {}) {
    this.state = { ...DEFAULT_STATE, ...initialState };
    this.observers = new Set();
    this.sessionId = this.generateSessionId();
    this.state.sessionId = this.sessionId;
    
    // Initialize from localStorage if available
    this.loadFromStorage();
    
    // Set up auto-save
    if (this.state.autoSaveEnabled) {
      this.setupAutoSave();
    }
    
    console.log('[WorkflowState] Initialized with session:', this.sessionId);
  }
  
  /**
   * Generate unique session ID
   */
  generateSessionId() {
    return `rho_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Subscribe to state changes
   */
  subscribe(observer) {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }
  
  /**
   * Notify all observers of state changes
   */
  notifyObservers(changedKeys = []) {
    this.observers.forEach(observer => {
      try {
        observer(this.state, changedKeys);
      } catch (error) {
        console.error('[WorkflowState] Observer error:', error);
      }
    });
  }
  
  /**
   * Update state with change tracking
   */
  updateState(updates, silent = false) {
    const changedKeys = Object.keys(updates);
    const previousState = { ...this.state };
    
    this.state = {
      ...this.state,
      ...updates,
      lastUpdated: Date.now()
    };
    
    if (!silent) {
      this.notifyObservers(changedKeys);
    }
    
    // Auto-save if enabled
    if (this.state.autoSaveEnabled) {
      this.saveToStorage();
    }
    
    console.log('[WorkflowState] Updated:', changedKeys);
    return previousState;
  }
  
  /**
   * Get current state (read-only)
   */
  getState() {
    return { ...this.state };
  }
  
  /**
   * Get specific state value
   */
  get(key) {
    return this.state[key];
  }
  
  /**
   * Set specific state value
   */
  set(key, value) {
    this.updateState({ [key]: value });
  }
  
  /**
   * Progress to next workflow stage
   */
  progressToStage(stageId) {
    if (!WORKFLOW_STAGES[stageId]) {
      console.error('[WorkflowState] Invalid stage:', stageId);
      return false;
    }
    
    const previousStage = this.state.currentStage;
    this.updateState({ 
      currentStage: stageId,
      statusMessage: `Moved to ${WORKFLOW_STAGES[stageId].label} stage`
    });
    
    this.addNotification(`Moved to ${WORKFLOW_STAGES[stageId].label} stage`, 'success');
    console.log('[WorkflowState] Stage progression:', previousStage, '→', stageId);
    return true;
  }
  
  /**
   * Add notification to queue
   */
  addNotification(message, type = 'info', duration = 5000) {
    const notification = {
      id: Date.now(),
      message,
      type,
      timestamp: Date.now()
    };
    
    const notifications = [...this.state.notifications, notification];
    this.updateState({ notifications });
    
    // Auto-remove after duration
    if (duration > 0) {
      setTimeout(() => {
        this.removeNotification(notification.id);
      }, duration);
    }
    
    return notification.id;
  }
  
  /**
   * Remove notification by ID
   */
  removeNotification(notificationId) {
    const notifications = this.state.notifications.filter(n => n.id !== notificationId);
    this.updateState({ notifications });
  }
  
  /**
   * Update loading state with progress
   */
  setLoading(loading, progress = null, statusMessage = null) {
    const updates = { loading };
    if (progress !== null) updates.progress = progress;
    if (statusMessage !== null) updates.statusMessage = statusMessage;
    this.updateState(updates);
  }
  
  /**
   * Reset workflow to initial state
   */
  resetWorkflow(preserveSettings = true) {
    const preserved = preserveSettings ? {
      userMode: this.state.userMode,
      autoSaveEnabled: this.state.autoSaveEnabled,
      advancedSidebarOpen: this.state.advancedSidebarOpen
    } : {};
    
    this.updateState({
      ...DEFAULT_STATE,
      ...preserved,
      sessionId: this.generateSessionId(),
      lastUpdated: Date.now()
    });
    
    this.addNotification('Workflow reset', 'info');
    console.log('[WorkflowState] Workflow reset');
  }
  
  /**
   * Check if stage is completed
   */
  isStageCompleted(stageId) {
    switch (stageId) {
      case 'input': 
        return this.state.narrativeText.length > 0;
      case 'analysis': 
        return this.state.currentRhoId !== null;
      case 'transform': 
        return this.state.transformations.length > 0;
      case 'visualize': 
        return this.state.trajectoryVisualizations.length > 0;
      case 'export': 
        return this.state.savedResults.length > 0;
      default: 
        return false;
    }
  }
  
  /**
   * Check if stage is accessible based on workflow progression
   */
  isStageAccessible(stageId) {
    const stages = Object.keys(WORKFLOW_STAGES);
    const currentIndex = stages.indexOf(this.state.currentStage);
    const targetIndex = stages.indexOf(stageId);
    
    // Allow access to current stage, previous stages, and next immediate stage
    return targetIndex <= currentIndex + 1;
  }
  
  /**
   * API helper: Safe fetch with error handling
   */
  async safeFetch(path, opts = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    try {
      const url = apiUrl(path);
      const res = await fetch(url, {
        ...opts,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        let bodyText = '<no body>';
        try {
          bodyText = await res.text();
        } catch (e) {
          bodyText = `<unable to read body: ${String(e)}>`;
        }
        const err = new Error(`${url} returned ${res.status} ${res.statusText} - ${bodyText}`);
        err.status = res.status;
        throw err;
      }
      
      return res;
    } catch (err) {
      clearTimeout(timeoutId);
      
      if (err.name === 'AbortError') {
        const timeoutErr = new Error(`Request timeout: ${path} took longer than 10 seconds`);
        console.error('[WorkflowState] Request timeout for', path);
        this.addNotification('Backend not responding - check if API server is running on port 8192', 'error');
        throw timeoutErr;
      }
      
      console.error('[WorkflowState] Network error for', path, err);
      this.addNotification(`Network error: ${err.message}`, 'error');
      throw err;
    }
  }
  
  /**
   * Create quantum state from narrative text
   */
  async createQuantumState() {
    if (!this.state.narrativeText.trim()) {
      this.addNotification('Please enter some text first', 'warning');
      return null;
    }
    
    this.setLoading(true, 10, 'Creating quantum density matrix...');
    
    try {
      // Initialize rho matrix
      const initRes = await this.safeFetch('/rho/init', { method: 'POST' });
      const initData = await initRes.json();
      const rhoId = initData.rho_id;
      
      this.setLoading(true, 30, 'Reading text into quantum state...');
      
      // Read text into matrix
      const readRes = await this.safeFetch(`/rho/${rhoId}/read_channel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          raw_text: this.state.narrativeText, 
          alpha: this.state.readingAlpha,
          channel_type: this.state.channelType
        })
      });
      const readData = await readRes.json();
      
      this.updateState({ 
        currentRhoId: rhoId,
        quantumDiagnostics: readData.diagnostics
      });
      
      this.setLoading(false, 100, 'Quantum state ready for analysis');
      this.addNotification('Quantum state created successfully', 'success');
      
      return rhoId;
    } catch (error) {
      console.error('[WorkflowState] Failed to create quantum state:', error);
      this.setLoading(false, 0, 'Ready to begin');
      this.addNotification('Failed to create quantum state', 'error');
      return null;
    }
  }
  
  /**
   * Apply POVM measurements
   */
  async runPOVMMeasurements(packId = 'advanced_narrative_pack') {
    if (!this.state.currentRhoId) {
      this.addNotification('Create quantum state first', 'warning');
      return null;
    }
    
    this.setLoading(true, 50, 'Applying POVM measurements...');
    
    try {
      const measureRes = await this.safeFetch(`/packs/measure/${this.state.currentRhoId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pack_id: packId })
      });
      const measureData = await measureRes.json();
      
      this.updateState({ 
        povmMeasurements: measureData.measurements,
        extractedAttributes: measureData
      });
      
      this.setLoading(false, 100, 'POVM measurements completed');
      this.addNotification('POVM measurements completed', 'success');
      
      return measureData;
    } catch (error) {
      console.error('[WorkflowState] POVM measurements failed:', error);
      this.setLoading(false, 0, 'Ready to begin');
      this.addNotification('POVM measurements failed', 'error');
      return null;
    }
  }
  
  /**
   * Save state to localStorage
   */
  saveToStorage() {
    try {
      const stateToSave = {
        ...this.state,
        // Don't save transient UI state
        loading: false,
        notifications: [],
        contextualSuggestions: []
      };
      
      localStorage.setItem(`rho_workflow_${this.sessionId}`, JSON.stringify(stateToSave));
      localStorage.setItem('rho_latest_session', this.sessionId);
    } catch (error) {
      console.error('[WorkflowState] Save to storage failed:', error);
    }
  }
  
  /**
   * Load state from localStorage
   */
  loadFromStorage() {
    try {
      const latestSession = localStorage.getItem('rho_latest_session');
      if (latestSession) {
        const savedState = localStorage.getItem(`rho_workflow_${latestSession}`);
        if (savedState) {
          const parsed = JSON.parse(savedState);
          this.state = { ...this.state, ...parsed };
          this.sessionId = latestSession;
          console.log('[WorkflowState] Loaded from storage:', latestSession);
        }
      }
    } catch (error) {
      console.error('[WorkflowState] Load from storage failed:', error);
    }
  }
  
  /**
   * Setup auto-save interval
   */
  setupAutoSave() {
    setInterval(() => {
      if (this.state.autoSaveEnabled) {
        this.saveToStorage();
      }
    }, 30000); // Save every 30 seconds
  }
  
  /**
   * Export state as JSON
   */
  exportState() {
    return JSON.stringify(this.state, null, 2);
  }
  
  /**
   * Import state from JSON
   */
  importState(jsonState) {
    try {
      const parsed = JSON.parse(jsonState);
      this.updateState(parsed);
      this.addNotification('State imported successfully', 'success');
      return true;
    } catch (error) {
      console.error('[WorkflowState] Import failed:', error);
      this.addNotification('Failed to import state', 'error');
      return false;
    }
  }
}

/**
 * Create singleton instance for global access
 */
export const workflowState = new WorkflowState();

/**
 * React hook for using workflow state in components
 */
export function useWorkflowState() {
  const [state, setState] = React.useState(() => {
    return workflowState.getState();
  });
  
  React.useEffect(() => {
    const unsubscribe = workflowState.subscribe((newState) => {
      setState(newState);
    });
    
    return unsubscribe;
  }, []);
  
  return {
    state,
    updateState: (updates) => workflowState.updateState(updates),
    progressToStage: (stageId) => workflowState.progressToStage(stageId),
    addNotification: (message, type, duration) => workflowState.addNotification(message, type, duration),
    removeNotification: (id) => workflowState.removeNotification(id),
    setLoading: (loading, progress, statusMessage) => workflowState.setLoading(loading, progress, statusMessage),
    resetWorkflow: (preserveSettings) => workflowState.resetWorkflow(preserveSettings),
    isStageCompleted: (stageId) => workflowState.isStageCompleted(stageId),
    isStageAccessible: (stageId) => workflowState.isStageAccessible(stageId),
    createQuantumState: () => workflowState.createQuantumState(),
    runPOVMMeasurements: (packId) => workflowState.runPOVMMeasurements(packId),
    exportState: () => workflowState.exportState(),
    importState: (jsonState) => workflowState.importState(jsonState)
  };
}

export default WorkflowState;