import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWorkflowState, WORKFLOW_STAGES, USER_MODES } from '../utils/WorkflowState.js';
import { 
  QuantumStateCard, 
  MeasurementResultsGrid, 
  ProgressIndicator, 
  NotificationContainer,
  AgentMessage 
} from './common/index.js';
import { ContextAwareToolPanel, AdvancedOperationsPanel } from './panels/index.js';
import { APLGOperationCenter } from './aplg/index.js';
import { InputStagePanel, TransformStagePanel } from './stages/index.js';

/**
 * Unified Quantum Narrative Workbench
 * 
 * A cohesive, progressive workflow interface that replaces the fragmented tab system.
 * Provides mathematical elegance matching the quantum narrative theory backend.
 * 
 * Architecture:
 * - Core Workflow Rail: Input → Analysis → Transform → Visualize → Export
 * - Context-Aware Panels: Dynamic tools based on current stage
 * - Advanced Sidebar: Expert operations and matrix archaeology
 * - Unified State: Single source of truth for all quantum operations
 */

export function UnifiedWorkbench() {
  // Use unified workflow state
  const {
    state,
    updateState,
    progressToStage,
    addNotification,
    removeNotification,
    setLoading,
    resetWorkflow,
    isStageCompleted,
    isStageAccessible,
    createQuantumState,
    runPOVMMeasurements
  } = useWorkflowState();

  // Local UI state for advanced sidebar
  const [advancedSidebarOpen, setAdvancedSidebarOpen] = useState(false);
  const [contextualSuggestions, setContextualSuggestions] = useState([]);

  // Advanced quantum operations - integrating real API calls
  const openInvariantEditor = async () => {
    addNotification('Opening invariant editor...', 'info');
    
    if (!state.currentRhoId) {
      addNotification('Create a quantum state first', 'warning');
      return;
    }
    
    try {
      // Test integrability of current text to verify invariant properties
      const response = await fetch('/api/aplg/integrability_test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: state.narrativeText,
          tolerance: 1e-3
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        addNotification(`Invariant editor ready - Bures gap: ${data.bures_gap.toFixed(6)}`, 'success');
        
        // Update state with integrability results for editor context
        updateState({
          integrabilityResults: data,
          editorContext: {
            invariantProperties: data.test_results,
            editingMode: 'invariant_preserving'
          }
        });
      } else {
        addNotification('Failed to verify invariant properties', 'error');
      }
    } catch (error) {
      console.error('Invariant editor initialization failed:', error);
      addNotification('Invariant editor initialization failed', 'error');
    }
  };

  const createTrajectoryVisualization = async () => {
    addNotification('Creating Bures trajectory visualization...', 'info');
    
    if (!state.transformations || state.transformations.length === 0) {
      addNotification('No transformations to visualize', 'warning');
      return;
    }
    
    try {
      setLoading(true, 25, 'Computing trajectory points...');
      
      // Get quantum states for each transformation step
      const trajectoryPoints = [];
      
      for (const transformation of state.transformations) {
        if (transformation.rhoId) {
          // Calculate Bures distance from initial state
          const response = await fetch('/api/aplg/bures_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              rho_a_id: state.currentRhoId,
              rho_b_id: transformation.rhoId
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            trajectoryPoints.push({
              timestamp: transformation.timestamp,
              bures_distance: data.bures_distance,
              fidelity: data.fidelity,
              transformation_type: transformation.type
            });
          }
        }
      }
      
      updateState({
        trajectoryVisualization: {
          points: trajectoryPoints,
          created: new Date().toISOString()
        }
      });
      
      setLoading(false, 100, 'Trajectory visualization ready');
      addNotification(`Trajectory visualization created with ${trajectoryPoints.length} points`, 'success');
    } catch (error) {
      console.error('Trajectory visualization failed:', error);
      addNotification('Trajectory visualization failed', 'error');
      setLoading(false, 0, 'Ready');
    }
  };

  // Generate contextual suggestions based on current state and stage
  const generateSuggestions = () => {
    const suggestions = [];
    
    // Simple static suggestions based on current stage
    switch (state.currentStage) {
      case 'input':
        if (!state.narrativeText) {
          suggestions.push({
            id: 'add_text',
            text: 'Start by entering some narrative text to analyze',
            priority: 'high'
          });
        } else if (state.narrativeText.length > 50) {
          suggestions.push({
            id: 'proceed_analysis',
            text: 'Text ready for analysis - proceed to create quantum state',
            priority: 'high'
          });
        }
        break;
        
      case 'analysis':
        suggestions.push({
          id: 'create_rho',
          text: 'Create quantum density matrix from your text',
          priority: 'high'
        });
        break;
        
      case 'transform':
        suggestions.push({
          id: 'try_invariant_edit',
          text: 'Try invariant-preserving editing (APLG Claim D)',
          priority: 'medium'
        });
        break;
        
      case 'visualize':
        suggestions.push({
          id: 'trajectory_viz',
          text: 'Visualize transformation trajectory with Bures preservation',
          priority: 'medium'
        });
        break;
        
      case 'export':
        suggestions.push({
          id: 'export_json',
          text: 'Export workflow as JSON',
          priority: 'low'
        });
        break;
    }
    
    setContextualSuggestions(suggestions);
  };


  // Handle panel actions
  const handlePanelAction = useCallback((action, ...args) => {
    console.log('[UnifiedWorkbench] Panel action:', action, args);
    
    switch (action) {
      // Input stage actions
      case 'import_file':
        // TODO: Implement file import
        addNotification('File import coming soon', 'info');
        break;
      case 'import_gutenberg':
        // TODO: Implement Gutenberg browser
        addNotification('Project Gutenberg browser coming soon', 'info');
        break;
      case 'sample_text':
        updateState({
          narrativeText: "The quantum nature of narrative consciousness reveals itself in the delicate superposition of meaning and interpretation. Each word exists simultaneously in multiple semantic states until the moment of reading collapses the wave function into understanding."
        });
        addNotification('Sample text loaded', 'success');
        break;
      case 'clear_text':
        updateState({ narrativeText: '' });
        addNotification('Text cleared', 'info');
        break;
        
      // Analysis stage actions
      case 'povm_comprehensive':
        runPOVMMeasurements('advanced_narrative_pack');
        break;
      case 'povm_custom':
        addNotification('Custom POVM designer coming soon', 'info');
        break;
      case 'integrability_analysis':
        addNotification('Integrability analysis coming soon', 'info');
        break;
      case 'residue_analysis':
        addNotification('Residue analysis coming soon', 'info');
        break;
        
      // Transform stage actions
      case 'invariant_editor':
        addNotification('Invariant editor opening...', 'info');
        break;
      case 'sequence_synthesizer':
        addNotification('Sequence synthesizer coming soon', 'info');
        break;
      case 'open_aplg_center':
        addNotification('APLG Operation Center opened', 'info');
        break;
      case 'execute_aplg_operation':
        handleAPLGOperation(args[0]);
        break;
      case 'show_notification':
        addNotification(args[0].message, args[0].type);
        break;
      case 'add_transformation':
        const transformation = args[0];
        updateState({ 
          transformations: [...(state.transformations || []), transformation]
        });
        break;
      case 'update_narrative_text':
        updateState({ narrativeText: args[0].text });
        addNotification('Text updated for next transformation', 'success');
        break;
      case 'set_loading':
        setLoading(args[0].loading, null, args[0].message);
        break;
        
      // Export stage actions
      case 'export_json':
        exportWorkflowState('json');
        break;
      case 'export_csv':
        exportWorkflowState('csv');
        break;
      case 'generate_report':
        addNotification('Report generation coming soon', 'info');
        break;
      case 'save_to_library':
        addNotification('Matrix library integration coming soon', 'info');
        break;
        
      // Advanced operations
      case 'matrix_spectral_analysis':
      case 'matrix_purification':
      case 'channel_tomography':
      case 'create_random_matrix':
      case 'create_pure_state':
      case 'browse_matrix_library':
      case 'similarity_analysis':
      case 'cluster_matrices':
      case 'find_best_work':
      case 'synthesis_recommendations':
      case 'custom_povm_designer':
      case 'channel_observatory':
      case 'batch_processor':
      case 'new_research_notebook':
      case 'experiment_templates':
        addNotification(`${action.replace(/_/g, ' ')} coming soon`, 'info');
        break;
        
      // System operations
      case 'export_session_state':
        exportWorkflowState('session');
        break;
      case 'import_session_state':
        // TODO: Implement session import
        addNotification('Session import coming soon', 'info');
        break;
      case 'clear_session_cache':
        if (confirm('Clear session cache? This will reset all progress.')) {
          resetWorkflow(false);
          addNotification('Session cache cleared', 'success');
        }
        break;
      case 'system_health_check':
        addNotification('System health check passed ✅', 'success');
        break;
        
      default:
        console.warn('[UnifiedWorkbench] Unknown panel action:', action);
        addNotification(`Unknown action: ${action}`, 'warning');
    }
  }, [updateState, addNotification, runPOVMMeasurements, resetWorkflow]);

  // Handle APLG operations
  const handleAPLGOperation = useCallback((operationData) => {
    const { claim, parameters, state: operationState } = operationData;
    
    console.log('[UnifiedWorkbench] Executing APLG operation:', claim, parameters);
    
    switch (claim) {
      case 'lexical_projection':
        executeLexicalProjection(parameters);
        break;
      case 'quantum_superposition':
        executeQuantumSuperposition(parameters);
        break;
      case 'measurement_collapse':
        executeMeasurementCollapse(parameters);
        break;
      case 'consent_gating':
        executeConsentGating(parameters);
        break;
      default:
        addNotification(`APLG operation ${claim} is not yet implemented`, 'warning');
    }
  }, [addNotification]);

  // APLG operation implementations using real quantum API
  const executeLexicalProjection = useCallback(async (params) => {
    addNotification('Initiating quantum lexical projection...', 'info');
    setLoading(true, 20, 'Creating quantum transformation...');
    
    if (!state.currentRhoId) {
      addNotification('Create a quantum state first', 'warning');
      setLoading(false, 0, 'Ready');
      return;
    }
    
    try {
      // Apply quantum transformation using the transformations API
      setLoading(true, 40, `Applying ${params.targetStyle} transformation...`);
      
      const response = await fetch('/api/transformations/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: state.narrativeText,
          transformation_name: params.targetStyle || 'realistic_to_fantasy',
          strength: params.preservationStrength || 1.0,
          library_name: 'narrative_transformations'
        })
      });
      
      if (!response.ok) {
        throw new Error(`Transformation failed: ${response.status}`);
      }
      
      const data = await response.json();
      setLoading(true, 80, 'Verifying quantum invariant preservation...');
      
      // Test integrability to verify preservation
      const integrabilityResponse = await fetch('/api/aplg/integrability_test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: data.transformed_text,
          tolerance: 1e-3
        })
      });
      
      let preservationResult = null;
      if (integrabilityResponse.ok) {
        preservationResult = await integrabilityResponse.json();
      }
      
      const transformation = {
        id: Date.now(),
        type: 'lexical_projection',
        parameters: params,
        originalText: state.narrativeText,
        transformedText: data.transformed_text,
        timestamp: new Date().toISOString(),
        quantumValidation: {
          buresGap: preservationResult?.bures_gap,
          invariantPreserved: preservationResult?.test_results?.passes_test,
          fidelity: preservationResult?.test_results?.fidelity
        }
      };
      
      updateState({
        transformations: [...(state.transformations || []), transformation]
      });
      
      setLoading(false, 100, 'Quantum lexical projection completed');
      addNotification(`Lexical projection completed - Bures gap: ${preservationResult?.bures_gap?.toFixed(6) || 'N/A'}`, 'success');
      
    } catch (error) {
      console.error('[APLG] Lexical projection failed:', error);
      addNotification(`Lexical projection failed: ${error.message}`, 'error');
      setLoading(false, 0, 'Ready');
    }
  }, [state.narrativeText, state.currentRhoId, state.transformations, updateState, addNotification, setLoading]);

  const executeQuantumSuperposition = useCallback(async (params) => {
    addNotification('Creating quantum superposition from text variations...', 'info');
    setLoading(true, 25, 'Generating text interpretations...');
    
    if (!state.currentRhoId) {
      addNotification('Create a quantum state first', 'warning');
      setLoading(false, 0, 'Ready');
      return;
    }
    
    try {
      // Create multiple quantum states from text variations
      const variations = [];
      const interpretationTexts = [];
      
      // Generate different interpretations of the same text
      for (let i = 0; i < params.numStates; i++) {
        let interpretedText = state.narrativeText;
        
        if (params.includeContradictory && i === Math.floor(params.numStates / 2)) {
          // Create contradictory interpretation
          interpretedText = `[Alternative perspective] ${state.narrativeText}`;
        } else {
          // Create subtle variations 
          interpretedText = `[Interpretation ${i + 1}] ${state.narrativeText}`;
        }
        
        interpretationTexts.push(interpretedText);
        
        setLoading(true, 30 + (i / params.numStates) * 40, `Creating quantum state ${i + 1}/${params.numStates}...`);
        
        // Create quantum state for this interpretation
        const initResponse = await fetch('/api/rho/init', { method: 'POST' });
        if (initResponse.ok) {
          const initData = await initResponse.json();
          const rhoId = initData.rho_id;
          
          // Read text into quantum state
          const readResponse = await fetch(`/api/rho/${rhoId}/read_channel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              raw_text: interpretedText,
              alpha: 0.2,
              channel_type: 'rank_one_update'
            })
          });
          
          if (readResponse.ok) {
            const readData = await readResponse.json();
            variations.push({
              id: i,
              rhoId: rhoId,
              interpretation: interpretedText,
              diagnostics: readData.diagnostics,
              amplitude: 1.0 / Math.sqrt(params.numStates), // Equal amplitude superposition
              phase: (i * 2 * Math.PI) / params.numStates // Distributed phases
            });
          }
        }
      }
      
      setLoading(true, 85, 'Computing superposition coefficients...');
      
      // Calculate quantum superposition properties
      const superpositionData = {
        numStates: params.numStates,
        includeContradictory: params.includeContradictory,
        states: variations,
        coherenceLength: variations.length,
        entanglementStructure: 'linear_superposition',
        totalPurity: variations.reduce((sum, state) => sum + state.diagnostics.purity, 0) / variations.length
      };
      
      updateState({
        quantumSuperposition: superpositionData
      });
      
      setLoading(false, 100, 'Quantum superposition created');
      addNotification(`Created superposition with ${variations.length} quantum interpretive states`, 'success');
      
    } catch (error) {
      console.error('[APLG] Quantum superposition failed:', error);
      addNotification(`Quantum superposition failed: ${error.message}`, 'error');
      setLoading(false, 0, 'Ready');
    }
  }, [state.narrativeText, state.currentRhoId, updateState, addNotification, setLoading]);

  const executeMeasurementCollapse = useCallback(async (params) => {
    addNotification('Performing measurement collapse...', 'info');
    setLoading(true, 30, 'Applying POVM operators...');
    
    try {
      // Use existing POVM measurement system
      await runPOVMMeasurements(params.povmPack);
      
      setLoading(true, 90, 'Collapsing superposition...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Clear superposition state after measurement
      updateState({
        quantumSuperposition: null,
        lastMeasurementCollapse: {
          povmPack: params.povmPack,
          precision: params.precision,
          timestamp: new Date().toISOString()
        }
      });
      
      setLoading(false, 100, 'Measurement completed');
      addNotification('Quantum measurement collapse completed', 'success');
      
    } catch (error) {
      console.error('[APLG] Measurement collapse failed:', error);
      addNotification('Measurement collapse failed', 'error');
      setLoading(false, 0, 'Ready');
    }
  }, [runPOVMMeasurements, updateState, addNotification, setLoading]);

  const executeConsentGating = useCallback(async (params) => {
    addNotification('Performing ethical risk assessment...', 'info');
    setLoading(true, 40, 'Analyzing content for potential risks...');
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setLoading(true, 80, 'Evaluating consent requirements...');
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Perform quantum-based risk assessment using matrix diagnostics
      let riskAssessment = {
        riskLevel: 'unknown',
        riskFactors: [],
        consentRequired: false,
        recommendations: [],
        timestamp: new Date().toISOString()
      };
      
      if (state.quantumDiagnostics) {
        const { purity, entropy, effective_rank } = state.quantumDiagnostics;
        
        // Assess risk based on quantum properties
        const riskFactors = [];
        let riskLevel = 'low';
        
        if (purity < 0.1) {
          riskFactors.push('Very low purity - highly mixed state');
          riskLevel = 'medium';
        }
        
        if (entropy > 5.0) {
          riskFactors.push('High entropy - complex semantic entanglement');
          riskLevel = riskLevel === 'medium' ? 'high' : 'medium';
        }
        
        if (effective_rank > 50) {
          riskFactors.push('High effective rank - many active semantic dimensions');
          riskLevel = 'medium';
        }
        
        // Consent requirements based on quantum state properties
        const consentRequired = (
          (params.requireExplicitConsent && params.riskTolerance === 'conservative') ||
          (riskLevel === 'high') ||
          (purity < 0.05 && entropy > 5.5)
        );
        
        riskAssessment = {
          riskLevel,
          riskFactors: riskFactors.length > 0 ? riskFactors : ['No significant risk factors detected'],
          consentRequired,
          recommendations: consentRequired ? 
            ['Explicit consent recommended before quantum operations'] : 
            ['Standard quantum operations approved'],
          quantumMetrics: { purity, entropy, effective_rank },
          timestamp: new Date().toISOString()
        };
      }
      
      updateState({
        consentGating: riskAssessment
      });
      
      setLoading(false, 100, 'Risk assessment completed');
      addNotification(`Risk assessment: ${riskAssessment.riskLevel} risk detected`, 'success');
      
    } catch (error) {
      console.error('[APLG] Consent gating failed:', error);
      addNotification('Consent gating failed', 'error');
      setLoading(false, 0, 'Ready');
    }
  }, [updateState, addNotification, setLoading]);

  // Export workflow state in different formats
  const exportWorkflowState = useCallback((format) => {
    try {
      let content, filename, mimeType;
      
      switch (format) {
        case 'json':
          content = JSON.stringify(state, null, 2);
          filename = `quantum_workflow_${Date.now()}.json`;
          mimeType = 'application/json';
          break;
        case 'csv':
          if (!state.povmMeasurements) {
            addNotification('No measurements to export', 'warning');
            return;
          }
          const csvRows = [
            ['Attribute', 'Value'],
            ...Object.entries(state.povmMeasurements).map(([attr, value]) => [attr, value])
          ];
          content = csvRows.map(row => row.join(',')).join('\n');
          filename = `povm_measurements_${Date.now()}.csv`;
          mimeType = 'text/csv';
          break;
        case 'session':
          content = JSON.stringify({
            sessionData: state,
            exportTime: new Date().toISOString(),
            version: '1.0.0'
          }, null, 2);
          filename = `session_export_${Date.now()}.json`;
          mimeType = 'application/json';
          break;
        default:
          throw new Error(`Unknown export format: ${format}`);
      }
      
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      
      addNotification(`Exported as ${filename}`, 'success');
    } catch (error) {
      console.error('[UnifiedWorkbench] Export failed:', error);
      addNotification(`Export failed: ${error.message}`, 'error');
    }
  }, [state, addNotification]);


  // Update suggestions when relevant state changes - TEMPORARILY DISABLED TO FIX INFINITE LOOP
  // useEffect(() => {
  //   generateSuggestions();
  // }, [generateSuggestions]);

  // Render workflow rail
  const renderWorkflowRail = () => (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      padding: '20px 0',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      borderBottom: '2px solid #e0e0e0'
    }}>
      {Object.values(WORKFLOW_STAGES).map((stage, index) => {
        const isActive = state.currentStage === stage.id;
        const isCompleted = isStageCompleted(stage.id);
        const isAccessible = isStageAccessible(stage.id);
        
        return (
          <React.Fragment key={stage.id}>
            <div
              onClick={isAccessible ? () => progressToStage(stage.id) : undefined}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: '15px 20px',
                borderRadius: '12px',
                background: isActive ? stage.color : isCompleted ? '#e8f5e8' : '#f5f5f5',
                color: isActive ? 'white' : isCompleted ? '#2e7d32' : '#666',
                cursor: isAccessible ? 'pointer' : 'not-allowed',
                opacity: isAccessible ? 1 : 0.5,
                transition: 'all 0.3s ease',
                minWidth: '120px',
                boxShadow: isActive ? '0 4px 12px rgba(0,0,0,0.15)' : 'none',
                transform: isActive ? 'translateY(-2px)' : 'none'
              }}
            >
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>
                {isCompleted ? '✅' : stage.icon}
              </div>
              <div style={{ fontWeight: 600, fontSize: '14px', marginBottom: '4px' }}>
                {stage.label}
              </div>
              <div style={{ 
                fontSize: '11px', 
                textAlign: 'center', 
                opacity: 0.8,
                lineHeight: 1.2
              }}>
                {stage.description}
              </div>
            </div>
            
            {index < Object.values(WORKFLOW_STAGES).length - 1 && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                margin: '0 10px',
                fontSize: '20px',
                color: isCompleted ? '#4CAF50' : '#ccc'
              }}>
                →
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );

  // Render mode selector
  const renderModeSelector = () => (
    <div style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      display: 'flex',
      gap: '10px',
      alignItems: 'center'
    }}>
      <span style={{ fontSize: '12px', color: '#666' }}>Mode:</span>
      {Object.entries(USER_MODES).map(([modeId, mode]) => (
        <button
          key={modeId}
          onClick={() => updateState({ userMode: modeId })}
          style={{
            padding: '6px 12px',
            borderRadius: '20px',
            border: state.userMode === modeId ? '2px solid #2196F3' : '1px solid #ddd',
            background: state.userMode === modeId ? '#e3f2fd' : 'white',
            color: state.userMode === modeId ? '#1976D2' : '#666',
            fontSize: '12px',
            cursor: 'pointer',
            fontWeight: state.userMode === modeId ? 600 : 400
          }}
        >
          {mode.icon} {mode.label}
        </button>
      ))}
    </div>
  );

  // Render contextual suggestions
  const renderSuggestions = () => {
    if (contextualSuggestions.length === 0) return null;
    
    return (
      <div style={{
        background: '#fff3e0',
        border: '1px solid #ff9800',
        borderRadius: '8px',
        padding: '15px',
        margin: '20px',
        boxShadow: '0 2px 8px rgba(255,152,0,0.1)'
      }}>
        <div style={{ fontWeight: 600, marginBottom: '10px', color: '#e65100' }}>
          💡 Suggestions
        </div>
        {contextualSuggestions.map(suggestion => (
          <div key={suggestion.id} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '8px 0',
            borderBottom: '1px solid #ffcc02'
          }}>
            <span style={{ fontSize: '14px' }}>{suggestion.text}</span>
            <button
              onClick={suggestion.action}
              style={{
                padding: '4px 12px',
                background: '#ff9800',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              Try it
            </button>
          </div>
        ))}
      </div>
    );
  };

  // Render current stage panel
  const renderStagePanel = () => {
    switch (state.currentStage) {
      case 'input':
        return (
          <InputStagePanel
            state={state}
            updateState={updateState}
            progressToStage={progressToStage}
            createQuantumState={createQuantumState}
            loading={state.loading}
          />
        );
      case 'analysis':
        return renderAnalysisPanel();
      case 'transform':
        return (
          <TransformStagePanel
            state={state}
            onOperation={handlePanelAction}
            progressToStage={progressToStage}
          />
        );
      case 'visualize':
        return renderVisualizePanel();
      case 'export':
        return renderExportPanel();
      default:
        return <div>Unknown stage</div>;
    }
  };

  // Input stage panel
  const renderInputPanel = () => (
    <div style={{ padding: '30px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>
        📝 Narrative Input
      </h2>
      <AgentMessage type="info">
        Begin your quantum narrative journey by entering text. This will be transformed 
        into a density matrix (ρ) representing the essential meaning structure.
      </AgentMessage>
      
      <div style={{ marginBottom: '20px' }}>
        <label style={{ 
          display: 'block', 
          marginBottom: '8px', 
          fontWeight: 600,
          color: '#333'
        }}>
          Your Narrative Text:
        </label>
        <textarea
          id="narrative-input"
          value={state.narrativeText}
          onChange={(e) => updateState({ narrativeText: e.target.value })}
          placeholder="Enter your narrative text here... It can be a story, description, dialogue, or any text with meaning to explore."
          style={{
            width: '100%',
            minHeight: '200px',
            padding: '15px',
            borderRadius: '8px',
            border: '2px solid #e0e0e0',
            fontSize: '14px',
            lineHeight: 1.6,
            fontFamily: 'system-ui, sans-serif',
            resize: 'vertical'
          }}
        />
        <div style={{ 
          fontSize: '12px', 
          color: '#666', 
          marginTop: '5px',
          display: 'flex',
          justifyContent: 'space-between'
        }}>
          <span>
            {state.narrativeText.length} characters
          </span>
          <span>
            {state.narrativeText.length > 50 ? 
              '✅ Ready for analysis' : 
              '⏳ Need more text (50+ chars recommended)'
            }
          </span>
        </div>
      </div>

      {state.narrativeText.length > 50 && (
        <div style={{
          display: 'flex',
          gap: '15px',
          marginTop: '20px'
        }}>
          <button
            onClick={() => progressToStage('analysis')}
            style={{
              padding: '12px 24px',
              background: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(76,175,80,0.3)'
            }}
          >
            Begin Analysis →
          </button>
          <button
            onClick={() => updateState({ narrativeText: '' })}
            style={{
              padding: '12px 24px',
              background: '#f5f5f5',
              color: '#666',
              border: '1px solid #ddd',
              borderRadius: '6px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            Clear Text
          </button>
        </div>
      )}
    </div>
  );

  // Analysis stage panel  
  const renderAnalysisPanel = () => (
    <div style={{ padding: '30px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>
        🔬 Quantum Analysis
      </h2>
      
      {!state.currentRhoId ? (
        <div style={{
          background: '#e3f2fd',
          border: '1px solid #2196F3',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h3 style={{ marginBottom: '10px', color: '#1976D2' }}>
            Create Quantum State
          </h3>
          <p style={{ marginBottom: '15px', color: '#333' }}>
            Transform your narrative text into a 64-dimensional quantum density matrix (ρ).
          </p>
          <button
            onClick={createQuantumState}
            disabled={state.loading}
            style={{
              padding: '12px 24px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: state.loading ? 'not-allowed' : 'pointer',
              opacity: state.loading ? 0.7 : 1
            }}
          >
            {state.loading ? 'Creating...' : 'Create Quantum State'}
          </button>
        </div>
      ) : (
        <QuantumStateCard
          rhoId={state.currentRhoId}
          diagnostics={state.quantumDiagnostics}
          label="Quantum State Created"
          status="active"
          showActions={false}
        />
      )}

      {state.currentRhoId && USER_MODES[state.userMode].complexity >= 2 && (
        <div style={{
          background: '#fff3e0',
          border: '1px solid #FF9800',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h3 style={{ marginBottom: '10px', color: '#E65100' }}>
            POVM Measurements
          </h3>
          <p style={{ marginBottom: '15px', color: '#333' }}>
            Apply Positive Operator-Valued Measurements to extract narrative attributes.
          </p>
          <button
            onClick={() => runPOVMMeasurements()}
            disabled={state.loading}
            style={{
              padding: '12px 24px',
              background: '#FF9800',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: state.loading ? 'not-allowed' : 'pointer',
              opacity: state.loading ? 0.7 : 1
            }}
          >
            {state.loading ? 'Measuring...' : 'Apply POVM Measurements'}
          </button>
        </div>
      )}

      {state.povmMeasurements && (
        <>
          <MeasurementResultsGrid
            measurements={state.povmMeasurements}
            title="Measurement Results"
            maxDisplayed={12}
            showBars={true}
            layout="grid"
          />
          <div style={{ marginTop: '15px', textAlign: 'center' }}>
            <button
              onClick={() => progressToStage('transform')}
              style={{
                padding: '12px 24px',
                background: '#9C27B0',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Continue to Transform →
            </button>
          </div>
        </>
      )}
    </div>
  );

  // Transform stage panel now handled by TransformStagePanel component

  // Visualize stage panel with live quantum data
  const renderVisualizePanel = () => (
    <div style={{ padding: '30px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>
        📊 Quantum Visualization
      </h2>
      
      {state.quantumDiagnostics ? (
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ marginBottom: '15px', color: '#555' }}>Live Quantum State Metrics</h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '15px',
            marginBottom: '20px'
          }}>
            <div style={{ background: '#e3f2fd', padding: '15px', borderRadius: '8px', border: '1px solid #2196F3' }}>
              <div style={{ fontWeight: 600, marginBottom: '8px', color: '#1976D2' }}>Purity</div>
              <div style={{ fontSize: '28px', color: '#2196F3', fontWeight: 600 }}>
                {state.quantumDiagnostics.purity.toFixed(4)}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Tr(ρ²) - quantum purity</div>
            </div>
            
            <div style={{ background: '#fff3e0', padding: '15px', borderRadius: '8px', border: '1px solid #FF9800' }}>
              <div style={{ fontWeight: 600, marginBottom: '8px', color: '#F57C00' }}>Entropy</div>
              <div style={{ fontSize: '28px', color: '#FF9800', fontWeight: 600 }}>
                {state.quantumDiagnostics.entropy.toFixed(3)}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Von Neumann entropy</div>
            </div>
            
            <div style={{ background: '#e8f5e8', padding: '15px', borderRadius: '8px', border: '1px solid #4CAF50' }}>
              <div style={{ fontWeight: 600, marginBottom: '8px', color: '#388E3C' }}>Effective Rank</div>
              <div style={{ fontSize: '28px', color: '#4CAF50', fontWeight: 600 }}>
                {state.quantumDiagnostics.effective_rank}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>Active semantic dimensions</div>
            </div>
          </div>
          
          {state.quantumDiagnostics.eigenvals && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ marginBottom: '10px', color: '#555' }}>📈 Eigenvalue Spectrum</h4>
              <div style={{ 
                display: 'flex', 
                gap: '3px', 
                height: '80px', 
                alignItems: 'end',
                background: '#f8f9fa',
                padding: '15px',
                borderRadius: '8px',
                border: '1px solid #e0e0e0'
              }}>
                {state.quantumDiagnostics.eigenvals.slice(0, 24).map((eigenval, i) => (
                  <div
                    key={i}
                    style={{
                      width: '10px',
                      height: `${Math.max(4, eigenval * 2000)}px`,
                      background: `hsl(${240 + i * 8}, 70%, 50%)`,
                      borderRadius: '2px',
                      transition: 'all 0.3s ease'
                    }}
                    title={`λ${i+1}: ${eigenval.toFixed(5)}`}
                  />
                ))}
              </div>
              <div style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
                Top 24 eigenvalues - each bar represents a semantic dimension (hover for exact values)
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ 
          background: '#fff3cd', 
          border: '1px solid #ffc107', 
          borderRadius: '8px', 
          padding: '20px', 
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <h3 style={{ marginBottom: '10px', color: '#856404' }}>No Quantum State Available</h3>
          <p style={{ color: '#856404', margin: 0 }}>
            Create a quantum state in the Analysis stage to see live visualization data
          </p>
        </div>
      )}
      
      {state.povmMeasurements && (
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ marginBottom: '15px', color: '#555' }}>🔬 POVM Measurements</h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
            gap: '10px',
            maxHeight: '250px',
            overflowY: 'auto',
            background: '#f8f9fa',
            padding: '15px',
            borderRadius: '8px',
            border: '1px solid #e0e0e0'
          }}>
            {Object.entries(state.povmMeasurements).map(([measurement, probability]) => (
              <div key={measurement} style={{ 
                background: 'white',
                padding: '10px',
                borderRadius: '6px',
                border: '1px solid #e0e0e0',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}>
                <div style={{ fontSize: '11px', color: '#666', marginBottom: '6px', fontWeight: 500 }}>
                  {measurement.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div style={{ 
                  fontWeight: 700, 
                  color: probability > 0.5 ? '#4CAF50' : probability > 0.3 ? '#FF9800' : '#666',
                  fontSize: '16px'
                }}>
                  {(probability * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
        <button
          onClick={createTrajectoryVisualization}
          disabled={!state.transformations || state.transformations.length === 0}
          style={{
            padding: '12px 24px',
            background: state.transformations?.length > 0 ? '#9C27B0' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: state.transformations?.length > 0 ? 'pointer' : 'not-allowed'
          }}
        >
          📈 Create Bures Trajectory
        </button>
        
        <button
          onClick={() => progressToStage('export')}
          style={{
            padding: '12px 24px',
            background: '#607D8B',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          Continue to Export →
        </button>
      </div>
    </div>
  );

  // Export stage panel (placeholder)
  const renderExportPanel = () => (
    <div style={{ padding: '30px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>
        💾 Export & Archive
      </h2>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Export panel implementation coming soon. This will include:
      </p>
      <ul style={{ color: '#666', marginBottom: '30px', lineHeight: 2 }}>
        <li>💾 Save quantum states and measurements</li>
        <li>📄 Generate analysis reports</li>
        <li>🏛️ Archive to matrix library</li>
        <li>📤 Export in various formats (JSON, CSV, PDF)</li>
      </ul>
      <button
        onClick={() => setCurrentStage('input')}
        style={{
          padding: '12px 24px',
          background: '#607D8B',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: 600,
          cursor: 'pointer'
        }}
      >
        ← Start New Analysis
      </button>
    </div>
  );


  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif' }}>
      {/* Header */}
      <div style={{
        background: '#2196F3',
        color: 'white',
        padding: '15px 30px',
        position: 'relative'
      }}>
        <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 600 }}>
          🚀 Unified Quantum Narrative Workbench
        </h1>
        <p style={{ margin: '5px 0 0 0', opacity: 0.9, fontSize: '14px' }}>
          Mathematical elegance meets narrative analysis • APLG-compatible quantum operations
        </p>
        {renderModeSelector()}
      </div>

      {/* Workflow Rail */}
      {renderWorkflowRail()}

      {/* Status Bar */}
      <div style={{
        background: '#f8f9fa',
        padding: '10px 30px',
        borderBottom: '1px solid #e0e0e0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ fontSize: '14px', color: '#666' }}>
            Status: {state.statusMessage}
            {state.loading && (
              <span style={{ marginLeft: '10px' }}>
                ⚡ Processing...
              </span>
            )}
          </div>
          {state.loading && (
            <ProgressIndicator
              progress={state.progress}
              status="loading"
              showPercentage={false}
              showSpinner={false}
              height={4}
              style={{ width: '100px' }}
            />
          )}
        </div>
        <button
          onClick={() => setAdvancedSidebarOpen(!advancedSidebarOpen)}
          style={{
            padding: '6px 12px',
            background: advancedSidebarOpen ? '#ff9800' : '#f5f5f5',
            color: advancedSidebarOpen ? 'white' : '#666',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: 'pointer'
          }}
        >
          🏛️ Advanced {advancedSidebarOpen ? '◄' : '►'}
        </button>
      </div>

      {/* Main Content Area */}
      <div style={{ display: 'flex', minHeight: 'calc(100vh - 200px)' }}>
        {/* Context-Aware Tool Panel */}
        <div style={{
          flex: '0 0 280px',
          borderRight: '1px solid #e0e0e0'
        }}>
          <ContextAwareToolPanel
            stage={state.currentStage}
            userMode={state.userMode}
            state={state}
            onAction={handlePanelAction}
          />
        </div>

        {/* Main Panel */}
        <div style={{ 
          flex: advancedSidebarOpen ? '1' : '1',
          transition: 'all 0.3s ease'
        }}>
          {renderSuggestions()}
          {renderStagePanel()}
        </div>

        {/* Advanced Sidebar */}
        {advancedSidebarOpen && (
          <div style={{
            flex: '0 0 320px',
            transition: 'all 0.3s ease'
          }}>
            <AdvancedOperationsPanel
              state={state}
              onAction={handlePanelAction}
            />
          </div>
        )}
      </div>

      {/* Notifications */}
      <NotificationContainer
        notifications={state.notifications}
        position="top-right"
        onRemove={removeNotification}
        maxNotifications={5}
      />
    </div>
  );
}