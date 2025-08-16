import React, { useState, useCallback, useEffect } from 'react';

/**
 * Transform Stage Panel - Kai Power Tools Inspired Interface
 * 
 * Elegant, intuitive transformation controls with:
 * - Simple agent requests 
 * - Attribute sliders
 * - Customizable dropdown lists
 * - Saveable attribute presets
 * 
 * Inspired by the elegant simplicity of Kai Power Tools from the 90s
 */
export function TransformStagePanel({ 
  state, 
  onOperation, 
  progressToStage 
}) {
  const [agentRequest, setAgentRequest] = useState('');
  const [activePreset, setActivePreset] = useState('custom');
  const [isResultsExpanded, setIsResultsExpanded] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [showAuditTrail, setShowAuditTrail] = useState(false);
  const [currentText, setCurrentText] = useState(state.narrativeText || '');
  
  // Update currentText when narrativeText changes (from other stages)
  useEffect(() => {
    if (state.narrativeText && state.narrativeText !== currentText) {
      setCurrentText(state.narrativeText);
    }
  }, [state.narrativeText]);
  // Dialectical transformation state - two axes with center as neutral (0,0)
  const [dialecticalPosition, setDialecticalPosition] = useState({
    formalPersonal: 0,    // -1 (Formal) to +1 (Personal)
    directLyrical: 0      // -1 (Direct) to +1 (Lyrical)
  });

  const [transformParams, setTransformParams] = useState({
    style: 'neutral',
    intensity: 0.5,
    creativity: 0.3,
    preservation: 0.8,
    tone: 'balanced',
    complexity: 0.5,
    focus: 'meaning'
  });

  // Predefined transformation presets
  const transformPresets = {
    subtle: {
      name: 'Subtle Refinement',
      icon: '✨',
      style: 'refined',
      intensity: 0.2,
      creativity: 0.1,
      preservation: 0.9,
      tone: 'elegant',
      complexity: 0.4,
      focus: 'clarity'
    },
    creative: {
      name: 'Creative Exploration',
      icon: '🎨',
      style: 'artistic',
      intensity: 0.7,
      creativity: 0.8,
      preservation: 0.6,
      tone: 'expressive',
      complexity: 0.7,
      focus: 'novelty'
    },
    academic: {
      name: 'Academic Polish',
      icon: '📚',
      style: 'formal',
      intensity: 0.4,
      creativity: 0.2,
      preservation: 0.9,
      tone: 'scholarly',
      complexity: 0.6,
      focus: 'precision'
    },
    poetic: {
      name: 'Poetic Flow',
      icon: '🌸',
      style: 'lyrical',
      intensity: 0.6,
      creativity: 0.9,
      preservation: 0.7,
      tone: 'melodic',
      complexity: 0.5,
      focus: 'beauty'
    },
    conversational: {
      name: 'Natural Speech',
      icon: '💭',
      style: 'casual',
      intensity: 0.3,
      creativity: 0.4,
      preservation: 0.8,
      tone: 'friendly',
      complexity: 0.3,
      focus: 'clarity'
    }
  };

  // Convert dialectical position to transformation parameters
  const dialecticalToParams = useCallback((formalPersonal, directLyrical) => {
    // Map dialectical axes to quantum transformation parameters
    
    // Formal-Personal axis affects tone and style
    const personalStrength = (formalPersonal + 1) / 2; // 0 to 1
    const formalStrength = 1 - personalStrength;
    
    // Direct-Lyrical axis affects creativity and complexity  
    const lyricalStrength = (directLyrical + 1) / 2; // 0 to 1
    const directStrength = 1 - lyricalStrength;
    
    return {
      // Style based on primary axis
      style: formalPersonal < -0.3 ? 'formal' : 
             formalPersonal > 0.3 ? 'casual' : 'neutral',
      
      // Tone mapping
      tone: formalPersonal < -0.5 ? 'scholarly' :
            formalPersonal > 0.5 ? 'friendly' :
            directLyrical > 0.5 ? 'melodic' : 'balanced',
            
      // Creativity from lyrical axis
      creativity: Math.max(0.1, lyricalStrength * 0.9),
      
      // Complexity blend of both axes
      complexity: Math.max(0.2, (formalStrength * 0.7) + (lyricalStrength * 0.5)),
      
      // Intensity based on distance from center
      intensity: Math.min(0.8, Math.sqrt(formalPersonal * formalPersonal + directLyrical * directLyrical) * 0.6 + 0.2),
      
      // Preservation - less change when near center
      preservation: Math.max(0.6, 1.0 - Math.sqrt(formalPersonal * formalPersonal + directLyrical * directLyrical) * 0.3),
      
      // Focus area
      focus: directLyrical < -0.3 ? 'clarity' :
             directLyrical > 0.3 ? 'beauty' : 'meaning'
    };
  }, []);

  // Update transform params when dialectical position changes
  useEffect(() => {
    const newParams = dialecticalToParams(dialecticalPosition.formalPersonal, dialecticalPosition.directLyrical);
    setTransformParams(newParams);
  }, [dialecticalPosition, dialecticalToParams]);

  // Available transformation styles
  const styleOptions = [
    { value: 'neutral', label: 'Neutral' },
    { value: 'refined', label: 'Refined' },
    { value: 'artistic', label: 'Artistic' },
    { value: 'formal', label: 'Formal' },
    { value: 'casual', label: 'Casual' },
    { value: 'lyrical', label: 'Lyrical' },
    { value: 'technical', label: 'Technical' }
  ];

  const toneOptions = [
    { value: 'balanced', label: 'Balanced' },
    { value: 'elegant', label: 'Elegant' },
    { value: 'expressive', label: 'Expressive' },
    { value: 'scholarly', label: 'Scholarly' },
    { value: 'friendly', label: 'Friendly' },
    { value: 'melodic', label: 'Melodic' },
    { value: 'authoritative', label: 'Authoritative' }
  ];

  const focusOptions = [
    { value: 'meaning', label: 'Preserve Meaning' },
    { value: 'clarity', label: 'Enhance Clarity' },
    { value: 'novelty', label: 'Increase Novelty' },
    { value: 'precision', label: 'Improve Precision' },
    { value: 'beauty', label: 'Enhance Beauty' },
    { value: 'impact', label: 'Maximize Impact' }
  ];

  // Handle preset selection
  const handlePresetChange = useCallback((presetKey) => {
    if (presetKey === 'custom') {
      setActivePreset('custom');
      return;
    }
    
    const preset = transformPresets[presetKey];
    if (preset) {
      setActivePreset(presetKey);
      setTransformParams({
        style: preset.style,
        intensity: preset.intensity,
        creativity: preset.creativity,
        preservation: preset.preservation,
        tone: preset.tone,
        complexity: preset.complexity,
        focus: preset.focus
      });
    }
  }, []);

  // Handle slider changes
  const handleSliderChange = useCallback((param, value) => {
    setTransformParams(prev => ({
      ...prev,
      [param]: value
    }));
    // Only reset preset when sliders (number values) move, not dropdowns (string values)
    if (typeof value === 'number') {
      setActivePreset('custom');
    }
  }, []);

  // Handle dropdown changes (separate to avoid preset reset)
  const handleDropdownChange = useCallback((param, value) => {
    setTransformParams(prev => ({
      ...prev,
      [param]: value
    }));
    // Don't reset preset for dropdown changes
  }, []);

  // Handle agent request
  const handleAgentRequest = useCallback(async () => {
    if (!agentRequest.trim()) return;
    
    let activeRhoId = state.currentRhoId;
    
    // Auto-create quantum state if needed
    if (!activeRhoId && currentText.trim()) {
      try {
        onOperation('set_loading', { loading: true, message: 'Creating quantum state...' });
        const response = await fetch('http://localhost:8192/rho/init', { method: 'POST' });
        const result = await response.json();
        activeRhoId = result.rho_id;
        onOperation('set_current_rho', { rhoId: activeRhoId });
        
        // Also read the text into the quantum state
        const readResponse = await fetch(`http://localhost:8192/rho/${activeRhoId}/read_channel`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ raw_text: currentText, alpha: 0.3 })
        });
        
        if (!readResponse.ok) {
          throw new Error('Failed to read text into quantum state');
        }
      } catch (error) {
        onOperation('set_loading', { loading: false });
        onOperation('show_notification', {
          message: 'Failed to create quantum state: ' + error.message,
          type: 'error'
        });
        return;
      }
    }
    
    let requestWithLanguage = agentRequest;
    
    // Add language instruction if target language is selected
    if (targetLanguage) {
      const languageMap = {
        'french': 'Translate to French',
        'spanish': 'Translate to Spanish', 
        'german': 'Translate to German',
        'italian': 'Translate to Italian',
        'portuguese': 'Translate to Portuguese',
        'japanese': 'Translate to Japanese',
        'chinese': 'Translate to Chinese',
        'russian': 'Translate to Russian'
      };
      requestWithLanguage = `${agentRequest}. ${languageMap[targetLanguage] || `Translate to ${targetLanguage}`}`;
    }
    
    onOperation('show_notification', {
      message: `Processing request: "${requestWithLanguage}"`,
      type: 'info'
    });

    try {
      // Parse natural language request into transformation parameters
      const request = agentRequest.toLowerCase();
      
      let newParams = { ...transformParams };
      let tone_target = 'balanced';
      
      // Style detection
      if (request.includes('poetic') || request.includes('lyrical') || request.includes('poetry')) {
        newParams.creativity = 0.9;
        tone_target = 'lyrical';
      } else if (request.includes('academic') || request.includes('scholarly') || request.includes('formal')) {
        newParams.complexity = 0.8;
        tone_target = 'scholarly';
      } else if (request.includes('simple') || request.includes('casual') || request.includes('friendly')) {
        newParams.complexity = 0.3;
        tone_target = 'friendly';
      } else if (request.includes('creative') || request.includes('artistic')) {
        newParams.creativity = 0.8;
        tone_target = 'creative';
      }
      
      // Language-specific tone adjustments
      if (targetLanguage) {
        if (['french', 'italian'].includes(targetLanguage)) {
          tone_target = tone_target === 'balanced' ? 'lyrical' : tone_target;
        } else if (['german', 'russian'].includes(targetLanguage)) {
          tone_target = tone_target === 'balanced' ? 'formal' : tone_target;
        }
      }
      
      // Apply transformation with natural language request
      await handleTransformWithRequest(requestWithLanguage, newParams, tone_target, activeRhoId);
      setAgentRequest('');
      
    } catch (error) {
      onOperation('show_notification', {
        message: 'Agent request failed',
        type: 'error'
      });
    }
  }, [agentRequest, targetLanguage, transformParams, onOperation]);

  // Handle transformation with natural language request  
  const handleTransformWithRequest = useCallback(async (requestText, params, toneTarget, rhoId = null) => {
    const currentRhoId = rhoId || state.currentRhoId;
    if (!currentRhoId) {
      onOperation('show_notification', {
        message: 'Please create a quantum state first',
        type: 'warning'
      });
      return;
    }

    try {
      onOperation('set_loading', { loading: true, message: 'Applying quantum transformation...' });
      
      const response = await fetch('http://localhost:8192/transformations/demo-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: currentText,
          transformation_name: `agent_request`,
          strength: params.intensity || 0.6,
          library_name: 'narrative_transformations',
          creativity_level: params.creativity || 0.5,
          preservation_level: params.preservation || 0.8,
          complexity_target: params.complexity || 0.5,
          tone_target: toneTarget,
          focus_area: params.focus || 'meaning',
          agent_request: requestText // Pass the natural language request
        })
      });

      if (!response.ok) {
        throw new Error(`Transformation failed: ${response.status}`);
      }

      const result = await response.json();
      
      onOperation('add_transformation', {
        id: Date.now(),
        type: 'agent_request',
        originalText: currentText,
        transformedText: result.transformed_text,
        quantum_distance: result.quantum_distance,
        bures_distance: result.bures_distance,
        transformation_type: result.transformation_type,
        parameters_used: result.parameters_used,
        timestamp: new Date().toISOString(),
        agent_request: requestText,
        audit_trail: result.audit_trail  // Include detailed audit trail
      });

      onOperation('show_notification', {
        message: 'Quantum transformation applied successfully',
        type: 'success'
      });

    } catch (error) {
      console.error('Transformation error:', error);
      onOperation('show_notification', {
        message: `Transformation failed: ${error.message}`,
        type: 'error'
      });
    } finally {
      onOperation('set_loading', { loading: false });
    }
  }, [state.currentRhoId, currentText, onOperation]);

  // Apply transformation
  const handleTransform = useCallback(async () => {
    let activeRhoId = state.currentRhoId;
    
    // Auto-create quantum state if needed
    if (!activeRhoId && currentText.trim()) {
      try {
        onOperation('set_loading', { loading: true, message: 'Creating quantum state...' });
        const response = await fetch('http://localhost:8192/rho/init', { method: 'POST' });
        const result = await response.json();
        activeRhoId = result.rho_id;
        onOperation('set_current_rho', { rhoId: activeRhoId });
        
        // Also read the text into the quantum state
        const readResponse = await fetch(`http://localhost:8192/rho/${activeRhoId}/read_channel`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ raw_text: currentText, alpha: 0.3 })
        });
        
        if (!readResponse.ok) {
          throw new Error('Failed to read text into quantum state');
        }
      } catch (error) {
        onOperation('set_loading', { loading: false });
        onOperation('show_notification', {
          message: 'Failed to create quantum state: ' + error.message,
          type: 'error'
        });
        return;
      }
    }
    
    if (!activeRhoId && !currentText.trim()) {
      onOperation('show_notification', {
        message: 'Please enter some text to transform',
        type: 'warning'
      });
      return;
    }

    // Use the transformation API to apply quantum transformation
    try {
      onOperation('set_loading', { loading: true, message: 'Applying quantum transformation...' });
      
      const response = await fetch('http://localhost:8192/transformations/demo-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: currentText,
          transformation_name: `${transformParams.style}_transformation`,
          strength: transformParams.intensity,
          library_name: 'narrative_transformations',
          creativity_level: transformParams.creativity,
          preservation_level: transformParams.preservation,
          complexity_target: transformParams.complexity,
          tone_target: transformParams.tone,
          focus_area: transformParams.focus
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        console.log('🔍 TRANSFORMATION DEBUG:');
        console.log('Original text:', currentText);
        console.log('Transformed text:', result.transformed_text);
        console.log('Are they the same?', currentText === result.transformed_text);
        console.log('Transformation type:', result.transformation_type);
        console.log('Quantum distance:', result.quantum_distance);
        
        onOperation('add_transformation', {
          id: Date.now(),
          type: activePreset !== 'custom' ? activePreset : 'custom',
          originalText: currentText,
          transformedText: result.transformed_text,
          params: transformParams,
          timestamp: new Date().toISOString(),
          quantum_distance: result.quantum_distance || null,
          bures_distance: result.bures_distance || null
        });

        onOperation('show_notification', {
          message: 'Transformation applied successfully!',
          type: 'success'
        });
      } else {
        const errorText = await response.text();
        throw new Error(`Transformation failed: ${response.status} - ${errorText}`);
      }
    } catch (error) {
      console.error('Transformation error:', error);
      onOperation('show_notification', {
        message: `Transformation failed: ${error.message}`,
        type: 'error'
      });
      
      // NO FALLBACKS - Let it fail so we can see the real problem
      throw error;
    } finally {
      onOperation('set_loading', { loading: false });
    }
  }, [state.currentRhoId, currentText, transformParams, activePreset, onOperation]);

  // NO DEMO TRANSFORMATIONS - Removed generateDemoTransformation

  // Side-by-side text comparison component
  const TextDiff = ({ originalText, transformedText, isExpanded }) => {

    const textStyle = {
      fontSize: '16px',
      lineHeight: 1.7,
      fontFamily: '"Georgia", "Times New Roman", serif',
      whiteSpace: 'pre-wrap',
      wordWrap: 'break-word',
      hyphens: 'auto'
    };

    const containerHeight = isExpanded ? '400px' : '200px';
    
    return (
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '20px',
        marginTop: '20px'
      }}>
        {/* Original Text Column */}
        <div style={{
          background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)',
          border: '2px solid #e0e0e0',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #9e9e9e 0%, #757575 100%)',
            color: 'white',
            padding: '12px 16px',
            fontWeight: '600',
            fontSize: '13px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            📄 Original Text
            <span style={{
              fontSize: '11px',
              background: 'rgba(255,255,255,0.2)',
              padding: '2px 6px',
              borderRadius: '8px'
            }}>
              {originalText.split(' ').length} words
            </span>
          </div>
          <div style={{
            padding: '20px',
            height: containerHeight,
            overflowY: 'auto',
            color: '#424242',
            ...textStyle
          }}>
            {originalText}
          </div>
        </div>
        
        {/* Transformed Text Column */}
        <div style={{
          background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
          border: '2px solid #03a9f4',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 4px 12px rgba(3, 169, 244, 0.15)'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #03a9f4 0%, #0288d1 100%)',
            color: 'white',
            padding: '12px 16px',
            fontWeight: '600',
            fontSize: '13px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            ✨ Transformed Text
            <span style={{
              fontSize: '11px',
              background: 'rgba(255,255,255,0.2)',
              padding: '2px 6px',
              borderRadius: '8px'
            }}>
              {transformedText.split(' ').length} words
            </span>
            {state.transformations && state.transformations.length > 0 && (() => {
              const latest = state.transformations[state.transformations.length - 1];
              return latest.quantum_distance && (
                <span style={{
                  fontSize: '10px',
                  background: 'rgba(255,255,255,0.3)',
                  color: 'white',
                  padding: '2px 6px',
                  borderRadius: '8px',
                  fontWeight: '700'
                }}>
                  Δ {latest.quantum_distance.toFixed(3)}
                </span>
              );
            })()}
          </div>
          <div style={{
            padding: '20px',
            height: containerHeight,
            overflowY: 'auto',
            color: '#01579b',
            ...textStyle
          }}>
            {transformedText}
          </div>
        </div>
      </div>
    );
  };

  // Dialectical Transformation Interface - 2D space between four poles
  const DialecticalInterface = () => {
    const size = 280;
    const center = size / 2;
    const maxRadius = center - 40;
    
    const handlePositionChange = useCallback((event) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left - center;
      const y = event.clientY - rect.top - center;
      
      // Constrain to circle
      const distance = Math.sqrt(x * x + y * y);
      const constrainedDistance = Math.min(distance, maxRadius);
      const angle = Math.atan2(y, x);
      
      const constrainedX = constrainedDistance * Math.cos(angle);
      const constrainedY = constrainedDistance * Math.sin(angle);
      
      // Convert to dialectical coordinates (-1 to 1)
      const formalPersonal = constrainedX / maxRadius;
      const directLyrical = -constrainedY / maxRadius; // Invert Y for intuitive up/down
      
      setDialecticalPosition({ formalPersonal, directLyrical });
    }, [maxRadius, center]);

    const currentX = dialecticalPosition.formalPersonal * maxRadius + center;
    const currentY = -dialecticalPosition.directLyrical * maxRadius + center;

    return (
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '25px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        border: '1px solid #e9ecef'
      }}>
        <h3 style={{ 
          margin: '0 0 20px 0',
          color: '#495057',
          fontSize: '16px',
          fontWeight: '600',
          textAlign: 'center'
        }}>
          🎭 Narrative Transformation Space
        </h3>
        
        <div style={{ position: 'relative', margin: '0 auto', width: size, height: size }}>
          {/* Background circle */}
          <svg 
            width={size} 
            height={size} 
            style={{ position: 'absolute', top: 0, left: 0 }}
            onClick={handlePositionChange}
          >
            {/* Background gradient definition */}
            <defs>
              <radialGradient id="dialecticalBg" cx="50%" cy="50%" r="50%">
                <stop offset="0%" style={{ stopColor: '#f8f9fa', stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: '#e9ecef', stopOpacity: 1 }} />
              </radialGradient>
            </defs>
            
            {/* Outer circle */}
            <circle
              cx={center}
              cy={center}
              r={maxRadius}
              fill="url(#dialecticalBg)"
              stroke="#dee2e6"
              strokeWidth="2"
            />
            
            {/* Axis lines */}
            <line x1={40} y1={center} x2={size-40} y2={center} stroke="#bbb" strokeWidth="1" strokeDasharray="3,3" />
            <line x1={center} y1={40} x2={center} y2={size-40} stroke="#bbb" strokeWidth="1" strokeDasharray="3,3" />
            
            {/* Current position */}
            <circle
              cx={currentX}
              cy={currentY}
              r="8"
              fill="#667eea"
              stroke="white"
              strokeWidth="3"
              style={{ cursor: 'pointer' }}
            />
          </svg>
          
          {/* Axis labels */}
          <div style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', fontSize: '12px', fontWeight: '600', color: '#666' }}>
            Formal
          </div>
          <div style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', fontSize: '12px', fontWeight: '600', color: '#666' }}>
            Personal
          </div>
          <div style={{ position: 'absolute', top: '10px', left: '50%', transform: 'translateX(-50%)', fontSize: '12px', fontWeight: '600', color: '#666' }}>
            Lyrical
          </div>
          <div style={{ position: 'absolute', bottom: '10px', left: '50%', transform: 'translateX(-50%)', fontSize: '12px', fontWeight: '600', color: '#666' }}>
            Direct
          </div>
          
          {/* Center indicator */}
          <div style={{ 
            position: 'absolute', 
            left: '50%', 
            top: '50%', 
            transform: 'translate(-50%, -50%)',
            fontSize: '10px',
            color: '#999',
            pointerEvents: 'none'
          }}>
            ⊕
          </div>
        </div>
        
        {/* Current position readout */}
        <div style={{ marginTop: '15px', textAlign: 'center', fontSize: '12px', color: '#666' }}>
          <div>
            <strong>Style:</strong> {transformParams.style} | <strong>Tone:</strong> {transformParams.tone}
          </div>
          <div style={{ marginTop: '5px' }}>
            <strong>Intensity:</strong> {(transformParams.intensity * 100).toFixed(0)}% | 
            <strong> Creativity:</strong> {(transformParams.creativity * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ 
      padding: '15px',
      background: 'linear-gradient(135deg, #fafbfc 0%, #f8f9fa 100%)',
      minHeight: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Compact Control Bar */}
      <div style={{
        background: 'white',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '15px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        border: '1px solid #e9ecef',
        display: 'flex',
        gap: '12px',
        alignItems: 'center',
        flexWrap: 'wrap'
      }}>
        {/* Smart Request Input */}
        <div style={{ flex: '1 1 300px', display: 'flex', gap: '8px' }}>
          <input
            type="text"
            value={agentRequest}
            onChange={(e) => setAgentRequest(e.target.value)}
            placeholder="Transform: 'make poetic', 'academic style', 'translate to French'..."
            style={{
              flex: 1,
              padding: '8px 12px',
              border: '1px solid #dee2e6',
              borderRadius: '6px',
              fontSize: '13px',
              outline: 'none'
            }}
            onKeyPress={(e) => e.key === 'Enter' && handleAgentRequest()}
          />
          <button
            onClick={handleAgentRequest}
            disabled={!agentRequest.trim()}
            style={{
              padding: '8px 16px',
              background: agentRequest.trim() ? '#667eea' : '#e9ecef',
              color: agentRequest.trim() ? 'white' : '#6c757d',
              border: 'none',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: '600',
              cursor: agentRequest.trim() ? 'pointer' : 'not-allowed',
              whiteSpace: 'nowrap'
            }}
          >
            ⚡ Apply
          </button>
        </div>

        {/* Language Selection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '12px', color: '#6c757d', fontWeight: '600' }}>🌐 Language:</span>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            style={{
              padding: '6px 8px',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              fontSize: '12px',
              background: 'white'
            }}
          >
            <option value="">Same language</option>
            <option value="french">🇫🇷 French</option>
            <option value="spanish">🇪🇸 Spanish</option>
            <option value="german">🇩🇪 German</option>
            <option value="italian">🇮🇹 Italian</option>
            <option value="portuguese">🇵🇹 Portuguese</option>
            <option value="japanese">🇯🇵 Japanese</option>
            <option value="chinese">🇨🇳 Chinese</option>
            <option value="russian">🇷🇺 Russian</option>
          </select>
        </div>

        {/* Quick Style Buttons */}
        <div style={{ display: 'flex', gap: '4px' }}>
          {['🎭 Poetic', '📚 Academic', '💬 Casual', '🎨 Creative'].map((style) => (
            <button
              key={style}
              onClick={() => {
                const styleMap = {
                  '🎭 Poetic': 'poetic_flow',
                  '📚 Academic': 'academic_polish', 
                  '💬 Casual': 'natural_speech',
                  '🎨 Creative': 'creative_exploration'
                };
                handlePresetChange(styleMap[style]);
              }}
              style={{
                padding: '6px 10px',
                background: '#f8f9fa',
                border: '1px solid #dee2e6',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
                whiteSpace: 'nowrap'
              }}
            >
              {style}
            </button>
          ))}
        </div>
      </div>

      {/* Text Input Area */}
      <div style={{
        background: 'white',
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '15px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        border: '1px solid #e9ecef'
      }}>
        <label style={{
          display: 'block',
          marginBottom: '8px',
          color: '#495057',
          fontSize: '14px',
          fontWeight: '600'
        }}>
          📝 Text to Transform
        </label>
        <textarea
          value={currentText}
          onChange={(e) => setCurrentText(e.target.value)}
          placeholder="Enter or edit your text here..."
          style={{
            width: '100%',
            minHeight: '80px',
            padding: '12px',
            border: '1px solid #dee2e6',
            borderRadius: '6px',
            fontSize: '16px',
            lineHeight: '1.5',
            fontFamily: 'system-ui, sans-serif',
            resize: 'vertical',
            outline: 'none',
            transition: 'border-color 0.2s ease'
          }}
          onFocus={(e) => e.target.style.borderColor = '#17a2b8'}
          onBlur={(e) => e.target.style.borderColor = '#dee2e6'}
        />
        <div style={{
          marginTop: '8px',
          fontSize: '12px',
          color: '#6c757d',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span>{currentText.split(' ').filter(w => w.length > 0).length} words</span>
          <span>Edit freely • Transform shows on the right</span>
        </div>
      </div>

      {/* Text Comparison - Takes up most of the screen */}
      {state.transformations && state.transformations.length > 0 ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {/* Latest Transformation */}
          {(() => {
            const latest = state.transformations[state.transformations.length - 1];
            return (
              <div style={{ marginBottom: '25px' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '15px'
                }}>
                  <h4 style={{ 
                    margin: 0,
                    color: '#495057',
                    fontSize: '14px',
                    fontWeight: '600'
                  }}>
                    Latest: {(latest.type || 'transformation').replace(/_/g, ' ')} ({new Date(latest.timestamp).toLocaleTimeString()})
                  </h4>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      onClick={() => setIsResultsExpanded(!isResultsExpanded)}
                      style={{
                        padding: '6px 12px',
                        background: isResultsExpanded ? '#FF9800' : '#f8f9fa',
                        color: isResultsExpanded ? 'white' : '#495057',
                        border: '1px solid #dee2e6',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}
                    >
                      {isResultsExpanded ? '📋 Collapse' : '📖 Expand'}
                    </button>
                    <button
                      onClick={() => {
                        // Transform the transformed text (chaining)
                        onOperation('update_narrative_text', { text: latest.transformedText });
                        onOperation('show_notification', { 
                          message: 'Transformed text loaded for new transformation', 
                          type: 'success' 
                        });
                        // Focus and highlight the input
                        setTimeout(() => {
                          const input = document.querySelector('input[placeholder*="Transform"]');
                          if (input) {
                            input.focus();
                            input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            input.style.boxShadow = '0 0 10px rgba(102, 126, 234, 0.5)';
                            setTimeout(() => { input.style.boxShadow = ''; }, 2000);
                          }
                        }, 100);
                      }}
                      style={{
                        padding: '6px 12px',
                        background: 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}
                    >
                      🔄 Transform This
                    </button>
                    <button
                      onClick={() => {
                        // Actually trigger a new transformation
                        if (agentRequest.trim()) {
                          // If there's an agent request, use that
                          handleAgentRequest();
                        } else {
                          // Otherwise do a basic transformation with current preset
                          handleTransform();
                        }
                      }}
                      style={{
                        padding: '6px 12px',
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}
                    >
                      ⚡ New Transform
                    </button>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(latest.transformedText);
                        onOperation('show_notification', { message: 'Copied to clipboard!', type: 'success' });
                      }}
                      style={{
                        padding: '6px 12px',
                        background: '#6c757d',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}
                    >
                      📋 Copy
                    </button>
                  </div>
                </div>
                
                {/* Expand/Collapse Toggle */}
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  marginBottom: '15px' 
                }}>
                  <button
                    onClick={() => setIsResultsExpanded(!isResultsExpanded)}
                    style={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      color: 'white',
                      border: 'none',
                      padding: '8px 16px',
                      borderRadius: '20px',
                      fontSize: '12px',
                      fontWeight: '600',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)'
                    }}
                  >
                    {isResultsExpanded ? '🔼 Collapse' : '🔽 Expand'} Comparison
                  </button>
                </div>
                
                {/* Side-by-side text comparison */}
                <TextDiff 
                  originalText={latest.originalText}
                  transformedText={latest.transformedText}
                  isExpanded={isResultsExpanded}
                />
                
                {/* Audit Trail Viewer */}
                {latest.audit_trail && (
                  <div style={{ marginTop: '20px' }}>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      marginBottom: '15px' 
                    }}>
                      <button
                        onClick={() => setShowAuditTrail(!showAuditTrail)}
                        style={{
                          background: 'linear-gradient(135deg, #17a2b8 0%, #138496 100%)',
                          color: 'white',
                          border: 'none',
                          padding: '8px 16px',
                          borderRadius: '20px',
                          fontSize: '12px',
                          fontWeight: '600',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          boxShadow: '0 2px 8px rgba(23, 162, 184, 0.3)'
                        }}
                      >
                        {showAuditTrail ? '🔼 Hide' : '🔍 Show'} Transformation Log
                      </button>
                    </div>
                    
                    {showAuditTrail && (
                      <div style={{
                        background: '#f8f9fa',
                        border: '1px solid #dee2e6',
                        borderRadius: '8px',
                        padding: '16px',
                        fontSize: '12px',
                        fontFamily: 'Monaco, Consolas, monospace'
                      }}>
                        <h4 style={{ 
                          margin: '0 0 12px 0', 
                          color: '#495057',
                          fontSize: '14px',
                          fontWeight: '600'
                        }}>
                          🔬 Quantum Transformation Audit Trail
                        </h4>
                        
                        {/* Performance Summary */}
                        {latest.audit_trail.performance && (
                          <div style={{
                            background: '#e9ecef',
                            padding: '8px',
                            borderRadius: '4px',
                            marginBottom: '12px'
                          }}>
                            <strong>⏱️ Performance:</strong> {(latest.audit_trail.performance.total_duration || 0).toFixed(3)}s total
                            {latest.audit_trail.performance.quantum_operations_time && (
                              <span> | Quantum: {latest.audit_trail.performance.quantum_operations_time.toFixed(3)}s</span>
                            )}
                            {latest.audit_trail.performance.llm_time && (
                              <span> | LLM: {latest.audit_trail.performance.llm_time.toFixed(3)}s</span>
                            )}
                          </div>
                        )}
                        
                        {/* Quantum Steps */}
                        {latest.audit_trail.quantum_steps && latest.audit_trail.quantum_steps.length > 0 && (
                          <div style={{ marginBottom: '12px' }}>
                            <strong>🔬 Quantum Operations:</strong>
                            {latest.audit_trail.quantum_steps.map((step, index) => (
                              <div key={index} style={{
                                background: step.success === false ? '#ffebee' : '#e8f5e9',
                                padding: '6px 8px',
                                margin: '4px 0',
                                borderRadius: '3px',
                                borderLeft: `3px solid ${step.success === false ? '#f44336' : '#4caf50'}`
                              }}>
                                <div><strong>Step {step.step}:</strong> {step.name}</div>
                                <div style={{ color: '#666', fontSize: '11px' }}>{step.description}</div>
                                {step.duration && <div style={{ color: '#666', fontSize: '11px' }}>Duration: {step.duration.toFixed(3)}s</div>}
                                {step.error && <div style={{ color: '#d32f2f', fontSize: '11px' }}>Error: {step.error}</div>}
                                {step.original_properties && (
                                  <div style={{ color: '#666', fontSize: '11px' }}>
                                    Trace: {step.original_properties.trace?.toFixed(6)} | 
                                    Purity: {step.original_properties.purity?.toFixed(6)} | 
                                    Rank: {step.original_properties.effective_rank}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        
                        {/* LLM Interaction */}
                        {latest.audit_trail.llm_interaction && (
                          <div style={{ marginBottom: '12px' }}>
                            <strong>🤖 LLM Interaction:</strong>
                            <div style={{
                              background: latest.audit_trail.llm_interaction.success ? '#e8f5e9' : '#ffebee',
                              padding: '6px 8px',
                              margin: '4px 0',
                              borderRadius: '3px',
                              borderLeft: `3px solid ${latest.audit_trail.llm_interaction.success ? '#4caf50' : '#f44336'}`
                            }}>
                              <div><strong>Provider:</strong> {latest.audit_trail.llm_interaction.provider || 'unknown'}</div>
                              <div><strong>Model:</strong> {latest.audit_trail.llm_interaction.model || 'unknown'}</div>
                              {latest.audit_trail.llm_interaction.duration && (
                                <div><strong>Duration:</strong> {latest.audit_trail.llm_interaction.duration.toFixed(3)}s</div>
                              )}
                              {latest.audit_trail.llm_interaction.raw_response && (
                                <div style={{ marginTop: '4px' }}>
                                  <div><strong>Raw Response:</strong></div>
                                  <div style={{ 
                                    background: 'white', 
                                    padding: '4px', 
                                    borderRadius: '2px',
                                    fontSize: '11px',
                                    maxHeight: '100px',
                                    overflow: 'auto'
                                  }}>
                                    {latest.audit_trail.llm_interaction.raw_response}
                                  </div>
                                </div>
                              )}
                              {latest.audit_trail.llm_interaction.cleaning_applied && (
                                <div style={{ color: '#ef6c00', fontSize: '11px' }}>
                                  ⚠️ Text cleaning applied to remove meta-commentary
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {/* Errors */}
                        {latest.audit_trail.errors && latest.audit_trail.errors.length > 0 && (
                          <div style={{ marginBottom: '12px' }}>
                            <strong>❌ Errors:</strong>
                            {latest.audit_trail.errors.map((error, index) => (
                              <div key={index} style={{
                                background: '#ffebee',
                                padding: '4px 8px',
                                margin: '2px 0',
                                borderRadius: '3px',
                                color: '#d32f2f',
                                fontSize: '11px'
                              }}>
                                {error}
                              </div>
                            ))}
                          </div>
                        )}
                        
                        {/* Raw Audit Trail (collapsible) */}
                        <details style={{ marginTop: '12px' }}>
                          <summary style={{ 
                            cursor: 'pointer', 
                            fontWeight: '600',
                            color: '#495057'
                          }}>
                            📋 Raw Audit Data
                          </summary>
                          <pre style={{
                            background: 'white',
                            padding: '8px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            overflow: 'auto',
                            maxHeight: '300px',
                            marginTop: '8px'
                          }}>
                            {JSON.stringify(latest.audit_trail, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })()}
          
          {/* Transformation History */}
          {state.transformations.length > 1 && (
            <div>
              <h4 style={{ 
                margin: '0 0 15px 0',
                color: '#495057',
                fontSize: '14px',
                fontWeight: '600'
              }}>
                📚 Transformation History
              </h4>
              <div style={{ display: 'grid', gap: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                {state.transformations.slice(0, -1).reverse().map((transformation, index) => (
                  <div key={transformation.id} style={{
                    padding: '12px 15px',
                    background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    border: '1px solid #dee2e6',
                    borderRadius: '6px',
                    fontSize: '12px',
                    display: 'grid',
                    gridTemplateColumns: '1fr auto auto',
                    gap: '10px',
                    alignItems: 'center'
                  }}>
                    <div>
                      <div style={{ fontWeight: '600', color: '#495057', marginBottom: '2px' }}>
                        {transformation.type.replace(/_/g, ' ')}
                      </div>
                      <div style={{ color: '#6c757d', fontSize: '10px' }}>
                        {new Date(transformation.timestamp).toLocaleString()}
                      </div>
                    </div>
                    <div style={{ fontSize: '10px', color: '#6c757d' }}>
                      {transformation.transformedText.substring(0, 40)}...
                    </div>
                    <div style={{ display: 'flex', gap: '4px' }}>
                      <button
                        onClick={() => {
                          onOperation('update_narrative_text', { text: transformation.transformedText });
                        }}
                        style={{
                          padding: '4px 8px',
                          background: '#6c757d',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          fontSize: '10px',
                          cursor: 'pointer'
                        }}
                      >
                        Use
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ 
          flex: 1, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          background: 'white',
          borderRadius: '12px',
          border: '2px dashed #dee2e6'
        }}>
          <div style={{ 
            textAlign: 'center', 
            color: '#6c757d',
            padding: '40px'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚗️</div>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '18px' }}>Ready for Transformation</h3>
            <p style={{ margin: 0, fontSize: '14px' }}>
              Enter a transformation request above to see side-by-side comparison
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default TransformStagePanel;