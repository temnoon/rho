import React, { useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../utils/api.js';

/**
 * Narrative Distillation Studio
 * 
 * A focused, elegant interface for extracting the quantum essence of narratives.
 * Provides precise controls for POVM-based separation of namespace, persona, 
 * and style into pure rho-embeddings.
 * 
 * Workflow:
 * 1. Input narrative text
 * 2. Select POVM measurement strategy
 * 3. Configure distillation parameters
 * 4. Apply measurements and extract essence
 * 5. Refine and export the distilled rho-embedding
 */
export function NarrativeDistillationStudio() {
  // Core state
  const [narrativeText, setNarrativeText] = useState('');
  const [currentRhoId, setCurrentRhoId] = useState(null);
  const [distillationStage, setDistillationStage] = useState('input'); // input, configure, distill, refine, export
  
  // Configuration state
  const [povmStrategy, setPovmStrategy] = useState('comprehensive');
  const [channelType, setChannelType] = useState('rank_one_update');
  const [readingAlpha, setReadingAlpha] = useState(0.3);
  const [measurementDepth, setMeasurementDepth] = useState('standard');
  
  // Results state
  const [extractedAttributes, setExtractedAttributes] = useState(null);
  const [namespaceEssence, setNamespaceEssence] = useState(null);
  const [personaEssence, setPersonaEssence] = useState(null);
  const [styleEssence, setStyleEssence] = useState(null);
  const [finalRhoEmbedding, setFinalRhoEmbedding] = useState(null);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [availablePacks, setAvailablePacks] = useState([]);

  // Distillation strategies
  const distillationStrategies = {
    comprehensive: {
      name: 'Comprehensive Analysis',
      description: 'Full spectrum POVM analysis across all narrative dimensions',
      measurements: ['persona', 'style', 'perspective', 'genre', 'reliability'],
      icon: '🔬'
    },
    focused: {
      name: 'Focused Extraction',
      description: 'Targeted measurement of specific narrative attributes',
      measurements: ['persona', 'style'],
      icon: '🎯'
    },
    essential: {
      name: 'Essential Distillation',
      description: 'Minimal measurements for core narrative essence',
      measurements: ['persona'],
      icon: '💎'
    },
    experimental: {
      name: 'Experimental Protocol',
      description: 'Advanced measurements with custom POVM combinations',
      measurements: ['all_available'],
      icon: '⚗️'
    }
  };

  // Channel configurations
  const channelConfigs = {
    rank_one_update: {
      name: 'Standard Reading',
      description: 'Balanced absorption with CPTP guarantees',
      icon: '📖',
      recommended: true
    },
    coherent_rotation: {
      name: 'Perspective Shift',
      description: 'Entropy-preserving narrative transformation',
      icon: '🔄',
      advanced: true
    },
    dephasing_mixture: {
      name: 'Ambiguous Integration',
      description: 'Multiple interpretation superposition',
      icon: '🌀',
      experimental: true
    }
  };

  // Load available POVM packs
  const loadAvailablePacks = useCallback(async () => {
    try {
      const response = await fetch(apiUrl('/packs'));
      if (response.ok) {
        const data = await response.json();
        console.log('Loaded POVM packs:', data);
        setAvailablePacks(data.packs || []);
      }
    } catch (error) {
      console.error('Failed to load POVM packs:', error);
    }
  }, []);

  // Initialize quantum state for distillation
  const initializeQuantumState = useCallback(async () => {
    if (!narrativeText.trim()) return null;

    setLoading(true);
    setProgress(10);
    setStatusMessage('Initializing quantum consciousness state...');

    try {
      const response = await fetch(apiUrl('/rho/init'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seed_text: narrativeText.substring(0, 200), // Use opening as seed
          label: `Distillation_${Date.now()}`
        })
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentRhoId(data.rho_id);
        setProgress(25);
        setStatusMessage('Quantum state initialized successfully');
        return data.rho_id;
      } else {
        throw new Error('Failed to initialize quantum state');
      }
    } catch (error) {
      console.error('Quantum state initialization failed:', error);
      setStatusMessage(`Error: ${error.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [narrativeText]);

  // Read narrative into quantum state
  const readNarrativeIntoState = useCallback(async (rhoId) => {
    if (!rhoId || !narrativeText.trim()) return false;

    setProgress(30);
    setStatusMessage('Reading narrative into quantum consciousness...');

    try {
      // Split narrative into manageable chunks
      const chunks = narrativeText.match(/.{1,500}/g) || [narrativeText];
      
      // Filter out empty chunks first
      const validChunks = chunks.filter(chunk => chunk.trim().length > 0);
      console.log(`Filtered ${chunks.length} chunks down to ${validChunks.length} valid chunks`);
      
      for (let i = 0; i < validChunks.length; i++) {
        const chunk = validChunks[i];
        console.log(`Processing chunk ${i + 1}/${validChunks.length}: length=${chunk.length}, content="${chunk.substring(0, 50)}..."`);
        setProgress(30 + (i / validChunks.length) * 30);
        setStatusMessage(`Processing chunk ${i + 1} of ${validChunks.length}...`);

        const response = await fetch(apiUrl(`/rho/${rhoId}/read_channel?channel_type=${channelType}`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            raw_text: chunk,
            alpha: readingAlpha
          })
        });

        if (!response.ok) {
          let errorMessage = `HTTP ${response.status}`;
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
          } catch (e) {
            // If JSON parsing fails, use status text
            errorMessage = response.statusText || errorMessage;
          }
          throw new Error(`Failed to read chunk ${i + 1}: ${errorMessage}`);
        }

        // Small delay to prevent overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      setProgress(60);
      setStatusMessage(`Narrative successfully integrated into quantum state (${validChunks.length} chunks processed)`);
      return true;
    } catch (error) {
      console.error('Narrative reading failed:', error);
      setStatusMessage(`Error reading narrative: ${error.message}`);
      return false;
    }
  }, [narrativeText, readingAlpha, channelType]);

  // Apply POVM measurements for attribute extraction
  const applyPovmMeasurements = useCallback(async (rhoId) => {
    if (!rhoId) return false;

    setProgress(65);
    setStatusMessage('Applying quantum measurements for attribute extraction...');

    try {
      const strategy = distillationStrategies[povmStrategy];
      const measurements = {};
      
      // Apply measurements for each category
      for (const category of strategy.measurements) {
        if (category === 'all_available') {
          // Use all available packs for experimental mode
          for (const pack of availablePacks) {
            if (!pack || !pack.pack_id) {
              console.warn('Skipping invalid pack:', pack);
              continue;
            }
            setStatusMessage(`Measuring ${pack.pack_id}...`);
            const response = await fetch(apiUrl(`/packs/measure/${rhoId}`), {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ pack_id: pack.pack_id })
            });
            
            if (response.ok) {
              const data = await response.json();
              measurements[pack.pack_id] = data;
            } else {
              console.error(`Failed to measure pack ${pack.pack_id}:`, response.status, response.statusText);
            }
          }
        } else {
          // Use specific measurement category
          setStatusMessage(`Measuring ${category} attributes...`);
          
          // Find appropriate pack for this category
          const pack = availablePacks.find(p => 
            p && p.pack_id && p.description &&
            (p.pack_id.toLowerCase().includes(category) || 
             p.description.toLowerCase().includes(category))
          );
          
          if (pack) {
            const response = await fetch(apiUrl(`/packs/measure/${rhoId}`), {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ pack_id: pack.pack_id })
            });
            
            if (response.ok) {
              const data = await response.json();
              measurements[category] = data;
            } else {
              console.error(`Failed to measure category ${category}:`, response.status, response.statusText);
            }
          }
        }
        
        setProgress(65 + (Object.keys(measurements).length / strategy.measurements.length) * 15);
      }

      setExtractedAttributes(measurements);
      setProgress(80);
      setStatusMessage('Quantum measurements completed successfully');
      return true;
    } catch (error) {
      console.error('POVM measurement failed:', error);
      setStatusMessage(`Error in measurements: ${error.message}`);
      return false;
    }
  }, [povmStrategy, availablePacks]);

  // Distill essence components
  const distillEssenceComponents = useCallback(async (rhoId, attributes) => {
    if (!rhoId || !attributes) return false;

    setProgress(85);
    setStatusMessage('Distilling narrative essence components...');

    try {
      // Extract namespace (semantic domain)
      const namespaceData = await extractNamespaceEssence(attributes);
      setNamespaceEssence(namespaceData);
      setProgress(87);
      
      // Extract persona (character/voice)
      const personaData = await extractPersonaEssence(attributes);
      setPersonaEssence(personaData);
      setProgress(90);
      
      // Extract style (linguistic patterns)
      const styleData = await extractStyleEssence(attributes);
      setStyleEssence(styleData);
      setProgress(93);

      setStatusMessage('Essence components successfully distilled');
      return true;
    } catch (error) {
      console.error('Essence distillation failed:', error);
      setStatusMessage(`Error distilling essence: ${error.message}`);
      return false;
    }
  }, []);

  // Create final rho-embedding
  const createFinalRhoEmbedding = useCallback(async (rhoId) => {
    if (!rhoId) return false;

    setProgress(98);
    setStatusMessage('Creating final rho-embedding...');

    try {
      // Get final quantum state
      const response = await fetch(apiUrl(`/rho/${rhoId}`));
      if (response.ok) {
        const data = await response.json();
        
        const embedding = {
          rho_id: rhoId,
          matrix: data.matrix,
          diagnostics: data.diagnostics,
          essence_components: {
            namespace: namespaceEssence,
            persona: personaEssence,
            style: styleEssence
          },
          distillation_metadata: {
            strategy: povmStrategy,
            channel_type: channelType,
            reading_alpha: readingAlpha,
            original_text_length: narrativeText.length,
            created_at: new Date().toISOString()
          }
        };

        setFinalRhoEmbedding(embedding);
        setProgress(100);
        setStatusMessage('Rho-embedding complete! Ready for export.');
        return true;
      }
    } catch (error) {
      console.error('Final embedding creation failed:', error);
      setStatusMessage(`Error creating final embedding: ${error.message}`);
      return false;
    }
  }, [namespaceEssence, personaEssence, styleEssence, povmStrategy, channelType, readingAlpha, narrativeText]);

  // Main distillation workflow
  const executeDistillationWorkflow = useCallback(async () => {
    if (!narrativeText.trim()) return;

    setDistillationStage('distill');
    setLoading(true);
    setProgress(0);

    try {
      // Step 1: Initialize quantum state
      const rhoId = await initializeQuantumState();
      if (!rhoId) return;

      // Step 2: Read narrative
      const readSuccess = await readNarrativeIntoState(rhoId);
      if (!readSuccess) return;

      // Step 3: Apply measurements
      const measureSuccess = await applyPovmMeasurements(rhoId);
      if (!measureSuccess) return;

      // Step 4: Distill essence
      const distillSuccess = await distillEssenceComponents(rhoId, extractedAttributes);
      if (!distillSuccess) return;

      // Step 5: Create final embedding
      const embeddingSuccess = await createFinalRhoEmbedding(rhoId);
      if (!embeddingSuccess) return;

      setDistillationStage('export');
    } catch (error) {
      console.error('Distillation workflow failed:', error);
      setStatusMessage(`Workflow error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }, [
    narrativeText, initializeQuantumState, readNarrativeIntoState, 
    applyPovmMeasurements, distillEssenceComponents, createFinalRhoEmbedding, extractedAttributes
  ]);

  // Helper functions for essence extraction
  const extractNamespaceEssence = async (attributes) => {
    // Extract semantic domain information from measurements
    const namespace = {
      genre_classification: {
        narrative_score: attributes.advanced_narrative_pack?.measurements?.narrative_concerns_narrative || 0,
        genre_conformity: attributes.advanced_narrative_pack?.measurements?.genre_conformity_conventional || 0
      },
      thematic_elements: {
        abstract_level: attributes.advanced_narrative_pack?.measurements?.abstract_information_abstract || 0,
        cultural_specificity: attributes.advanced_narrative_pack?.measurements?.cultural_specificity_culture_specific || 0
      },
      semantic_domain: attributes.advanced_narrative_pack?.measurements?.field_register_specialized > 0.5 ? 'specialized' : 'general',
      conceptual_density: attributes.advanced_narrative_pack?.measurements?.cognitive_load_complex || 0
    };
    return namespace;
  };

  const extractPersonaEssence = async (attributes) => {
    // Extract character/voice information from measurements
    const persona = {
      narrative_voice: {
        formality: attributes.advanced_narrative_pack?.measurements?.tenor_formality_formal || 0,
        affect: attributes.advanced_narrative_pack?.measurements?.tenor_affect_affective || 0,
        engagement: attributes.advanced_narrative_pack?.measurements?.reader_engagement_engaging || 0
      },
      perspective: {
        temporal: attributes.advanced_narrative_pack?.measurements?.temporal_perspective_retrospective || 0,
        focalization: attributes.advanced_narrative_pack?.measurements?.focalization_type_internal || 0,
        distance: attributes.advanced_narrative_pack?.measurements?.narrative_distance_close || 0
      },
      reliability_index: 1 - (attributes.advanced_narrative_pack?.measurements?.overt_persuasion_overt_persuasive || 0),
      emotional_signature: attributes.advanced_narrative_pack?.measurements?.tenor_affect_affective || 0
    };
    return persona;
  };

  const extractStyleEssence = async (attributes) => {
    // Extract linguistic style information from measurements
    const style = {
      linguistic_patterns: {
        mode_density: attributes.advanced_narrative_pack?.measurements?.mode_density_dense || 0,
        elaboration: attributes.advanced_narrative_pack?.measurements?.elaborated_reference_elaborated || 0,
        multimodal: attributes.advanced_narrative_pack?.measurements?.multimodal_integration_multimodal || 0
      },
      complexity_level: attributes.advanced_narrative_pack?.measurements?.cognitive_load_complex || 0,
      register_formality: attributes.advanced_narrative_pack?.measurements?.tenor_formality_formal || 0,
      discourse_coherence: attributes.advanced_narrative_pack?.measurements?.discourse_coherence_tight || 0
    };
    return style;
  };

  // Load packs on mount
  useEffect(() => {
    loadAvailablePacks();
  }, [loadAvailablePacks]);

  // Render input stage
  const renderInputStage = () => (
    <div style={{ padding: 20 }}>
      <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>📝 Narrative Input</h3>
      
      <div style={{ marginBottom: 15 }}>
        <label style={{ display: 'block', marginBottom: 5, fontSize: 14, fontWeight: 'bold' }}>
          Extended Narrative Text:
        </label>
        <textarea
          value={narrativeText}
          onChange={(e) => setNarrativeText(e.target.value)}
          placeholder="Paste your narrative text here for quantum distillation..."
          style={{
            width: '100%',
            height: 300,
            padding: 12,
            border: '2px solid #ddd',
            borderRadius: 8,
            fontSize: 13,
            fontFamily: 'Georgia, serif',
            lineHeight: 1.6,
            resize: 'vertical'
          }}
        />
        <div style={{ fontSize: 11, color: '#666', marginTop: 5 }}>
          Characters: {narrativeText.length} | Estimated reading time: {Math.ceil(narrativeText.length / 200)} quantum cycles
        </div>
      </div>

      <div style={{ textAlign: 'center' }}>
        <button
          onClick={() => setDistillationStage('configure')}
          disabled={!narrativeText.trim()}
          style={{
            padding: '12px 24px',
            backgroundColor: narrativeText.trim() ? '#4CAF50' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 'bold',
            cursor: narrativeText.trim() ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s'
          }}
        >
          ⚙️ Configure Distillation
        </button>
      </div>
    </div>
  );

  // Render configuration stage
  const renderConfigurationStage = () => (
    <div style={{ padding: 20 }}>
      <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>⚙️ Distillation Configuration</h3>
      
      {/* POVM Strategy Selection */}
      <div style={{ marginBottom: 20 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Measurement Strategy</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 10 }}>
          {Object.entries(distillationStrategies).map(([key, strategy]) => (
            <div
              key={key}
              onClick={() => setPovmStrategy(key)}
              style={{
                padding: 12,
                border: `2px solid ${povmStrategy === key ? '#9C27B0' : '#ddd'}`,
                borderRadius: 8,
                cursor: 'pointer',
                backgroundColor: povmStrategy === key ? '#f3e5f5' : 'white',
                transition: 'all 0.2s'
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 'bold', marginBottom: 5 }}>
                {strategy.icon} {strategy.name}
              </div>
              <div style={{ fontSize: 11, color: '#666' }}>
                {strategy.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Channel Type Selection */}
      <div style={{ marginBottom: 20 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Reading Channel Type</h4>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {Object.entries(channelConfigs).map(([key, config]) => (
            <button
              key={key}
              onClick={() => setChannelType(key)}
              style={{
                padding: '8px 12px',
                border: `2px solid ${channelType === key ? '#2196F3' : '#ddd'}`,
                borderRadius: 6,
                backgroundColor: channelType === key ? '#2196F3' : 'white',
                color: channelType === key ? 'white' : '#333',
                cursor: 'pointer',
                fontSize: 12,
                transition: 'all 0.2s'
              }}
            >
              {config.icon} {config.name}
              {config.recommended && <span style={{ marginLeft: 5 }}>⭐</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Reading Parameters */}
      <div style={{ marginBottom: 20 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 14 }}>Reading Parameters</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 15 }}>
          <div>
            <label style={{ fontSize: 12, display: 'block', marginBottom: 5 }}>
              Reading Alpha (Absorption Rate): {readingAlpha}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={readingAlpha}
              onChange={(e) => setReadingAlpha(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: 10, color: '#666' }}>
              Lower: Gentle integration | Higher: Rapid absorption
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
        <button
          onClick={() => setDistillationStage('input')}
          style={{
            padding: '10px 20px',
            backgroundColor: '#666',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer'
          }}
        >
          ← Back to Input
        </button>
        <button
          onClick={executeDistillationWorkflow}
          style={{
            padding: '10px 20px',
            backgroundColor: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            fontSize: 14,
            fontWeight: 'bold'
          }}
        >
          🧪 Begin Distillation
        </button>
      </div>
    </div>
  );

  // Render distillation stage
  const renderDistillationStage = () => (
    <div style={{ padding: 20, textAlign: 'center' }}>
      <h3 style={{ margin: '0 0 20px 0', color: '#333' }}>🧪 Quantum Distillation in Progress</h3>
      
      <div style={{ marginBottom: 20 }}>
        <div style={{
          width: '100%',
          height: 20,
          backgroundColor: '#f0f0f0',
          borderRadius: 10,
          overflow: 'hidden',
          marginBottom: 10
        }}>
          <div style={{
            width: `${progress}%`,
            height: '100%',
            backgroundColor: '#FF9800',
            transition: 'width 0.3s ease'
          }} />
        </div>
        <div style={{ fontSize: 14, color: '#666' }}>
          {Math.round(progress)}% - {statusMessage}
        </div>
      </div>

      <div style={{ 
        padding: 20,
        backgroundColor: '#f8f9fa',
        borderRadius: 8,
        fontSize: 12,
        fontStyle: 'italic',
        color: '#555'
      }}>
        The quantum consciousness is processing your narrative through POVM measurements,
        extracting the essential components of namespace, persona, and style into pure
        rho-embeddings. This process reveals the deep structure of narrative meaning
        through quantum mechanical principles.
      </div>
    </div>
  );

  // Render export stage
  const renderExportStage = () => (
    <div style={{ padding: 20 }}>
      <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>✨ Distillation Complete</h3>
      
      {finalRhoEmbedding && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ 
            padding: 15,
            backgroundColor: '#e8f5e8',
            borderRadius: 8,
            border: '2px solid #4CAF50',
            marginBottom: 15
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#2E7D32' }}>
              🎯 Narrative Essence Successfully Extracted
            </h4>
            <div style={{ fontSize: 12, color: '#555' }}>
              Your narrative has been distilled into a pure quantum rho-embedding containing
              the essential components of meaning, separated into analyzable dimensions.
            </div>
          </div>

          {/* Essence Components Display */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 15, marginBottom: 20 }}>
            {namespaceEssence && (
              <div style={{ padding: 12, border: '1px solid #ddd', borderRadius: 6, backgroundColor: '#fff3e0' }}>
                <h5 style={{ margin: '0 0 8px 0', color: '#E65100' }}>🏷️ Namespace Essence</h5>
                <div style={{ fontSize: 11 }}>
                  <div>Genre: {namespaceEssence.semantic_domain}</div>
                  <div>Density: {(namespaceEssence.conceptual_density * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}
            
            {personaEssence && (
              <div style={{ padding: 12, border: '1px solid #ddd', borderRadius: 6, backgroundColor: '#e3f2fd' }}>
                <h5 style={{ margin: '0 0 8px 0', color: '#1565C0' }}>👤 Persona Essence</h5>
                <div style={{ fontSize: 11 }}>
                  <div>Emotional Signature: {(personaEssence.emotional_signature * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}
            
            {styleEssence && (
              <div style={{ padding: 12, border: '1px solid #ddd', borderRadius: 6, backgroundColor: '#f3e5f5' }}>
                <h5 style={{ margin: '0 0 8px 0', color: '#7B1FA2' }}>✍️ Style Essence</h5>
                <div style={{ fontSize: 11 }}>
                  <div>Complexity: {(styleEssence.complexity_level * 100).toFixed(1)}%</div>
                  <div>Formality: {(styleEssence.register_formality * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}
          </div>

          {/* Export Options */}
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={() => {
                const dataStr = JSON.stringify(finalRhoEmbedding, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `narrative_essence_${Date.now()}.json`;
                link.click();
              }}
              style={{
                padding: '12px 24px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 'bold',
                cursor: 'pointer',
                marginRight: 10
              }}
            >
              💾 Export Rho-Embedding
            </button>
            
            <button
              onClick={() => {
                setDistillationStage('input');
                setNarrativeText('');
                setCurrentRhoId(null);
                setExtractedAttributes(null);
                setFinalRhoEmbedding(null);
                setProgress(0);
              }}
              style={{
                padding: '12px 24px',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: 8,
                fontSize: 14,
                cursor: 'pointer'
              }}
            >
              🔄 New Distillation
            </button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: 20 }}>
      <div style={{ 
        textAlign: 'center',
        marginBottom: 30,
        borderBottom: '2px solid #eee',
        paddingBottom: 20
      }}>
        <h1 style={{ 
          margin: 0, 
          fontSize: 24, 
          color: '#333',
          background: 'linear-gradient(45deg, #9C27B0, #2196F3)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          ⚗️ Narrative Distillation Studio
        </h1>
        <div style={{ fontSize: 14, color: '#666', marginTop: 8 }}>
          Extract the quantum essence of narratives through precise POVM measurements
        </div>
      </div>

      <div style={{
        backgroundColor: 'white',
        borderRadius: 12,
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        overflow: 'hidden'
      }}>
        {distillationStage === 'input' && renderInputStage()}
        {distillationStage === 'configure' && renderConfigurationStage()}
        {distillationStage === 'distill' && renderDistillationStage()}
        {distillationStage === 'export' && renderExportStage()}
      </div>
    </div>
  );
}

export default NarrativeDistillationStudio;