/**
 * POVMCreator - Interface for creating custom POVM measurements.
 * 
 * Allows users to define new POVMs based on dialectical concepts
 * and narrative dimensions, with AI assistance for concept definition.
 */

import React, { useState, useCallback } from 'react';
import { safeFetch } from '../utils/api.js';

function POVMCreator({ onPOVMCreated, disabled = false }) {
  const [povmName, setPovmName] = useState('');
  const [dialecticalConcept, setDialecticalConcept] = useState('');
  const [conceptDescription, setConceptDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [povmType, setPovmType] = useState('dialectical');
  const [numMeasurements, setNumMeasurements] = useState(8);

  const dialecticalPresets = [
    {
      name: 'hope_despair',
      label: 'Hope ⟷ Despair',
      description: 'Optimistic vs pessimistic narrative outlook',
      concept: 'The fundamental tension between hopeful anticipation and despairing resignation'
    },
    {
      name: 'objective_subjective',
      label: 'Objective ⟷ Subjective',
      description: 'Factual reporting vs personal interpretation',
      concept: 'The spectrum from objective observation to subjective experience'
    },
    {
      name: 'known_unknown',
      label: 'Known ⟷ Unknown',
      description: 'Certainty vs mystery in narrative elements',
      concept: 'The dialectic between revealed knowledge and hidden mystery'
    },
    {
      name: 'individual_collective',
      label: 'Individual ⟷ Collective',
      description: 'Personal vs societal perspective',
      concept: 'The tension between individual agency and collective influence'
    },
    {
      name: 'past_future',
      label: 'Past ⟷ Future',
      description: 'Historical grounding vs forward projection',
      concept: 'The temporal dialectic between what was and what might be'
    },
    {
      name: 'order_chaos',
      label: 'Order ⟷ Chaos',
      description: 'Structured vs chaotic narrative flow',
      concept: 'The fundamental opposition between systematic order and creative chaos'
    }
  ];

  const handlePresetSelect = useCallback((preset) => {
    setPovmName(preset.name);
    setDialecticalConcept(preset.label);
    setConceptDescription(preset.description);
  }, []);

  const createCustomPOVM = useCallback(async () => {
    if (!povmName.trim() || !dialecticalConcept.trim()) {
      alert('Please provide both a name and dialectical concept.');
      return;
    }

    setIsCreating(true);
    try {
      const response = await safeFetch('/advanced/create-povm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pack_name: povmName.trim(),
          dialectical_concept: dialecticalConcept.trim(),
          description: conceptDescription.trim(),
          povm_type: povmType,
          n_measurements: numMeasurements
        })
      });

      const result = await response.json();
      
      if (onPOVMCreated) {
        onPOVMCreated(result);
      }

      // Reset form
      setPovmName('');
      setDialecticalConcept('');
      setConceptDescription('');
      alert(`POVM "${povmName}" created successfully!`);

    } catch (error) {
      console.error('Failed to create POVM:', error);
      alert('Failed to create POVM. Please try again.');
    } finally {
      setIsCreating(false);
    }
  }, [povmName, dialecticalConcept, conceptDescription, povmType, numMeasurements, onPOVMCreated]);

  const generateOptimalPOVM = useCallback(async () => {
    if (!povmName.trim()) {
      alert('Please provide a name for the POVM.');
      return;
    }

    setIsCreating(true);
    try {
      const response = await safeFetch('/advanced/generate-optimal-povm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pack_name: povmName.trim(),
          description: conceptDescription.trim(),
          n_measurements: numMeasurements
        })
      });

      const result = await response.json();
      
      if (onPOVMCreated) {
        onPOVMCreated(result);
      }

      alert(`Optimal POVM "${povmName}" generated successfully!`);

    } catch (error) {
      console.error('Failed to generate optimal POVM:', error);
      alert('Failed to generate optimal POVM. Please try again.');
    } finally {
      setIsCreating(false);
    }
  }, [povmName, conceptDescription, numMeasurements, onPOVMCreated]);

  return (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '16px',
      marginTop: '16px',
      backgroundColor: '#fafafa'
    }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', color: '#333' }}>
        🧪 POVM Creator
      </h3>
      
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', fontSize: '12px' }}>
          POVM Name:
        </label>
        <input
          type="text"
          value={povmName}
          onChange={(e) => setPovmName(e.target.value)}
          placeholder="e.g., custom_narrative_dimension"
          disabled={disabled || isCreating}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', fontSize: '12px' }}>
          Dialectical Presets:
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          {dialecticalPresets.map((preset) => (
            <button
              key={preset.name}
              onClick={() => handlePresetSelect(preset)}
              disabled={disabled || isCreating}
              style={{
                padding: '8px',
                fontSize: '11px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                backgroundColor: povmName === preset.name ? '#e3f2fd' : 'white',
                cursor: disabled || isCreating ? 'not-allowed' : 'pointer',
                textAlign: 'left'
              }}
              title={preset.concept}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', fontSize: '12px' }}>
          Custom Dialectical Concept:
        </label>
        <input
          type="text"
          value={dialecticalConcept}
          onChange={(e) => setDialecticalConcept(e.target.value)}
          placeholder="e.g., Sacred ⟷ Profane"
          disabled={disabled || isCreating}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px'
          }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', fontSize: '12px' }}>
          Description:
        </label>
        <textarea
          value={conceptDescription}
          onChange={(e) => setConceptDescription(e.target.value)}
          placeholder="Describe what this dialectical dimension measures..."
          disabled={disabled || isCreating}
          rows={3}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px',
            resize: 'vertical'
          }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            padding: '4px 8px',
            fontSize: '11px',
            border: 'none',
            backgroundColor: 'transparent',
            color: '#666',
            cursor: 'pointer',
            textDecoration: 'underline'
          }}
        >
          {showAdvanced ? '▼ Hide Advanced' : '▶ Show Advanced'}
        </button>

        {showAdvanced && (
          <div style={{ marginTop: '12px', padding: '12px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', fontSize: '12px' }}>
                POVM Type:
              </label>
              <select
                value={povmType}
                onChange={(e) => setPovmType(e.target.value)}
                disabled={disabled || isCreating}
                style={{
                  width: '100%',
                  padding: '6px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '12px'
                }}
              >
                <option value="dialectical">Dialectical (Binary Opposition)</option>
                <option value="multiclass">Multiclass (Multiple Categories)</option>
                <option value="coverage">Coverage (Comprehensive Measurement)</option>
                <option value="attribute">Attribute (Specific Property)</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', fontSize: '12px' }}>
                Number of Measurements:
              </label>
              <input
                type="number"
                min="2"
                max="32"
                value={numMeasurements}
                onChange={(e) => setNumMeasurements(parseInt(e.target.value) || 8)}
                disabled={disabled || isCreating}
                style={{
                  width: '80px',
                  padding: '6px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '12px'
                }}
              />
            </div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={createCustomPOVM}
          disabled={disabled || isCreating || !povmName.trim() || !dialecticalConcept.trim()}
          style={{
            padding: '10px 16px',
            fontSize: '12px',
            fontWeight: 'bold',
            border: 'none',
            borderRadius: '4px',
            backgroundColor: disabled || isCreating || !povmName.trim() || !dialecticalConcept.trim() 
              ? '#f5f5f5' : '#4caf50',
            color: disabled || isCreating || !povmName.trim() || !dialecticalConcept.trim() 
              ? '#999' : 'white',
            cursor: disabled || isCreating || !povmName.trim() || !dialecticalConcept.trim() 
              ? 'not-allowed' : 'pointer',
            flex: 1
          }}
        >
          {isCreating ? '⏳ Creating...' : '🎯 Create Custom POVM'}
        </button>

        <button
          onClick={generateOptimalPOVM}
          disabled={disabled || isCreating || !povmName.trim()}
          style={{
            padding: '10px 16px',
            fontSize: '12px',
            fontWeight: 'bold',
            border: 'none',
            borderRadius: '4px',
            backgroundColor: disabled || isCreating || !povmName.trim() 
              ? '#f5f5f5' : '#ff9800',
            color: disabled || isCreating || !povmName.trim() 
              ? '#999' : 'white',
            cursor: disabled || isCreating || !povmName.trim() 
              ? 'not-allowed' : 'pointer',
            flex: 1
          }}
        >
          {isCreating ? '⏳ Generating...' : '🤖 Generate Optimal'}
        </button>
      </div>

      <div style={{ marginTop: '12px', fontSize: '11px', color: '#666', fontStyle: 'italic' }}>
        💡 Tip: Use dialectical concepts to define measurements that capture the fundamental tensions in your narrative.
        Optimal generation uses current narrative content to create maximally discriminating POVMs.
      </div>
    </div>
  );
}

export default POVMCreator;