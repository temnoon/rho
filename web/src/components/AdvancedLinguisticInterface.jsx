/**
 * AdvancedLinguisticInterface - Professional narrative analysis interface.
 * 
 * Provides comprehensive control over narrative attributes based on:
 * - Biber's Multi-Dimensional Analysis 
 * - Systemic Functional Linguistics
 * - Computational Narratology
 * - Reader Response Theory
 */

import React, { useState, useEffect } from 'react';

function AdvancedLinguisticInterface({ currentRho, onAttributeChange, disabled = false, manualMode = false, onManualUpdate }) {
  const [activeCategory, setActiveCategory] = useState('biber_dimensions');
  const [measurements, setMeasurements] = useState({});
  const [attributeValues, setAttributeValues] = useState({});
  const [impactAnalysis, setImpactAnalysis] = useState(null);
  const [showImpactHelp, setShowImpactHelp] = useState(false);

  // Comprehensive attribute framework based on linguistic research
  const attributeCategories = {
    biber_dimensions: {
      title: 'Biber Multi-Dimensional Analysis',
      description: 'Six fundamental dimensions of linguistic variation',
      color: '#1976d2',
      attributes: {
        involved_production: {
          label: 'Involved ↔ Informational',
          description: 'Personal engagement vs. detached presentation',
          left: 'Informational (detached, factual)',
          right: 'Involved (personal, engaged)',
          features: ['1st person pronouns', 'contractions', 'present tense']
        },
        narrative_concerns: {
          label: 'Non-narrative ↔ Narrative',
          description: 'Expository vs. story-telling modes',
          left: 'Non-narrative (expository)',
          right: 'Narrative (story-telling)',
          features: ['past tense', '3rd person', 'perfect aspect']
        },
        elaborated_reference: {
          label: 'Situation-dependent ↔ Elaborated',
          description: 'Contextual vs. explicit meaning',
          left: 'Situation-dependent',
          right: 'Elaborated reference',
          features: ['relative clauses', 'nominalizations']
        },
        overt_persuasion: {
          label: 'Covert ↔ Overt Persuasion',
          description: 'Subtle vs. explicit argumentative stance',
          left: 'Covert persuasion',
          right: 'Overt persuasion',
          features: ['modals', 'infinitives', 'conditionals']
        },
        abstract_information: {
          label: 'Concrete ↔ Abstract',
          description: 'Tangible vs. conceptual content',
          left: 'Concrete information',
          right: 'Abstract information',
          features: ['passives', 'conjuncts', 'technical terms']
        },
        online_elaboration: {
          label: 'Planned ↔ Online Elaboration',
          description: 'Pre-planned vs. real-time discourse',
          left: 'Planned discourse',
          right: 'Online elaboration',
          features: ['complement clauses', 'wh-clauses']
        }
      }
    },
    sfl_metafunctions: {
      title: 'Systemic Functional Linguistics',
      description: 'Three metafunctions of language',
      color: '#388e3c',
      attributes: {
        field_register: {
          label: 'Everyday ↔ Specialized',
          description: 'Domain specificity and expertise level',
          left: 'Everyday topics',
          right: 'Specialized field',
          features: ['technical vocabulary', 'domain terms']
        },
        tenor_formality: {
          label: 'Informal ↔ Formal',
          description: 'Social relationship and power dynamics',
          left: 'Informal register',
          right: 'Formal register',
          features: ['honorifics', 'modal verbs', 'hedging']
        },
        tenor_affect: {
          label: 'Neutral ↔ Affective',
          description: 'Emotional stance and attitude',
          left: 'Neutral tone',
          right: 'Affective engagement',
          features: ['evaluative adjectives', 'intensifiers']
        },
        mode_density: {
          label: 'Sparse ↔ Dense',
          description: 'Information density and complexity',
          left: 'Sparse information',
          right: 'Dense information',
          features: ['lexical density', 'complex clauses']
        }
      }
    },
    narratology: {
      title: 'Computational Narratology',
      description: 'Narrative structure and perspective',
      color: '#7b1fa2',
      attributes: {
        temporal_perspective: {
          label: 'Retrospective ↔ Prospective',
          description: 'Temporal orientation in narrative',
          left: 'Retrospective view',
          right: 'Prospective view',
          features: ['tense patterns', 'temporal adverbials']
        },
        focalization_type: {
          label: 'External ↔ Internal',
          description: 'Narrative perspective and access to consciousness',
          left: 'External focalization',
          right: 'Internal focalization',
          features: ['consciousness verbs', 'mental states']
        },
        narrative_distance: {
          label: 'Close ↔ Distant',
          description: 'Proximity to events and characters',
          left: 'Close narrative distance',
          right: 'Distant narrative distance',
          features: ['deictic expressions', 'spatial markers']
        }
      }
    },
    reader_response: {
      title: 'Reader Response Metrics',
      description: 'Audience engagement and processing',
      color: '#f57c00',
      attributes: {
        reader_engagement: {
          label: 'Passive ↔ Engaging',
          description: 'Reader participation and involvement',
          left: 'Passive reception',
          right: 'Active engagement',
          features: ['direct address', 'questions', 'imperatives']
        },
        cognitive_load: {
          label: 'Simple ↔ Complex',
          description: 'Processing demands on the reader',
          left: 'Simple processing',
          right: 'Complex processing',
          features: ['sentence length', 'syntactic complexity']
        },
        cultural_specificity: {
          label: 'Universal ↔ Culture-specific',
          description: 'Cultural knowledge requirements',
          left: 'Universal content',
          right: 'Culture-specific',
          features: ['cultural references', 'idioms']
        }
      }
    },
    discourse_analysis: {
      title: 'Discourse & Genre Analysis',
      description: 'Text structure and generic conventions',
      color: '#d32f2f',
      attributes: {
        discourse_coherence: {
          label: 'Loose ↔ Tight',
          description: 'Coherence and cohesion in discourse',
          left: 'Loose coherence',
          right: 'Tight coherence',
          features: ['discourse markers', 'lexical chains']
        },
        genre_conformity: {
          label: 'Conventional ↔ Innovative',
          description: 'Adherence to genre conventions',
          left: 'Conventional forms',
          right: 'Genre innovation',
          features: ['genre markers', 'structural patterns']
        },
        intertextual_density: {
          label: 'Original ↔ Intertextual',
          description: 'References to other texts and works',
          left: 'Original content',
          right: 'Dense intertextuality',
          features: ['quotations', 'allusions', 'parody']
        }
      }
    }
  };

  // Fetch impact analysis when currentRho changes
  useEffect(() => {
    if (currentRho?.rho_id) {
      fetchImpactAnalysis();
    }
  }, [currentRho?.rho_id]);

  const fetchImpactAnalysis = async () => {
    if (!currentRho?.rho_id) return;
    
    try {
      const response = await fetch(`http://localhost:8192/povm-attributes/${currentRho.rho_id}/impact_analysis`);
      if (response.ok) {
        const data = await response.json();
        setImpactAnalysis(data);
      } else {
        console.warn('Failed to fetch impact analysis:', response.status);
      }
    } catch (error) {
      console.warn('Impact analysis unavailable:', error);
    }
  };

  const getAttributeImpactStyle = (attributeKey) => {
    if (!impactAnalysis?.impact_categories?.[attributeKey]) {
      return { borderColor: '#e0e0e0', backgroundColor: '#fafafa' };
    }
    
    const impact = impactAnalysis.impact_categories[attributeKey];
    const baseStyle = {
      borderWidth: '2px',
      borderStyle: 'solid'
    };
    
    switch (impact.level) {
      case 'high':
        return {
          ...baseStyle,
          borderColor: '#ff4444',
          backgroundColor: '#fff5f5',
          boxShadow: '0 0 0 1px rgba(255, 68, 68, 0.2)'
        };
      case 'medium':
        return {
          ...baseStyle,
          borderColor: '#ff9800',
          backgroundColor: '#fff8f0',
          boxShadow: '0 0 0 1px rgba(255, 152, 0, 0.2)'
        };
      case 'low':
        return {
          ...baseStyle,
          borderColor: '#9e9e9e',
          backgroundColor: '#fafafa'
        };
      default:
        return { borderColor: '#e0e0e0', backgroundColor: '#fafafa' };
    }
  };

  const getImpactIndicator = (attributeKey) => {
    if (!impactAnalysis?.impact_categories?.[attributeKey]) return null;
    
    const impact = impactAnalysis.impact_categories[attributeKey];
    const icons = {
      high: '🔴',
      medium: '🟡', 
      low: '⚪'
    };
    
    return (
      <span 
        style={{ 
          fontSize: '16px', 
          marginLeft: '8px',
          cursor: 'help'
        }}
        title={`${impact.level.toUpperCase()} impact: ${impact.level === 'high' ? 'Strong effect on narrative style' : impact.level === 'medium' ? 'Moderate stylistic changes' : 'Subtle effects, good for fine-tuning'}`}
      >
        {icons[impact.level]}
      </span>
    );
  };

  const CategorySelector = () => (
    <div style={{ marginBottom: '16px' }}>
      <select
        value={activeCategory}
        onChange={(e) => setActiveCategory(e.target.value)}
        style={{
          width: '100%',
          padding: '12px',
          border: '2px solid #e0e0e0',
          borderRadius: '8px',
          fontSize: '16px',
          fontWeight: '500',
          backgroundColor: 'white',
          color: '#333',
          cursor: 'pointer'
        }}
      >
        {Object.entries(attributeCategories).map(([key, category]) => (
          <option key={key} value={key}>
            {category.title} ({Object.keys(category.attributes).length} attributes)
          </option>
        ))}
      </select>
    </div>
  );

  const AttributeSlider = ({ attributeKey, attribute, manualMode }) => {
    const currentValue = attributeValues[attributeKey] || 0;
    
    const handleChange = (newValue) => {
      setAttributeValues(prev => ({...prev, [attributeKey]: newValue}));
      // Only call onAttributeChange immediately if not in manual mode
      if (onAttributeChange && !manualMode) {
        onAttributeChange(attributeKey, newValue);
      }
    };

    return (
      <div style={{ 
        marginBottom: '16px',
        padding: '12px',
        borderRadius: '6px',
        ...getAttributeImpactStyle(attributeKey)
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '8px'
        }}>
          <div style={{ 
            fontWeight: '600', 
            fontSize: '14px',
            color: '#333',
            display: 'flex',
            alignItems: 'center',
            flex: 1
          }}>
            {attribute.label.split(' ↔ ')[0]}
            {getImpactIndicator(attributeKey)}
          </div>
          <div style={{ 
            minWidth: '50px',
            textAlign: 'center',
            fontSize: '16px',
            fontWeight: 'bold',
            color: attributeCategories[activeCategory].color
          }}>
            {currentValue.toFixed(2)}
          </div>
        </div>
        
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px'
        }}>
          <span style={{ 
            fontSize: '10px', 
            color: '#666', 
            minWidth: '60px',
            textAlign: 'right'
          }}>
            {attribute.left.split(' ')[0]}
          </span>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={currentValue}
            onChange={(e) => handleChange(parseFloat(e.target.value))}
            disabled={disabled}
            style={{
              flex: 1,
              height: '4px',
              background: `linear-gradient(to right, #ccc 0%, ${attributeCategories[activeCategory].color} 50%, #ccc 100%)`,
              outline: 'none',
              borderRadius: '2px'
            }}
          />
          <span style={{ 
            fontSize: '10px', 
            color: '#666', 
            minWidth: '60px',
            textAlign: 'left'
          }}>
            {attribute.right.split(' ')[0]}
          </span>
        </div>
        
        {measurements[attributeKey] && (
          <div style={{ 
            fontSize: '10px', 
            color: '#888',
            marginTop: '6px',
            textAlign: 'center'
          }}>
            📊 {(measurements[attributeKey] * 100).toFixed(1)}%
          </div>
        )}
      </div>
    );
  };

  const activeAttributes = attributeCategories[activeCategory]?.attributes || {};

  return (
    <div style={{ padding: '12px' }}>
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{ 
          margin: '0 0 4px 0', 
          color: '#333',
          fontSize: '18px'
        }}>
          🎯 Linguistic Attributes
        </h3>
        <p style={{ 
          margin: '0 0 12px 0', 
          color: '#666',
          fontSize: '13px'
        }}>
          Professional narrative control via computational linguistics
        </p>
      </div>

      <CategorySelector />

      {/* Compact Impact Analysis */}
      {impactAnalysis && (
        <div style={{
          border: '1px solid #2196f3',
          borderRadius: '6px',
          padding: '10px',
          marginBottom: '16px',
          backgroundColor: '#f8fbff',
          fontSize: '12px'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '8px'
          }}>
            <strong style={{ color: '#2196f3', fontSize: '13px' }}>
              Impact Guide
            </strong>
            <button
              onClick={() => setShowImpactHelp(!showImpactHelp)}
              style={{
                padding: '2px 6px',
                background: 'transparent',
                border: '1px solid #2196f3',
                borderRadius: '3px',
                color: '#2196f3',
                cursor: 'pointer',
                fontSize: '10px'
              }}
            >
              {showImpactHelp ? '−' : '+'}
            </button>
          </div>
          
          {showImpactHelp ? (
            <div style={{ lineHeight: '1.3' }}>
              <div><strong>🔴 High:</strong> Major style changes</div>
              <div><strong>🟡 Medium:</strong> Moderate changes</div>
              <div><strong>⚪ Low:</strong> Subtle fine-tuning</div>
            </div>
          ) : (
            <div style={{ color: '#666' }}>
              🔴 High impact • 🟡 Medium • ⚪ Low - click + to expand
            </div>
          )}
        </div>
      )}

      <div style={{ marginBottom: '12px' }}>
        <h4 style={{ 
          margin: '0 0 8px 0',
          color: attributeCategories[activeCategory].color,
          fontSize: '16px'
        }}>
          {attributeCategories[activeCategory].title}
        </h4>
      </div>

      <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
        {Object.entries(activeAttributes).map(([key, attribute]) => (
          <AttributeSlider 
            key={key}
            attributeKey={key}
            attribute={attribute}
            manualMode={manualMode}
          />
        ))}
      </div>

      {disabled && (
        <div style={{
          padding: '8px',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffeaa7',
          borderRadius: '4px',
          color: '#8c6c00',
          textAlign: 'center',
          fontSize: '12px',
          marginTop: '12px'
        }}>
          ⚡ Processing...
        </div>
      )}
    </div>
  );
}

export default AdvancedLinguisticInterface;