import React, { useState } from 'react';
import { ChevronRight, Sparkles, Brain, Layers, Settings, BookOpen, Zap, Eye } from 'lucide-react';
import { useAnalyticLexicology } from '../../App.jsx';
import { SAMPLE_NARRATIVES, MASTERY_LEVELS } from '../../utils/constants.js';

// Import level-specific components
import { NoviceView } from '../levels/NoviceView.jsx';
import { CuriousView } from '../levels/CuriousView.jsx';
import { ExplorerView } from '../levels/ExplorerView.jsx';
import { ExpertView } from '../levels/ExpertView.jsx';

export const ProgressiveInterface = () => {
  const {
    masteryLevel,
    showUpgradePrompt,
    promoteUser,
    dismissUpgradePrompt,
    completeAction,
    transformNarrative,
    analyzeField,
    applyStanceTransformation,
    isLoading,
    error,
    quantumState,
    clearError
  } = useAnalyticLexicology();

  const [narrative, setNarrative] = useState(SAMPLE_NARRATIVES.simple);
  const [selectedField, setSelectedField] = useState([]);
  const [stanceMode, setStanceMode] = useState('literal');
  const [showingResults, setShowingResults] = useState(false);

  // Handle narrative transformation
  const handleTransform = async () => {
    clearError();
    setShowingResults(true);
    
    const result = await transformNarrative(narrative);
    if (result) {
      completeAction('transform_text');
      
      // Track additional actions based on mastery level
      if (masteryLevel === 'curious') {
        completeAction('view_quantum_metrics');
      }
    }
  };

  // Handle field analysis
  const handleFieldAnalysis = async () => {
    if (selectedField.length >= 2) {
      const result = await analyzeField(narrative, selectedField);
      if (result) {
        completeAction('select_field');
        completeAction('analyze_relationships');
      }
    }
  };

  // Handle stance transformation
  const handleStanceTransformation = async () => {
    const result = await applyStanceTransformation(narrative, selectedField, stanceMode);
    if (result) {
      completeAction('apply_stance');
    }
  };

  // Render mastery level badge
  const renderMasteryBadge = () => (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-gray-500">Level:</span>
      <span className={`level-badge level-${masteryLevel}`}>
        {masteryLevel}
      </span>
    </div>
  );

  // Render upgrade prompt based on current level
  const renderUpgradePrompt = () => {
    if (!showUpgradePrompt) return null;

    const prompts = {
      [MASTERY_LEVELS.NOVICE]: {
        icon: Eye,
        color: 'blue',
        title: 'Want to see what\'s happening inside?',
        description: 'Your text just became a quantum state—a mathematical representation of meaning that we can measure and transform.',
        buttonText: 'Show me the quantum view'
      },
      [MASTERY_LEVELS.CURIOUS]: {
        icon: Layers,
        color: 'purple',
        title: 'Ready to explore lexical fields?',
        description: 'Words don\'t exist in isolation—they form fields where meanings influence each other. Want to see how words in your text relate?',
        buttonText: 'Explore lexical fields'
      },
      [MASTERY_LEVELS.EXPLORER]: {
        icon: Brain,
        color: 'green',
        title: 'Master advanced stance control?',
        description: 'Control irony, metaphor, and complex meaning relationships with full Analytic Lexicology tools.',
        buttonText: 'Unlock expert controls'
      }
    };

    const prompt = prompts[masteryLevel];
    if (!prompt) return null;

    const IconComponent = prompt.icon;

    return (
      <div className={`upgrade-prompt bg-${prompt.color}-50 border border-${prompt.color}-200`}>
        <div className={`flex items-center gap-2 text-${prompt.color}-800 mb-2`}>
          <IconComponent className="w-4 h-4" />
          <span className="font-medium">{prompt.title}</span>
        </div>
        <p className={`text-${prompt.color}-700 text-sm mb-3`}>
          {prompt.description}
        </p>
        <div className="flex gap-2">
          <button 
            onClick={promoteUser}
            className={`bg-${prompt.color}-600 text-white px-4 py-2 rounded-md text-sm hover:bg-${prompt.color}-700 transition-colors`}
          >
            {prompt.buttonText}
          </button>
          <button 
            onClick={dismissUpgradePrompt}
            className="text-gray-500 hover:text-gray-700 px-4 py-2 text-sm"
          >
            Maybe later
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Analytic Lexicology</h1>
                <p className="text-sm text-gray-600">Transform narrative through quantum meaning</p>
              </div>
            </div>
            {renderMasteryBadge()}
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Main input area */}
        <div className="quantum-card mb-6">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-medium text-gray-900">Your Narrative</h2>
          </div>
          
          <textarea
            value={narrative}
            onChange={(e) => setNarrative(e.target.value)}
            className="w-full h-32 p-4 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Enter your narrative to transform its quantum meaning state..."
          />
          
          <div className="flex items-center justify-between mt-4">
            <div className="text-sm text-gray-500">
              {narrative.split(' ').filter(w => w.length > 0).length} words
            </div>
            
            <button
              onClick={handleTransform}
              disabled={!narrative.trim() || isLoading}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Zap className="w-4 h-4" />
              {isLoading ? 'Transforming...' : 'Transform'}
            </button>
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="text-red-800 font-medium">Error</div>
            <div className="text-red-700 text-sm">{error}</div>
          </div>
        )}

        {/* Progressive results based on mastery level */}
        {showingResults && quantumState && (
          <div className="space-y-6">
            {/* Render appropriate view based on mastery level */}
            {masteryLevel === MASTERY_LEVELS.NOVICE && (
              <NoviceView 
                quantumState={quantumState}
                narrative={narrative}
              />
            )}
            
            {masteryLevel === MASTERY_LEVELS.CURIOUS && (
              <CuriousView 
                quantumState={quantumState}
                narrative={narrative}
                onViewMetrics={() => completeAction('understand_measurements')}
              />
            )}
            
            {masteryLevel === MASTERY_LEVELS.EXPLORER && (
              <ExplorerView 
                quantumState={quantumState}
                narrative={narrative}
                selectedField={selectedField}
                onFieldSelect={setSelectedField}
                onFieldAnalysis={handleFieldAnalysis}
              />
            )}
            
            {masteryLevel === MASTERY_LEVELS.EXPERT && (
              <ExpertView 
                quantumState={quantumState}
                narrative={narrative}
                selectedField={selectedField}
                stanceMode={stanceMode}
                onFieldSelect={setSelectedField}
                onStanceModeChange={setStanceMode}
                onStanceTransformation={handleStanceTransformation}
              />
            )}
          </div>
        )}

        {/* Mastery progression prompt */}
        {renderUpgradePrompt()}

        {/* Footer with theory link */}
        <div className="mt-12 text-center">
          <div className="text-sm text-gray-500">
            Powered by <button className="text-blue-600 hover:text-blue-700 underline">Analytic Lexicology</button> theory
          </div>
        </div>
      </div>
    </div>
  );
};
