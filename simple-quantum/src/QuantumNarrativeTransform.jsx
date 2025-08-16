import React, { useState } from 'react';
import TransformationInterface from './components/TransformationInterface';
import AdvancedControls from './components/AdvancedControls';
import AuditTrail from './components/AuditTrail';
import TransformationHistory from './components/TransformationHistory';
import TestbedInterface from './components/TestbedInterface';
import { useQuantumTransformations } from './hooks/useQuantumTransformations';

const QuantumNarrativeTransform = () => {
  const [inputText, setInputText] = useState('');
  const [transformRequest, setTransformRequest] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showAuditTrail, setShowAuditTrail] = useState(false);
  const [showTestbed, setShowTestbed] = useState(false);
  const [transformationMode, setTransformationMode] = useState('compass'); // 'quick' or 'compass'
  
  // Advanced transformation parameters
  const [advancedParams, setAdvancedParams] = useState({
    strength: 0.7,
    creativity: 0.8,
    preservation: 0.8,
    complexity: 0.5,
    temperature: 0.3,
    language: ''
  });

  // Use custom hook for transformation logic
  const {
    isTransforming,
    transformedText,
    auditTrail,
    quantumDistance,
    currentRhoState,
    transformHistory,
    handleTransform,
    handleCompassTransformation,
    copyToClipboard
  } = useQuantumTransformations();

  const handleMainTransform = () => {
    const prompt = transformRequest || 'transform this text creatively';
    handleTransform(inputText, prompt, advancedParams);
  };

  const handleQuickTransform = (prompt) => {
    setTransformRequest(prompt);
    handleTransform(inputText, prompt, advancedParams);
  };

  const handleCompassTransform = (compassConfig) => {
    handleCompassTransformation(inputText, compassConfig, advancedParams);
  };

  return (
    <div style={{
      minHeight: '45vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
      fontFamily: 'SF Pro Display, -apple-system, BlinkMacSystemFont, system-ui, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        textAlign: 'center',
        marginBottom: '30px',
        color: 'white'
      }}>
        <h1 style={{
          fontSize: '42px',
          fontWeight: '700',
          margin: '0 0 8px 0',
          textShadow: '0 2px 10px rgba(0,0,0,0.3)'
        }}>
          🌀 Quantum Narrative Transform
        </h1>
        <p style={{
          fontSize: '18px',
          opacity: 0.9,
          margin: 0,
          fontWeight: '400'
        }}>
          Transform your stories through quantum mechanics
        </p>
      </div>

      {/* Main Interface */}
      <TransformationInterface
        inputText={inputText}
        setInputText={setInputText}
        transformedText={transformedText}
        isTransforming={isTransforming}
        transformRequest={transformRequest}
        setTransformRequest={setTransformRequest}
        transformationMode={transformationMode}
        setTransformationMode={setTransformationMode}
        quantumDistance={quantumDistance}
        currentRhoState={currentRhoState}
        showAdvanced={showAdvanced}
        setShowAdvanced={setShowAdvanced}
        advancedParams={advancedParams}
        setAdvancedParams={setAdvancedParams}
        onTransform={handleMainTransform}
        onQuickTransform={handleQuickTransform}
        onCompassTransformation={handleCompassTransform}
        onCopyToClipboard={copyToClipboard}
      />

      {/* Audit Trail */}
      <AuditTrail
        auditTrail={auditTrail}
        showAuditTrail={showAuditTrail}
        setShowAuditTrail={setShowAuditTrail}
      />

      {/* Transformation History */}
      <TransformationHistory
        transformHistory={transformHistory}
      />

      {/* Testbed Interface */}
      <TestbedInterface
        showTestbed={showTestbed}
        setShowTestbed={setShowTestbed}
      />

      {/* Footer */}
      <div style={{
        textAlign: 'center',
        marginTop: '30px',
        color: 'rgba(255,255,255,0.8)',
        fontSize: '14px'
      }}>
        Powered by quantum density matrices and POVM measurements
      </div>
    </div>
  );
};

export default QuantumNarrativeTransform;