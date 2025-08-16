import React, { useState, useEffect, useMemo } from 'react';
import { analyzeWordFrequency, generateWordCloudLayout } from '../utils/textAnalysis.js';

export const WordCloud = ({ 
  text, 
  selectedWords = [], 
  onWordSelect, 
  width = 600, 
  height = 400,
  maxWords = 40 
}) => {
  const [hoveredWord, setHoveredWord] = useState(null);

  // Analyze text and generate word cloud layout with collision detection
  const layoutData = useMemo(() => {
    if (!text) return { words: [], actualWidth: width, actualHeight: height };
    
    const wordAnalysis = analyzeWordFrequency(text, 3, maxWords);
    return generateWordCloudLayout(wordAnalysis, width, height);
  }, [text, width, height, maxWords]);

  const { words: wordsWithLayout, actualWidth, actualHeight } = layoutData;

  const handleWordClick = (word) => {
    if (onWordSelect) {
      onWordSelect(word);
    }
  };

  const isWordSelected = (word) => {
    return selectedWords.includes(word);
  };

  if (!text || wordsWithLayout.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center text-gray-500">
          <div className="text-lg font-medium mb-2">No text to analyze</div>
          <div className="text-sm">Enter some narrative text to generate a word cloud</div>
        </div>
      </div>
    );
  }

  return (
    <div className="word-cloud-container">
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm font-medium text-gray-700">
          Interactive Word Cloud ({wordsWithLayout.length} words)
        </div>
        <div className="text-xs text-gray-500">
          Click words to select for lexical field analysis
        </div>
      </div>
      
      <div className="relative border rounded-lg bg-white shadow-sm">
        <svg 
          width={actualWidth} 
          height={actualHeight}
          className="w-full h-auto max-w-full"
          viewBox={`0 0 ${actualWidth} ${actualHeight}`}
        >
          {/* Background */}
          <rect width={actualWidth} height={actualHeight} fill="#fafafa" rx="8" />
          
          {/* Words */}
          {wordsWithLayout.map((wordData, index) => {
            const isSelected = isWordSelected(wordData.word);
            const isHovered = hoveredWord === wordData.word;
            
            return (
              <g key={`${wordData.word}-${index}`}>
                {/* Selection background */}
                {isSelected && (
                  <ellipse
                    cx={wordData.x}
                    cy={wordData.y - wordData.fontSize * 0.2}
                    rx={wordData.word.length * wordData.fontSize * 0.3}
                    ry={wordData.fontSize * 0.6}
                    fill="#3b82f6"
                    fillOpacity="0.2"
                    stroke="#3b82f6"
                    strokeWidth="2"
                  />
                )}
                
                {/* Hover background */}
                {isHovered && !isSelected && (
                  <ellipse
                    cx={wordData.x}
                    cy={wordData.y - wordData.fontSize * 0.2}
                    rx={wordData.word.length * wordData.fontSize * 0.3}
                    ry={wordData.fontSize * 0.6}
                    fill="#6b7280"
                    fillOpacity="0.1"
                    stroke="#6b7280"
                    strokeWidth="1"
                  />
                )}
                
                {/* Word text */}
                <text
                  x={wordData.x}
                  y={wordData.y}
                  fontSize={wordData.fontSize}
                  fontFamily="system-ui, -apple-system, sans-serif"
                  fontWeight={isSelected ? 'bold' : wordData.weight > 0.7 ? '600' : '400'}
                  fill={isSelected ? '#1e40af' : wordData.color}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="cursor-pointer select-none transition-all duration-200"
                  style={{
                    filter: isHovered ? 'brightness(1.2)' : 'none',
                    transform: isHovered ? 'scale(1.05)' : 'scale(1)',
                    transformOrigin: `${wordData.x}px ${wordData.y}px`
                  }}
                  onMouseEnter={() => setHoveredWord(wordData.word)}
                  onMouseLeave={() => setHoveredWord(null)}
                  onClick={() => handleWordClick(wordData.word)}
                >
                  {wordData.word}
                </text>
                
                {/* Frequency indicator for high-frequency words */}
                {wordData.weight > 0.8 && (
                  <circle
                    cx={wordData.x + wordData.word.length * wordData.fontSize * 0.25}
                    cy={wordData.y - wordData.fontSize * 0.4}
                    r="3"
                    fill="#f59e0b"
                    opacity="0.7"
                  />
                )}
              </g>
            );
          })}
          
          {/* Legend */}
          <g transform={`translate(${actualWidth - 120}, 20)`}>
            <rect x="0" y="0" width="110" height="60" fill="white" fillOpacity="0.9" stroke="#e5e7eb" rx="4" />
            <text x="5" y="15" fontSize="10" fill="#6b7280" fontWeight="600">Word Size:</text>
            <text x="5" y="28" fontSize="8" fill="#6b7280">Large = Frequent</text>
            <text x="5" y="40" fontSize="8" fill="#6b7280">• = High frequency</text>
            <text x="5" y="52" fontSize="8" fill="#3b82f6">Blue = Selected</text>
          </g>
        </svg>
      </div>
      
      {/* Selected words summary */}
      {selectedWords.length > 0 && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="text-sm font-medium text-blue-800 mb-2">
            Selected Words ({selectedWords.length}):
          </div>
          <div className="flex flex-wrap gap-2">
            {selectedWords.map(word => (
              <span 
                key={word}
                className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full cursor-pointer hover:bg-blue-200"
                onClick={() => handleWordClick(word)}
              >
                {word}
                <span className="ml-1 text-blue-600">×</span>
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Instructions */}
      <div className="mt-2 text-xs text-gray-500 text-center">
        Word size reflects frequency in your text. Click words to build your lexical field for analysis.
      </div>
    </div>
  );
};