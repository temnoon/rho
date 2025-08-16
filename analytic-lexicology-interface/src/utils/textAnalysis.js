/**
 * Text analysis utilities for word clouds and lexical field selection
 */

// Common stop words to exclude from word clouds
export const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
  'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
  'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
  'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else', 'than', 'as', 'so',
  'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
  'very', 'too', 'just', 'only', 'now', 'also', 'more', 'most', 'much', 'many',
  'some', 'any', 'all', 'each', 'both', 'either', 'neither', 'other', 'another',
  'same', 'different', 'new', 'old', 'first', 'last', 'next', 'previous',
  'before', 'after', 'during', 'while', 'until', 'since', 'from', 'into', 'through'
]);

/**
 * Analyze text and extract word frequencies
 * @param {string} text - Input text to analyze
 * @param {number} minLength - Minimum word length (default: 3)
 * @param {number} maxWords - Maximum number of words to return (default: 50)
 * @returns {Array} Array of {word, frequency, weight} objects sorted by frequency
 */
export const analyzeWordFrequency = (text, minLength = 3, maxWords = 50) => {
  if (!text || typeof text !== 'string') {
    return [];
  }

  // Clean and tokenize text
  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ') // Replace punctuation with spaces
    .split(/\s+/)
    .filter(word => 
      word.length >= minLength && 
      !STOP_WORDS.has(word) &&
      /^[a-z]+$/.test(word) // Only letters, no numbers
    );

  // Count frequencies
  const frequencies = {};
  words.forEach(word => {
    frequencies[word] = (frequencies[word] || 0) + 1;
  });

  // Convert to array and sort by frequency
  const wordArray = Object.entries(frequencies)
    .map(([word, frequency]) => ({
      word,
      frequency,
      weight: frequency // Will be normalized later
    }))
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, maxWords);

  // Normalize weights (0.1 to 1.0)
  if (wordArray.length > 0) {
    const maxFreq = wordArray[0].frequency;
    const minFreq = wordArray[wordArray.length - 1].frequency;
    
    wordArray.forEach(item => {
      if (maxFreq === minFreq) {
        item.weight = 1.0;
      } else {
        item.weight = 0.1 + (0.9 * (item.frequency - minFreq) / (maxFreq - minFreq));
      }
    });
  }

  return wordArray;
};

/**
 * Generate word cloud layout with collision detection (Pauli exclusion principle)
 * @param {Array} words - Array of word objects with weight property
 * @param {number} width - Initial SVG width (will grow as needed)
 * @param {number} height - Initial SVG height (will grow as needed)
 * @returns {Object} {words: Array, actualWidth: number, actualHeight: number}
 */
export const generateWordCloudLayout = (words, width = 600, height = 400) => {
  if (!words || words.length === 0) {
    return { words: [], actualWidth: width, actualHeight: height };
  }

  const baseFontSize = 12;
  const maxFontSize = 48;
  const padding = 8; // Space between words
  const maxAttempts = 100; // Max placement attempts per word
  
  // Sort words by weight (largest first for better packing)
  const sortedWords = [...words].sort((a, b) => b.weight - a.weight);
  
  const placedWords = [];
  let minX = width / 2, maxX = width / 2;
  let minY = height / 2, maxY = height / 2;
  
  sortedWords.forEach((word, index) => {
    const fontSize = baseFontSize + (maxFontSize - baseFontSize) * word.weight;
    const wordWidth = word.word.length * fontSize * 0.6; // Approximate text width
    const wordHeight = fontSize * 1.2; // Text height with padding
    
    let placed = false;
    let attempts = 0;
    let bestPosition = null;
    let minDistanceToCenter = Infinity;
    
    // Try to place the word starting from center and spiraling outward
    while (!placed && attempts < maxAttempts) {
      const spiralRadius = Math.sqrt(attempts) * 20;
      const angle = attempts * 0.5; // Golden angle for better distribution
      
      const tryX = width / 2 + spiralRadius * Math.cos(angle);
      const tryY = height / 2 + spiralRadius * Math.sin(angle);
      
      // Check collision with existing words
      const hasCollision = placedWords.some(placedWord => {
        const dx = Math.abs(tryX - placedWord.x);
        const dy = Math.abs(tryY - placedWord.y);
        
        // Calculate collision boundaries (word dimensions + padding)
        const collisionWidth = (wordWidth + placedWord.wordWidth) / 2 + padding;
        const collisionHeight = (wordHeight + placedWord.wordHeight) / 2 + padding;
        
        return dx < collisionWidth && dy < collisionHeight;
      });
      
      if (!hasCollision) {
        const distanceToCenter = Math.sqrt((tryX - width/2)**2 + (tryY - height/2)**2);
        if (distanceToCenter < minDistanceToCenter) {
          minDistanceToCenter = distanceToCenter;
          bestPosition = { x: tryX, y: tryY };
        }
        
        // If we found a good spot, use it (prefer closer to center)
        if (attempts > 50 || (distanceToCenter < 150 && attempts > 10)) {
          placed = true;
        }
      }
      
      attempts++;
    }
    
    // Use best position found, or fallback position
    const finalPosition = bestPosition || {
      x: width / 2 + (Math.random() - 0.5) * width,
      y: height / 2 + (Math.random() - 0.5) * height
    };
    
    const placedWord = {
      ...word,
      x: finalPosition.x,
      y: finalPosition.y,
      fontSize,
      wordWidth,
      wordHeight,
      color: getWordColor(word.weight, index)
    };
    
    placedWords.push(placedWord);
    
    // Update bounding box
    minX = Math.min(minX, finalPosition.x - wordWidth / 2);
    maxX = Math.max(maxX, finalPosition.x + wordWidth / 2);
    minY = Math.min(minY, finalPosition.y - wordHeight / 2);
    maxY = Math.max(maxY, finalPosition.y + wordHeight / 2);
  });
  
  // Add margin and ensure minimum size
  const margin = 40;
  const actualWidth = Math.max(width, maxX - minX + margin * 2);
  const actualHeight = Math.max(height, maxY - minY + margin * 2);
  
  // Center the layout in the container
  const offsetX = (actualWidth - (maxX - minX)) / 2 - minX;
  const offsetY = (actualHeight - (maxY - minY)) / 2 - minY;
  
  const centeredWords = placedWords.map(word => ({
    ...word,
    x: word.x + offsetX,
    y: word.y + offsetY
  }));
  
  return {
    words: centeredWords,
    actualWidth,
    actualHeight
  };
};

/**
 * Get color for word based on weight and position
 * @param {number} weight - Word weight (0-1)
 * @param {number} index - Word index for variation
 * @returns {string} CSS color
 */
const getWordColor = (weight, index) => {
  const colors = [
    '#3b82f6', // blue
    '#8b5cf6', // purple  
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
  ];
  
  const baseColor = colors[index % colors.length];
  const opacity = 0.6 + (0.4 * weight); // More frequent = more opaque
  
  return baseColor + Math.round(opacity * 255).toString(16).padStart(2, '0');
};

/**
 * Check if a word is semantically meaningful for lexical field analysis
 * @param {string} word - Word to check
 * @returns {boolean} True if word is meaningful
 */
export const isMeaningfulWord = (word) => {
  if (!word || word.length < 3) return false;
  if (STOP_WORDS.has(word.toLowerCase())) return false;
  
  // Exclude pure numbers, urls, etc.
  if (/^\d+$/.test(word)) return false;
  if (word.includes('http')) return false;
  if (word.includes('@')) return false;
  
  return true;
};

/**
 * Extract semantic word pairs for relationship analysis
 * @param {Array} selectedWords - Array of selected words
 * @returns {Array} Array of word pairs for commutator analysis
 */
export const generateWordPairs = (selectedWords) => {
  if (!selectedWords || selectedWords.length < 2) {
    return [];
  }
  
  const pairs = [];
  for (let i = 0; i < selectedWords.length; i++) {
    for (let j = i + 1; j < selectedWords.length; j++) {
      pairs.push([selectedWords[i], selectedWords[j]]);
    }
  }
  
  return pairs;
};