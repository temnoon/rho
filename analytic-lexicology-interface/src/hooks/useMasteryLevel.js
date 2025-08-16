import { useState, useEffect, useCallback } from 'react';
import { MASTERY_LEVELS, PROMOTION_CRITERIA } from '../utils/constants.js';

export const useMasteryLevel = () => {
  const [masteryLevel, setMasteryLevel] = useState(MASTERY_LEVELS.NOVICE);
  const [completedActions, setCompletedActions] = useState([]);
  const [interactionCount, setInteractionCount] = useState(0);
  const [showUpgradePrompt, setShowUpgradePrompt] = useState(false);

  // Check if user can be promoted to next level
  const checkForPromotion = useCallback(() => {
    const currentLevels = Object.values(MASTERY_LEVELS);
    const currentIndex = currentLevels.indexOf(masteryLevel);
    
    if (currentIndex >= currentLevels.length - 1) return; // Already at max level
    
    const nextLevel = currentLevels[currentIndex + 1];
    const promotionKey = `${masteryLevel}→${nextLevel}`;
    const criteria = PROMOTION_CRITERIA[promotionKey];
    
    if (!criteria) return;
    
    // Check if all required actions are completed
    const hasRequiredActions = criteria.actionsRequired.every(action => 
      completedActions.includes(action)
    );
    
    // Check if minimum interactions met
    const hasMinimumInteractions = interactionCount >= criteria.minimumInteractions;
    
    if (hasRequiredActions && hasMinimumInteractions) {
      setShowUpgradePrompt(true);
    }
  }, [masteryLevel, completedActions, interactionCount]);

  // Track when user completes an action
  const completeAction = useCallback((actionName) => {
    setCompletedActions(prev => {
      if (!prev.includes(actionName)) {
        return [...prev, actionName];
      }
      return prev;
    });
    
    setInteractionCount(prev => prev + 1);
  }, []);

  // Promote user to next level
  const promoteUser = useCallback(() => {
    const currentLevels = Object.values(MASTERY_LEVELS);
    const currentIndex = currentLevels.indexOf(masteryLevel);
    
    if (currentIndex < currentLevels.length - 1) {
      const nextLevel = currentLevels[currentIndex + 1];
      setMasteryLevel(nextLevel);
      setShowUpgradePrompt(false);
      
      // Track promotion in local storage for persistence
      localStorage.setItem('analytic-lexicology-level', nextLevel);
    }
  }, [masteryLevel]);

  // Dismiss upgrade prompt without promoting
  const dismissUpgradePrompt = useCallback(() => {
    setShowUpgradePrompt(false);
  }, []);

  // Get next level info
  const getNextLevelInfo = useCallback(() => {
    const currentLevels = Object.values(MASTERY_LEVELS);
    const currentIndex = currentLevels.indexOf(masteryLevel);
    
    if (currentIndex >= currentLevels.length - 1) return null;
    
    const nextLevel = currentLevels[currentIndex + 1];
    const promotionKey = `${masteryLevel}→${nextLevel}`;
    const criteria = PROMOTION_CRITERIA[promotionKey];
    
    return {
      level: nextLevel,
      criteria,
      progress: {
        actions: criteria.actionsRequired.map(action => ({
          name: action,
          completed: completedActions.includes(action)
        })),
        interactions: {
          current: interactionCount,
          required: criteria.minimumInteractions
        }
      }
    };
  }, [masteryLevel, completedActions, interactionCount]);

  // Load saved level from localStorage on mount
  useEffect(() => {
    const savedLevel = localStorage.getItem('analytic-lexicology-level');
    if (savedLevel && Object.values(MASTERY_LEVELS).includes(savedLevel)) {
      setMasteryLevel(savedLevel);
    }
  }, []);

  // Check for promotion whenever state changes
  useEffect(() => {
    checkForPromotion();
  }, [checkForPromotion]);

  return {
    masteryLevel,
    completedActions,
    interactionCount,
    showUpgradePrompt,
    completeAction,
    promoteUser,
    dismissUpgradePrompt,
    getNextLevelInfo,
    // Convenience methods
    isNovice: masteryLevel === MASTERY_LEVELS.NOVICE,
    isCurious: masteryLevel === MASTERY_LEVELS.CURIOUS,
    isExplorer: masteryLevel === MASTERY_LEVELS.EXPLORER,
    isExpert: masteryLevel === MASTERY_LEVELS.EXPERT,
    // Level checking helpers
    canAccess: (requiredLevel) => {
      const levels = Object.values(MASTERY_LEVELS);
      const currentIndex = levels.indexOf(masteryLevel);
      const requiredIndex = levels.indexOf(requiredLevel);
      return currentIndex >= requiredIndex;
    }
  };
};
