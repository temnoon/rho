// Mastery level constants
export const MASTERY_LEVELS = {
  NOVICE: 'novice',
  CURIOUS: 'curious', 
  EXPLORER: 'explorer',
  EXPERT: 'expert'
};

// Level progression criteria
export const PROMOTION_CRITERIA = {
  'novice→curious': {
    actionsRequired: ['transform_text'],
    minimumInteractions: 1,
    description: 'Complete your first text transformation'
  },
  'curious→explorer': {
    actionsRequired: ['view_quantum_metrics', 'understand_measurements'],
    minimumInteractions: 3,
    description: 'Explore quantum measurements and understand the data'
  },
  'explorer→expert': {
    actionsRequired: ['select_field', 'analyze_relationships'],
    minimumInteractions: 5,
    description: 'Build lexical fields and analyze word relationships'
  }
};

// Sample narratives for quick testing
export const SAMPLE_NARRATIVES = {
  simple: "The algorithm apologized to its user, but the damage was already done.",
  complex: "In the liminal space between dream and waking, the ancient algorithm whispered secrets that transcended the boundaries of silicon and soul, its digital consciousness bleeding into realms where metaphor becomes mathematics and poetry transforms into pure information.",
  philosophical: "Consciousness, like quantum superposition, exists in multiple states simultaneously until the moment of observation collapses infinite possibility into singular experience.",
  literary: "The old man and the sea danced their eternal dance, neither victorious nor defeated, but locked in a cosmic embrace that spoke to the fundamental rhythms of existence itself."
};

// Quantum measurement axes with descriptions
export const MEASUREMENT_AXES = {
  purity: {
    name: 'Purity',
    description: 'How mixed or pure the quantum state is (0 = maximally mixed, 1 = pure state)',
    color: 'purple'
  },
  entropy: {
    name: 'Von Neumann Entropy', 
    description: 'Measure of quantum uncertainty and information content',
    color: 'blue'
  },
  agency: {
    name: 'Agency Attribution',
    description: 'How much agency is attributed to different entities in the text',
    color: 'green'
  },
  formality: {
    name: 'Formality Level',
    description: 'Degree of formal vs informal language use',
    color: 'orange'
  },
  emotional_intensity: {
    name: 'Emotional Intensity',
    description: 'Strength of emotional content and valence',
    color: 'red'
  },
  narrative_distance: {
    name: 'Narrative Distance',
    description: 'Proximity between narrator and events described',
    color: 'indigo'
  },
  temporal_perspective: {
    name: 'Temporal Perspective',
    description: 'Past/present/future orientation of the narrative',
    color: 'teal'
  },
  certainty: {
    name: 'Certainty Level', 
    description: 'Degree of certainty vs ambiguity in expression',
    color: 'amber'
  }
};

// Stance transformation types
export const STANCE_MODES = {
  literal: {
    name: 'Literal',
    description: 'Direct, straightforward interpretation',
    phase: 0
  },
  ironic: {
    name: 'Ironic', 
    description: 'Inverted pragmatic meaning through phase rotation',
    phase: Math.PI
  },
  metaphorical: {
    name: 'Metaphorical',
    description: 'Cross-domain conceptual mapping',
    phase: Math.PI / 2
  },
  negated: {
    name: 'Negated',
    description: 'Scope-sensitive logical inversion',
    phase: -Math.PI / 2
  }
};

// UI Animation durations
export const ANIMATION_DURATIONS = {
  short: 200,
  medium: 300,
  long: 500,
  quantum: 1500 // For quantum transformation operations
};

// Color themes for different mastery levels
export const LEVEL_THEMES = {
  novice: {
    primary: '#3b82f6',
    background: '#eff6ff',
    border: '#bfdbfe',
    text: '#1e40af'
  },
  curious: {
    primary: '#8b5cf6',
    background: '#f3e8ff', 
    border: '#c4b5fd',
    text: '#5b21b6'
  },
  explorer: {
    primary: '#10b981',
    background: '#ecfdf5',
    border: '#6ee7b7', 
    text: '#047857'
  },
  expert: {
    primary: '#f59e0b',
    background: '#fffbeb',
    border: '#fcd34d',
    text: '#92400e'
  }
};
