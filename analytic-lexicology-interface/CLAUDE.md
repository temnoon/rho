# CLAUDE.md - Analytic Lexicology Interface

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the **Analytic Lexicology Interface**, a progressive disclosure tutorial system for quantum narrative consciousness built on the Rho Small-Embedding Demo foundation.

## Project Overview

The **Analytic Lexicology Interface** is a standalone educational GUI designed to introduce users to quantum narrative transformation through **progressive disclosure**. Unlike the comprehensive Rho interface, this system starts simple and gradually reveals advanced capabilities as users demonstrate mastery.

### Key Philosophy
- **Learning by Doing**: Theory emerges through practice, not documentation
- **Progressive Disclosure**: Users see only what they're ready for
- **Mastery-Based Progression**: Advancement through demonstrated competency
- **Non-Overwhelming**: Complex quantum concepts introduced gently

### Educational Journey
1. **Novice 🌱**: "Magic" text transformations (enhanced/subdued)
2. **Curious 🔍**: Quantum state metrics and measurement visualization
3. **Explorer 🗺️**: Lexical field analysis and word relationships
4. **Expert ⚗️**: Full stance control with phase rotations and research tools

## Architecture

### Technology Stack
- **Frontend**: React 18 + Vite + Tailwind CSS
- **State Management**: Context API with custom hooks
- **Backend Integration**: Same Rho API (port 8192)
- **Port**: 5174 (separate from main web interface)
- **Icons**: Lucide React

### Core Design Patterns

#### Progressive Disclosure System
The interface reveals functionality based on **mastery levels** rather than user preferences:

```javascript
// Level-based feature access
canShowAdvancedFeatures: masteryLevel >= 'curious',
canShowFieldAnalysis: masteryLevel >= 'explorer', 
canShowStanceControls: masteryLevel >= 'expert'
```

#### Mastery Progression Criteria
Users advance through **action completion** and **interaction thresholds**:

```javascript
'novice→curious': {
  actionsRequired: ['transform_text'],
  minimumInteractions: 1,
  description: 'Complete your first text transformation'
},
'curious→explorer': {
  actionsRequired: ['view_quantum_metrics', 'understand_measurements'], 
  minimumInteractions: 3,
  description: 'Explore quantum measurements and understand the data'
}
```

#### Context-Based State Management
Single context provider manages all application state:

```javascript
<AnalyticLexicologyProvider>
  <ProgressiveInterface />
</AnalyticLexicologyProvider>
```

## Development Commands

### Environment Setup
```bash
# Navigate to interface directory
cd rho/analytic-lexicology-interface

# Install dependencies (first time only)
npm install

# Start development server
npm run dev  # Runs on http://localhost:5174
```

### Prerequisites
**CRITICAL**: The Rho API backend must be running:
```bash
# In separate terminal - start Rho API
cd rho/api
export GROQ_API_KEY="$(security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w)"
python3 main.py  # Must be on port 8192
```

### Build Commands
```bash
npm run build    # Production build
npm run preview  # Preview production build  
npm run lint     # ESLint checking
```

## Component Architecture

### Level-Based Views
Each mastery level has a dedicated view component:

1. **`NoviceView.jsx`** - Simple enhanced/subdued transformations
2. **`CuriousView.jsx`** - Quantum metrics and measurement axes
3. **`ExplorerView.jsx`** - Lexical field selection and relationship analysis
4. **`ExpertView.jsx`** - Full stance controls and research capabilities

### Core Components
- **`ProgressiveInterface.jsx`** - Main orchestrator, renders appropriate view
- **`AnalyticLexicologyProvider`** - Context provider with all state
- **Mastery System Hooks** - `useMasteryLevel.js`, `useQuantumAPI.js`

### State Management Layers
1. **Local State**: Component-specific UI state
2. **Context State**: Shared application state via Context API
3. **Persistent State**: Mastery level saved to localStorage
4. **API State**: Quantum operations via Rho backend
5. **Memory Integration**: ChromaDB Memory MCP server for operation logging

## Mastery Level System

### Level Definitions
```javascript
MASTERY_LEVELS = {
  NOVICE: 'novice',     // Simple transformations, no technical terms
  CURIOUS: 'curious',   // Quantum metrics, measurement introduction  
  EXPLORER: 'explorer', // Lexical fields, word relationships
  EXPERT: 'expert'      // Full stance control, research tools
}
```

### Promotion Mechanics
- **Action Tracking**: `completeAction('transform_text')` calls tracked
- **Interaction Counting**: Minimum interaction thresholds required
- **Upgrade Prompts**: Contextual invitations to advance levels
- **Persistent Progress**: Level saved to localStorage across sessions

### Level-Specific Features

#### Novice Level 🌱
- **Goal**: Experience the "magic" without technical overhead
- **Features**: Enhanced (170%) and subdued (70%) transformations
- **Language**: No quantum terminology, focus on results
- **Promotion**: Complete one text transformation

#### Curious Level 🔍  
- **Goal**: Understand what's happening "under the hood"
- **Features**: Purity, entropy, measurement axes visualization
- **Language**: Introduce basic quantum concepts with explanations
- **Promotion**: View metrics and demonstrate understanding

#### Explorer Level 🗺️
- **Goal**: Build lexical fields and analyze relationships
- **Features**: Word selection, field topology, commutator analysis
- **Language**: Lexical field theory, semantic relationships
- **Promotion**: Select fields and analyze word relationships

#### Expert Level ⚗️
- **Goal**: Full research-grade control over quantum transformations  
- **Features**: Stance modes (ironic, metaphorical, negated), phase rotation
- **Language**: Full technical terminology, mathematical precision
- **Promotion**: Already at maximum level

## API Integration

### Backend Dependency
**CRITICAL**: This interface requires the main Rho API backend running on port 8192.

### API Endpoints Used
- **Novice**: `POST /rho/init`, `POST /rho/{id}/read_channel`
- **Curious**: + `POST /packs/measure/{id}`, `GET /audit/channel_health/{id}`
- **Explorer**: + `POST /aplg/field_analysis`, `/aplg/commutator_analysis`  
- **Expert**: + `POST /aplg/stance_transformation`, full research endpoints

### API Configuration
Centralized in `src/utils/api.js`:
```javascript
const API_BASE = 'http://localhost:8192';
export const apiUrl = (path) => `${API_BASE}${path}`;
```

## UI/UX Design Principles

### Progressive Revelation
- **Information Architecture**: Reveal complexity gradually
- **Cognitive Load**: Never overwhelm users with too much at once
- **Contextual Help**: Explanations appear when needed, not before

### Level-Appropriate Language
```javascript
// Novice: "Enhanced" and "Subdued" transformations
// Curious: "Quantum state analysis" with gentle explanations  
// Explorer: "Lexical field topology" and "semantic relationships"
// Expert: Full mathematical terminology and research precision
```

### Visual Hierarchy
- **Level Badges**: Clear indication of current mastery level
- **Upgrade Prompts**: Attractive invitations to advance
- **Color Themes**: Each level has distinct visual identity
- **Animation**: Smooth transitions between states

### Layout Stability
**Golden Rule**: Don't move UI elements after placing them
- Reserve space for dynamic content with `minHeight`
- Use containers with `overflowY: 'auto'` for variable content
- Provide meaningful loading states and placeholders

## Memory Management & Persistence

### Required Memory Writes
All significant operations must be logged to ChromaDB Memory:

```javascript
await store_memory({
  content: "User advanced to curious level after completing first transformation",
  metadata: {
    tags: "mastery-progression,analytic-lexicology,tutorial",
    type: "progression", 
    from_level: "novice",
    to_level: "curious",
    actions_completed: ["transform_text"],
    interaction_count: 3
  }
});
```

### Memory Categories for This Interface
- **Progression**: Level advancements and mastery milestones
- **Learning**: User comprehension of quantum concepts
- **Interaction**: Usage patterns and feature adoption
- **Errors**: Tutorial flow issues and user confusion points
- **Success**: Successful operation completions by level

### Persistence Layers
1. **Session State**: React Context (ephemeral)
2. **Local Storage**: Mastery level and progress (browser persistent)
3. **API Backend**: Quantum states and transformations (server persistent)
4. **Memory Database**: Complete learning journey tracking (ChromaDB)

## Error Handling Philosophy

### Tutorial-Friendly Error Messages
Unlike the research-grade main interface, errors here must be **educational**:

```javascript
// ✅ Good: Educational error for tutorial context
"Let's try a different approach! The quantum state needs more complexity. Try adding more descriptive words or emotional content."

// ❌ Bad: Technical error (appropriate for main interface, not tutorial)
"QuantumStateError: Insufficient rank for PSD projection in 64-dimensional space"
```

### Error Recovery Patterns
- **Gentle Guidance**: Suggest specific improvements rather than technical fixes
- **Context Preservation**: Maintain user's work during error states
- **Learning Opportunities**: Turn errors into teaching moments

## Sample Narratives & Test Cases

### Built-in Samples
```javascript
SAMPLE_NARRATIVES = {
  simple: "The algorithm apologized to its user, but the damage was already done.",
  complex: "In the liminal space between dream and waking...",
  philosophical: "Consciousness, like quantum superposition...",
  literary: "The old man and the sea danced their eternal dance..."
}
```

### Testing Progression
1. **Novice Testing**: Verify simple transformations work without technical exposure
2. **Curious Testing**: Ensure quantum metrics appear with clear explanations
3. **Explorer Testing**: Test lexical field selection and analysis features
4. **Expert Testing**: Verify full stance controls and research capabilities

## File Structure

```
analytic-lexicology-interface/
├── src/
│   ├── components/
│   │   ├── core/
│   │   │   └── ProgressiveInterface.jsx    # Main orchestrator
│   │   └── levels/
│   │       ├── NoviceView.jsx             # Simple transformations
│   │       ├── CuriousView.jsx            # Quantum metrics intro
│   │       ├── ExplorerView.jsx           # Lexical field analysis  
│   │       └── ExpertView.jsx             # Full stance controls
│   ├── hooks/
│   │   ├── useMasteryLevel.js             # Progression system
│   │   └── useQuantumAPI.js               # Backend integration
│   ├── utils/
│   │   ├── constants.js                   # Levels, criteria, samples
│   │   └── api.js                         # Centralized API config
│   ├── App.jsx                            # Context provider setup
│   ├── main.jsx                           # React entry point
│   └── index.css                          # Tailwind imports
├── package.json                           # Dependencies & scripts
├── tailwind.config.js                     # Styling configuration
├── vite.config.js                         # Build configuration
└── CLAUDE.md                              # This documentation
```

## Integration with Main Rho System

### Relationship to Main Interface
- **Complementary**: Tutorial introduction to main research interface
- **Shared Backend**: Same API endpoints, different presentation
- **Different Audiences**: Tutorial (newcomers) vs Research (experts)
- **Port Separation**: 5174 vs 5173 to avoid conflicts

### Transition Path
Users graduate from this tutorial interface to the main Rho interface:
1. Complete expert level mastery
2. Export quantum states for import into main system
3. Seamless continuation of quantum narrative work

### API Compatibility
All quantum states created in this interface are fully compatible with:
- Matrix Archaeology Studio (main interface)
- Narrative Distillation Studio 
- Channel Observatory
- Liminal Space Explorer

## Development Best Practices

### Tutorial-Specific Guidelines
- **Language Evolution**: Terminology complexity should match mastery level
- **Concept Introduction**: New ideas through hands-on experience first
- **Error Messaging**: Educational rather than technical
- **UI Affordances**: Clear indicators of what's possible at each level

### State Management
- **Context Over Props**: Use Context API for cross-component state
- **Hook Separation**: Separate mastery logic from quantum API logic
- **Action Tracking**: Consistent `completeAction()` calls for progression
- **Persistence**: Save mastery level to localStorage

### Performance Considerations
- **Lazy Loading**: Load advanced components only when levels unlock
- **API Efficiency**: Cache quantum states appropriately
- **Memory Usage**: Clean up quantum matrices after operations

## Testing & Quality Assurance

### User Journey Testing
1. **Fresh User Path**: Test complete novice→expert progression
2. **Returning User**: Verify localStorage persistence works
3. **Error Recovery**: Test error states at each mastery level
4. **API Failures**: Graceful degradation when backend unavailable

### Cross-Level Consistency
- Ensure quantum states work consistently across all mastery views
- Verify all level transitions preserve user context
- Test that advancement criteria trigger correctly

## Security & API Key Management

### Development Environment
```bash
# Use macOS Keychain for secure API key storage
export GROQ_API_KEY="$(security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w)"
```

### Production Deployment
- Same security model as main Rho system
- Environment variable injection for containers
- No API keys in client-side code

## Deployment Options

### Development
```bash
npm run dev  # Vite dev server on 5174
```

### Production
```bash
npm run build
npx serve dist -p 5174
```

### Docker Integration
Add to main `docker-compose.yml`:
```yaml
analytic-lexicology:
  build:
    context: ./analytic-lexicology-interface
  ports:
    - "5174:5174"
  depends_on:
    - api
```

## Next Development Priorities

1. **Advanced Tutorials**: Add guided tours for complex concepts
2. **Theory Integration**: Link to formal Analytic Lexicology documentation
3. **Export Capabilities**: Allow quantum state export to main interface
4. **Analytics**: Track learning effectiveness and common confusion points
5. **Accessibility**: Full screen reader and keyboard navigation support

## Important Implementation Notes

- **API Dependency**: Cannot function without main Rho API backend
- **Port Configuration**: Must not conflict with main web interface (5173)
- **Memory Integration**: All mastery progression logged to ChromaDB
- **Educational Focus**: Prioritize learning experience over feature completeness
- **Progressive Enhancement**: Each level builds on previous understanding

Remember: This interface serves as the **gentle introduction** to quantum narrative consciousness. Every design decision should prioritize the user's learning journey over technical completeness. The goal is to create "quantum narrative natives" who understand these concepts intuitively before encountering the full research interface.