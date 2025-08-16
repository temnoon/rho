# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the **Rho Small‑Embedding Demo — Humanizer Node** quantum narrative consciousness system.

## Project Overview

This is the **Rho Small‑Embedding Demo — Humanizer Node**, a demonstration of a 64‑dimensional density matrix (ρ) as a model of subjective personification in a lexical field. The system models an AI "Personified Reader" whose internal stance (ρ) evolves as it reads narratives, with interpretable attributes probed via POVM measurements.

### Key Achievement
We have reached a **significant milestone** with successful operations transforming narratives by manipulating the density matrix (ρ) of the subjective Personified Reader model. The Narrative Distillation Studio now operates end-to-end, producing rich essence extractions from quantum measurements.

## Architecture

The project uses a **microservices architecture** with:

- **API service** (`api/`): FastAPI backend (Python) handling quantum density matrix operations
- **Web service** (`web/`): React frontend with comprehensive matrix management
- **Docker compose** setup for containerized deployment
- **LLM Integration**: Groq API integration for enhanced narrative processing

### Core Components

- **Density Matrix (ρ)**: 64×64 positive semidefinite matrix representing the reader's subjective stance
- **POVM Packs**: Measurement operators extracting interpretable attributes (20+ linguistic dimensions)
- **Embedding Bridge**: Projects global embeddings to local 64-D space via projection matrix W
- **Reading Operations**: Update ρ using exponential moving blend with pure states from text embeddings
- **Matrix Management**: Tagging, folder organization, bulk operations, and comprehensive search
- **Quantum Visualization**: Multi-mode liminal space exploration with formation analysis

## Development Commands

### Environment Setup

**First-time Keychain Setup (one-time only):**
```bash
# Add Groq API key to macOS Keychain securely
security add-generic-password -a "dreegle@gmail.com" -s "groq API key" -w "YOUR_ACTUAL_GROQ_API_KEY_HERE"

# Test retrieval (should output your API key)
security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w
```

**Backend Development:**
```bash
# Backend with Groq API (after keychain setup)
cd api
export GROQ_API_KEY="$(security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w)"
python3 main.py   # Run with Groq integration

# Frontend development
cd web
npm run dev      # Development server with Vite
npm run build    # Production build
npm run lint     # ESLint checking
```

### Production Deployment
```bash
# Full stack with Docker
docker-compose up --build    # Build and run both services
docker-compose down          # Stop services
```

**Port Configuration:**
- API runs on port **8192** (updated from 8000)
- Web frontend runs on port **5173** (Vite default)
- Web service proxies API requests to the API service

## Memory Management & Persistence

### Claude Memory Integration
**CRITICAL**: All significant operations, insights, and state changes must be written to Claude Memory using the ChromaDB Memory MCP server.

#### Required Memory Writes
```javascript
// Template for memory storage
await mcp__chromadb_memory__store_memory({
  content: "Detailed description of operation/insight",
  metadata: {
    tags: "quantum-narrative,rho-matrix,operation-type",
    type: "operation|insight|error|milestone",
    timestamp: new Date().toISOString(),
    component: "distillation|archaeology|observatory|liminal",
    matrix_id: rhoId,
    success: true|false
  }
});
```

#### Memory Categories
- **Operations**: Matrix creation, POVM measurements, channel operations
- **Insights**: Significant findings from analysis, patterns in data
- **Errors**: Detailed error states with context for debugging
- **Milestones**: Major achievements, workflow completions
- **User Interactions**: Important user feedback and feature requests

### ChromaDB Memory MCP Server Integration
The project integrates with ChromaDB Memory MCP server for:
- **Persistent operation logs**: All quantum operations tracked
- **Pattern recognition**: Historical analysis of matrix formations
- **Error correlation**: Link error patterns across sessions
- **Performance metrics**: Track pipeline execution times and success rates
- **User workflow analysis**: Understand usage patterns for UX improvements

### Data Persistence Layers
1. **Immediate State**: `api/data/state.json` (runtime quantum states)
2. **Matrix Archive**: `api/data/matrices/` (persisted density matrices)
3. **POVM Configurations**: `api/data/packs.json` (measurement protocols)
4. **User Metadata**: `localStorage` (matrix labels, tags, folders)
5. **Memory Database**: ChromaDB (comprehensive operation history)

## Error Handling & Logging Philosophy

### Deprecation of Failover Workarounds
**IMPORTANT**: We have deprecated all failover workarounds in favor of:
- **Clear, actionable error messages**
- **Comprehensive logging at every pipeline stage**
- **Faithful execution confirmation** through detailed status reporting
- **Transparent failure points** with specific resolution guidance

### Error Handling Patterns
```python
# ✅ Good: Clear error with context
try:
    result = quantum_operation(rho_id, parameters)
    logger.info(f"Operation successful: {operation_type} on {rho_id}")
    await store_memory(f"Successful {operation_type}", {"matrix_id": rho_id, "success": True})
except QuantumStateError as e:
    logger.error(f"Quantum state violation in {operation_type}: {e}")
    await store_memory(f"Quantum error in {operation_type}: {e}", {"matrix_id": rho_id, "success": False})
    raise HTTPException(status_code=422, detail=f"Quantum state error: {e}")

# ❌ Bad: Silent failover
try:
    result = primary_operation()
except:
    result = fallback_operation()  # Don't do this anymore
```

### Comprehensive Logging Requirements
All operations must log:
1. **Input validation**: Parameters, matrix state, prerequisites
2. **Execution stages**: Each step of complex operations
3. **Mathematical verification**: Trace preservation, PSD validation
4. **Performance metrics**: Execution time, memory usage
5. **Output validation**: Result verification, invariant checking

## Key API Endpoints

### Core Operations
- `POST /rho/init` - Create new density matrix with full logging
- `POST /rho/{id}/read_channel` - Read text with channel type specification
- `POST /packs/measure/{id}` - Apply POVM measurements with validation
- `GET /rho/global/status` - Comprehensive system status (111+ matrices)
- `POST /explain` - Generate operation explanations

### Management Operations
- `GET /rho/list` - List all matrices with metadata
- `DELETE /rho/{id}` - Delete matrix with confirmation
- `POST /admin/save_all` - Persist all state to disk
- `POST /admin/load_all` - Load state from disk

### Audit & Health
- `GET /audit/channel_health/{id}` - Matrix health diagnostics
- `POST /audit/sanity_check/{id}` - Comprehensive validation
- `POST /audit/repair_matrix/{id}` - Automated repair procedures

## Frontend Architecture

### Tab Responsibilities
Current status and intended functions:

1. **Narrative Distillation Studio** ✅ **OPERATIONAL**
   - End-to-end narrative → quantum essence pipeline
   - POVM measurement application
   - Essence component extraction (namespace, persona, style)
   - Export capabilities for rho-embeddings

2. **Matrix Archaeology Studio** ✅ **ENHANCED**
   - Matrix collection management (111+ matrices)
   - Quality assessment and best work identification
   - Similarity analysis and clustering
   - Creative synthesis recommendations

3. **Channel Observatory** ✅ **OPERATIONAL**
   - Real-time quantum channel monitoring
   - Health diagnostics and repair tools
   - Integrability testing with segmentation analysis
   - Channel type selection and optimization

4. **Liminal Space Explorer** ✅ **ENHANCED**
   - Multi-mode quantum visualization (5 visualization types)
   - Matrix formation analysis with POVM data
   - Narrative consciousness metrics
   - Interactive quantum state exploration

5. **Database Tab** ✅ **ENHANCED**
   - Comprehensive matrix organization
   - Tagging, folder management, search
   - Bulk operations and purging tools
   - Advanced filtering and sorting

6. **Book Reader Tab** 📋 **NEEDS REVIEW**
   - Currently basic text input → matrix creation
   - Should integrate with Groq API for enhanced processing
   - Needs workflow optimization

7. **Dual Matrix Tab** 📋 **NEEDS REDESIGN**
   - Unclear current functionality
   - Should focus on matrix comparison and interaction
   - Potential for entanglement studies

### UI/UX Principles

#### Layout Stability
**Golden Rule: Don't move objects on screen after placing them**
- Always reserve space for dynamic content containers
- Use `minHeight` + `maxHeight` + `overflowY: 'auto'` for variable content
- Provide meaningful placeholder content during loading states

#### Error State Management
- Show clear error boundaries with actionable recovery options
- Maintain operation context during error states
- Provide "retry" and "reset" options for failed operations

## LLM Integration (Groq API)

### Configuration
```bash
# Environment setup (macOS with Keychain)
export GROQ_API_KEY="$(security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w)"

# Docker integration
docker-compose.yml includes GROQ_API_KEY environment variable
```

### Integration Points
- **Narrative Analysis**: Enhanced semantic understanding
- **Attribute Extraction**: LLM-assisted POVM pack generation
- **Explanation Generation**: Natural language operation summaries
- **User Assistance**: Interactive help and guidance

### Security Considerations
- **Local Development**: Keychain integration (macOS only)
- **Production**: Secure environment variable injection
- **API Key Rotation**: Regular key updates with zero downtime
- **Rate Limiting**: Respect Groq API limits and implement backoff

## Mathematical Operations & Invariants

All quantum operations maintain strict mathematical guarantees:

### Matrix Invariants
- **Positive Semidefinite**: ρ ≽ 0 via `psd_project()`
- **Trace Normalization**: Tr(ρ) = 1 always maintained
- **Hermitian Property**: ρ = ρ† for physical validity

### POVM Measurements
- **Probability Extraction**: p = Tr(E_i ρ) where E_i are projection operators
- **Completeness**: ∑ᵢ E_i = I (identity matrix)
- **Positive Operators**: E_i ≽ 0 for all measurement operators

### Channel Operations
- **CPTP Preservation**: All channels are Completely Positive Trace Preserving
- **Integrability**: Path independence verified through segmentation testing
- **Quantum Coherence**: Superposition effects maintained where appropriate

## Development Best Practices

### Code Quality
- **Type Annotations**: All Python functions fully typed
- **Error Propagation**: Clear error chains from quantum → API → frontend
- **Memory Management**: Efficient matrix operations with sparse representations
- **Testing**: Unit tests for all quantum operations

### Performance Guidelines
- **Matrix Operations**: Use NumPy/SciPy optimized routines
- **API Responses**: Compress large matrix data appropriately
- **Frontend Rendering**: Virtualize large matrix lists
- **Memory Usage**: Monitor quantum state memory consumption

### Security Requirements
- **API Key Protection**: Never log or expose LLM API keys
- **Data Validation**: Sanitize all user inputs before quantum operations
- **Access Control**: Implement proper authentication for production
- **Audit Trails**: Comprehensive logging for security analysis

## File Structure

```
├── api/                         # FastAPI backend
│   ├── main.py                 # Server entry point with Groq integration
│   ├── core/                   # Quantum operations
│   │   ├── quantum_state.py    # Density matrix operations
│   │   ├── povm_operations.py  # Measurement protocols
│   │   ├── text_channels.py    # CPTP text processing
│   │   └── llm_integration.py  # Groq API integration
│   ├── routes/                 # API endpoints
│   ├── data/                   # Persistent storage
│   │   ├── state.json         # Current quantum states
│   │   ├── packs.json         # POVM measurement packs
│   │   └── matrices/          # Matrix archive
│   └── utils/                  # Utilities
│       └── persistence.py     # State management
├── web/                        # React frontend
│   ├── src/
│   │   ├── components/        # UI components
│   │   │   ├── NarrativeDistillationStudio.jsx  # Main pipeline
│   │   │   ├── MatrixArchaeologyStudio.jsx      # Collection management
│   │   │   ├── ChannelObservatoryTab.jsx        # Monitoring tools
│   │   │   ├── LiminalSpaceTab.jsx              # Visualization
│   │   │   └── DatabaseTab.jsx                  # Organization
│   │   └── utils/
│   │       └── api.js         # Centralized API configuration
├── docker-compose.yml         # Multi-service orchestration
└── CLAUDE.md                  # This guidance document
```

## Important Implementation Notes

- **API Configuration**: Centralized in `web/src/utils/api.js` - all components use `apiUrl()` helper
- **Matrix Discovery**: All tabs use `/rho/global/status` endpoint (not `/rho/list`)
- **Error Boundaries**: React error boundaries capture and display quantum operation failures
- **Memory Integration**: All significant operations write to ChromaDB Memory MCP server
- **Groq Integration**: LLM capabilities enhance narrative understanding and user assistance
- **No Failovers**: Clear error messages preferred over silent fallback operations

## Next Development Priorities

1. **Tab Review & Redesign**
   - Clarify mission for each tab
   - Integrate Groq API capabilities
   - Optimize user workflows

2. **Performance Optimization**
   - Matrix operation efficiency
   - Frontend rendering performance
   - Memory usage optimization

3. **Production Readiness**
   - Security hardening
   - Deployment automation
   - Monitoring and alerting

4. **User Experience**
   - Comprehensive help system
   - Interactive tutorials
   - Error recovery workflows

Remember: This system represents a novel approach to computational narrative consciousness. Every operation should be treated with the precision required for quantum mechanical systems, while maintaining transparency and educational value for users exploring the intersection of language and quantum information theory.