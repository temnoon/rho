# Analytic Lexicology Interface - Setup & Deployment Guide

This directory contains the standalone Analytic Lexicology Interface - a progressive disclosure tutorial for quantum narrative transformation.

## Quick Start

```bash
# Install dependencies
cd /Users/tem/rho/analytic-lexicology-interface
npm install

# Start development server
npm run dev

# Interface will be available at http://localhost:5174
```

## Prerequisites

1. **Rho API Backend**: Must be running on port 8192
   ```bash
   cd /Users/tem/rho/api
   export GROQ_API_KEY="$(security find-generic-password -a 'dreegle@gmail.com' -s 'groq API key' -w)"
   python3 main.py
   ```

2. **Node.js**: Version 16+ required for Vite

## Development Commands

```bash
npm run dev      # Development server (port 5174)
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # ESLint checking
```

## Interface Overview

The interface uses **progressive disclosure** to introduce users to Analytic Lexicology:

### Level 1: Novice 🌱
- Simple text transformation (enhanced/subdued)
- No technical terminology
- Focus on experiencing the "magic"

### Level 2: Curious 🔍
- Quantum state metrics (purity, entropy)
- Measurement axes visualization
- Introduction to quantum concepts

### Level 3: Explorer 🗺️
- Lexical field selection and analysis
- Word relationship visualization
- Commutator analysis introduction

### Level 4: Expert ⚗️
- Full stance control (irony, metaphor, negation)
- Phase rotation and complex analysis
- Research-grade export capabilities

## API Integration

The interface connects to your existing Rho backend:

- **Novice**: `/rho/init` + `/rho/{id}/read_channel`
- **Curious**: + `/packs/measure/{id}` + diagnostics
- **Explorer**: + `/aplg/field_analysis` + commutators
- **Expert**: + `/aplg/stance_transformation` + full analysis

## Configuration

### Port Configuration
- Interface: `5174` (separate from main web interface on 5173)
- API Backend: `8192` (configured in `src/utils/api.js`)

### Customization
- Modify `src/utils/constants.js` for sample narratives and themes
- Adjust progression criteria in `src/hooks/useMasteryLevel.js`
- Customize API endpoints in `src/utils/api.js`

## Production Deployment

```bash
# Build for production
npm run build

# Serve with static file server
npx serve dist -p 5174
```

For Docker deployment, add to main `docker-compose.yml`:

```yaml
analytic-lexicology:
  build:
    context: ./analytic-lexicology-interface
    dockerfile: Dockerfile
  ports:
    - "5174:5174"
  depends_on:
    - api
```

## Features

### Progressive Mastery System
- Automatic level advancement based on user interaction
- Persistent progress via localStorage
- Contextual upgrade prompts

### Quantum Integration
- Real-time quantum state analysis
- POVM measurement visualization
- Field topology analysis
- Stance transformation controls

### Educational Design
- Theory introduction through practice
- Contextual explanations
- No overwhelming technical details initially

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure Rho backend is running on port 8192
   - Check CORS configuration in main API

2. **Styles Not Loading**
   - Verify Tailwind CSS installation: `npm list tailwindcss`
   - Check `tailwind.config.js` content paths

3. **Component Import Errors**
   - Ensure all component files exist in correct directories
   - Check relative import paths

### Development Tips

- Use browser dev tools to monitor API calls
- Check console for React errors and warnings  
- Test progression through all mastery levels
- Verify quantum state persistence between operations

## Architecture Notes

This interface is designed as a **standalone introduction** to Analytic Lexicology that:

1. **Complements** the main Rho interface (doesn't replace it)
2. **Teaches** through progressive disclosure rather than documentation
3. **Scales** from simple demo to research tool
4. **Integrates** seamlessly with existing quantum backend

The progressive disclosure ensures users are never overwhelmed while providing a clear path to advanced capabilities.
