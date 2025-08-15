# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Rho Small‑Embedding Demo — Humanizer Node**, a demonstration of a 64‑dimensional density matrix (ρ) as a model of subjective personification in a lexical field. The system models an AI "Personified Reader" whose internal stance (ρ) evolves as it reads narratives, with interpretable attributes probed via POVM measurements.

## Architecture

The project uses a **microservices architecture** with:

- **API service** (`api/`): FastAPI backend (Python) handling the quantum density matrix operations
- **Web service** (`web/`): React frontend with vanilla JS (no additional UI libraries)
- **Docker compose** setup for containerized deployment

### Core Components

- **Density Matrix (ρ)**: 64×64 positive semidefinite matrix representing the reader's subjective stance
- **POVM Packs**: Measurement operators that extract interpretable attributes (narrator distance, reliability, affect, etc.)
- **Embedding Bridge**: Projects global embeddings to local 64-D space via projection matrix W
- **Reading Operations**: Update ρ using exponential moving blend with pure states from text embeddings

## Development Commands

### Frontend (web/)
```bash
cd web
npm run dev      # Development server with Vite
npm run build    # Production build
npm run lint     # ESLint checking
```

### Backend (api/)
```bash
cd api
python main.py   # Run FastAPI server directly
# or
uvicorn main:app --reload --port 8000
```

### Full Stack (Docker)
```bash
docker-compose up --build    # Build and run both services
docker-compose down          # Stop services
```

- API runs on port 8000
- Web frontend runs on port 8080
- Web service proxies `/api/*` requests to the API service

## Key API Endpoints

- `POST /rho/init` - Create new density matrix
- `POST /rho/{id}/read` - Read text and update ρ with blending parameter α
- `POST /rho/{id}/measure` - Apply POVM pack measurements
- `GET /packs` - List available measurement packs
- `POST /explain` - Get explanations of recent operations

## Data Persistence

- State and POVM packs persist in `api/data/` (bind mounted in Docker)
- Use `/api/admin/save_all` and `/api/admin/load_all` for manual persistence
- Demo includes pre-configured POVM packs in `api/data/packs.json`

## Mathematical Operations

All matrix operations maintain these invariants:
- ρ is positive semidefinite via `psd_project()` 
- Trace normalization: Tr(ρ) = 1
- POVM measurements: p = Tr(E_i ρ) where E_i are projection operators

## Important Implementation Notes

- Embedding dimension is configurable via `EMBED_URL` environment variable
- Projection matrix W handles dimension mismatches between global embeddings and 64-D local space
- Frontend uses dependency-free React implementation for minimal bundle size
- All operations show transparent input → math → output pipeline for educational purposes

## UI/UX Best Practices

### CSS Layout Stability
**Golden Rule: Don't move objects on screen after placing them**

- Always reserve space for dynamic content containers to prevent layout jumping
- Use `minHeight` + `maxHeight` + `overflowY: 'auto'` for variable content
- Provide placeholder content when dynamic content is empty
- Fixed-height containers prevent Cumulative Layout Shift (CLS)

Example implementation:
```jsx
// ❌ Bad: Conditional rendering causes layout jumping
{data && <div>{data}</div>}

// ✅ Good: Reserved space with overflow handling
<div style={{ minHeight: 60, maxHeight: 120, overflowY: 'auto' }}>
  {data || <span style={{ color: '#999' }}>Loading...</span>}
</div>
```

### Content Overflow Management
- Long dynamic content should scroll within fixed boundaries
- Use visual indicators (borders, backgrounds) to show content state changes
- Maintain consistent spacing regardless of content presence

## File Structure

```
├── api/
│   ├── main.py              # FastAPI backend implementation
│   ├── requirements.txt     # Python dependencies
│   └── data/               # Persistent data directory
│       └── packs.json      # POVM measurement packs
├── web/
│   ├── package.json        # Node.js dependencies and scripts
│   └── src/
│       └── main.jsx        # React frontend (single file)
└── docker-compose.yml      # Multi-service orchestration
```
- Please look over this project Rho. We have reached a significant milestone with the successful operations of transforming narratives by manipulating the density matrix (rho) of the subjective Personified Reader model. We need to review each tab, and decide what it should be doing (none of the other tabs have as clear a mission. Several don't quite do what they were designed for) and I want to integrate Groq.com LLM server into this docker container, so it would operate as well locally or in the cloud. It is impractical to rent a droplet beefy enough to run a very slow ollama model, so I want to get that running now. I now have a developer API key, I want to install it. Let's do that first, please. 

The Groq API key is saved in my apple keychain (manually) with name "groq API key", user "dreegle@gmail.com", password field is the API key. This has been working fine for my database keys, and was recently working in the various /Users/tem/humanizer-lighthouse/ projects. This IS only going to work on my mac, we'll figure out the security on the remote server later. You can advise me on best practices when the time comes. For now, I want to see how groq performs, and then start reviewing the tabs, and reimagining a prototype.