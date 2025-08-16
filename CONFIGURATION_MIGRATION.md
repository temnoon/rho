# Configuration Migration Guide

## Issues Identified

The Rho project had several configuration management issues:

1. **Port Confusion**: Docker exposes internal port 8000 as external 8192, but hardcoded 8192 throughout codebase
2. **Scattered Hardcoded Values**: URLs, ports, and constants spread across 36+ files  
3. **No LLM Prompt Management**: Prompts embedded in code without versioning or usage tracking
4. **Docker Health Check Failure**: Missing curl in container causing unhealthy status

## Solutions Implemented

### 1. Centralized Configuration System

Created `/config/centralized_config.py` with:

- **Unified Service Configuration**: All ports, URLs, and endpoints in one place
- **LLM Provider Management**: Configurable models, API keys, and parameters  
- **Prompt Template System**: Versioned prompts with usage tracking
- **Environment Override Support**: Environment variables take precedence
- **Path Management**: Centralized file and directory path handling

### 2. Fixed Docker Issues

**Docker Compose Health Check** (Fixed):
```yaml
# Before (broken - curl not available)
test: ["CMD-SHELL", "curl -fsS http://localhost:8000/healthz || exit 1"]

# After (working - uses Python requests)  
test: ["CMD-SHELL", "python3 -c \"import requests; requests.get('http://localhost:8000/healthz', timeout=5)\" || exit 1"]
```

**Port Mapping Clarification**:
- Container runs on internal port **8000**
- Docker exposes it as external port **8192**
- Applications should use **8192** for external access
- Use centralized config to manage this mapping

### 3. Frontend Configuration Updates

**Analytic Lexicology Interface**:
- Created `src/utils/centralizedConfig.js` 
- Replaced hardcoded `http://localhost:8192` with `getApiUrl()`
- Added environment variable support: `VITE_API_URL`

**Main Web Interface**:
- Updated to use `VITE_API_URL` environment variable
- Maintains backward compatibility with hardcoded fallback

## Migration Steps

### For Developers

1. **Use Centralized Config**:
   ```python
   # Before
   API_URL = "http://localhost:8192"
   
   # After  
   from config.centralized_config import config
   API_URL = config.get_api_url()
   ```

2. **Environment Variables**:
   ```bash
   # Development
   export VITE_API_URL="http://localhost:8192"
   export RHO_API_PORT=8192
   export GROQ_API_KEY="your-key-here"
   
   # Production
   export VITE_API_URL="https://your-domain.com"
   export RHO_API_PORT=443
   ```

3. **LLM Prompt Updates**:
   ```python
   # Before
   prompt = f"Transform this text: {text}"
   
   # After
   from config.centralized_config import config
   prompt = config.render_prompt('quantum_narrative_transformer', 
                                original_text=text, 
                                guidance_text=guidance)
   config.log_prompt_usage('quantum_narrative_transformer', 
                          {'original_text': text}, response)
   ```

### For Docker Deployment

1. **Environment File** (`.env`):
   ```bash
   GROQ_API_KEY=your-groq-key
   RHO_API_PORT=8192
   RHO_WEB_PORT=5173
   RHO_DIMENSION=64
   ```

2. **Docker Compose Updates**:
   ```yaml
   services:
     api:
       environment:
         - GROQ_API_KEY=${GROQ_API_KEY}
         - RHO_API_PORT=${RHO_API_PORT:-8192}
   ```

### For Production

1. **Load Balancer Configuration**:
   - Route `/api/*` to API service (port 8192)
   - Route `/*` to Web service (port 5173)
   - Route `/analytic-lexicology/*` to tutorial interface (port 5174)

2. **Environment Configuration**:
   ```bash
   # API service
   export RHO_API_PORT=8000  # Internal port
   export GROQ_API_KEY=$(security find-generic-password ...)
   
   # Web services  
   export VITE_API_URL="https://api.your-domain.com"
   ```

## Configuration File Structure

```
/config/
├── centralized_config.py      # Main Python configuration  
├── rho_config.json           # JSON configuration data
├── prompt_templates.json     # LLM prompt templates
└── README.md                # Configuration documentation

/analytic-lexicology-interface/src/utils/
├── centralizedConfig.js      # Frontend configuration
└── api.js                   # Updated API client

/.env                        # Environment variables (development)
/docker-compose.yml          # Container configuration
```

## Benefits Achieved

1. **Single Source of Truth**: All configuration in `/config/` directory
2. **Environment Flexibility**: Easy dev/staging/production configuration
3. **Prompt Management**: Version control and usage analytics for LLM prompts
4. **Docker Health**: Fixed container health checks
5. **Maintainability**: No more hunting for hardcoded values
6. **Consistency**: Unified configuration access patterns

## Remaining Work

### High Priority
- [ ] Update remaining hardcoded values in 30+ files identified
- [ ] Create production deployment configuration templates
- [ ] Add configuration validation on startup

### Medium Priority  
- [ ] Web UI for configuration management
- [ ] Prompt template editor interface
- [ ] Configuration change audit logging

### Future Enhancements
- [ ] Dynamic configuration reloading
- [ ] A/B testing for prompt templates
- [ ] Configuration backup and restore
- [ ] Integration with secrets management systems

## Testing the Changes

1. **Restart Docker Services**:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

2. **Verify Health Checks**:
   ```bash
   docker ps  # Should show "healthy" status
   curl http://localhost:8192/healthz  # Should return {"ok":true,...}
   ```

3. **Test Both Interfaces**:
   - Main interface: http://localhost:5173
   - Tutorial interface: http://localhost:5174 
   - Both should connect to API without hardcoded URLs

4. **Configuration Validation**:
   ```python
   from config.centralized_config import config
   print(config.to_dict())  # Verify all settings
   ```

This migration eliminates configuration fragmentation and provides a foundation for scalable deployment across environments.