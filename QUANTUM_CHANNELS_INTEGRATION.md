# Quantum Channels Integration Guide

## Overview: From Convex Combinations to Proper Channels

This document outlines the transition from the current **convex combination** approach to proper **quantum channel theory** as requested in the "Analytic Post-Lexical Grammatology" framework.

## Current Anti-Patterns Identified

### ❌ **Problem 1: Non-CPTP Text Updates**
```python
# Current implementation in quantum_state.py:131
rho_new = (1.0 - alpha) * rho_old + alpha * pure  # NOT a channel!
```

**Issue**: This is a convex combination, not a quantum channel. It violates:
- **Composability**: Cannot safely compose text operations
- **Auditability**: No stable channel object to log/analyze
- **Integrability**: Different segmentations may yield different results

### ✅ **Solution: Proper Kraus Channels**
```python
# New implementation using text_channels.py
text_channel = text_to_channel(text_embedding, alpha, "rank_one_update")
rho_new = text_channel.apply(rho_old)  # Φ(ρ) with CPTP guarantees
```

## Integration Steps

### **Step 1: Update Text Reading Functions**

Replace the current text reading logic in your main reading functions:

```python
# OLD (in archive_main_original.py:458):
rho_new = (1.0 - alpha) * rho_old + alpha * pure

# NEW:
from core.quantum_state import apply_text_channel
rho_new = apply_text_channel(rho_old, text_embedding, alpha, "rank_one_update")
```

### **Step 2: Channel Type Selection**

Choose appropriate channel types based on narrative context:

- **"rank_one_update"**: For factual/expository text (preserves existing structure)
- **"coherent_rotation"**: For perspective shifts (unitary, entropy-preserving)  
- **"dephasing_mixture"**: For ambiguous/multi-interpretable text

### **Step 3: Add Channel Auditing**

Integrate the sanity check endpoints:

```python
# Add to main.py imports:
from routes.channel_audit_routes import router as audit_router
app.include_router(audit_router)

# Use in reading operations:
@app.post("/rho/{rho_id}/read_with_audit")
async def read_text_with_audit(rho_id: str, request: ReadRequest):
    # Apply channel
    rho_new = apply_text_channel(STATE[rho_id]["rho"], embedding, request.alpha)
    STATE[rho_id]["rho"] = rho_new
    
    # Audit channel properties
    audit_result = await run_channel_sanity_check(rho_id, ChannelAuditRequest(rho_id=rho_id))
    
    return {
        "success": True,
        "channel_audit": audit_result,
        "matrix_state": diagnostics(rho_new)
    }
```

### **Step 4: Channel Logging and Composition**

Store channels for auditability and composition:

```python
# In STATE structure, add:
STATE[rho_id]["channels"] = []  # Store applied channels

# When applying text:
text_channel = text_to_channel(embedding, alpha, channel_type)
STATE[rho_id]["channels"].append({
    "timestamp": time.time(),
    "channel": text_channel,
    "text_preview": text[:100],
    "alpha": alpha,
    "channel_type": channel_type
})
```

## Channel Sanity Checklist Implementation

### **Automatic Checks** (every update):
```python
def post_update_audit(rho_id: str) -> bool:
    rho = STATE[rho_id]["rho"]
    
    # 1. Trace check
    trace_error = abs(np.trace(rho) - 1.0)
    if trace_error > 1e-8:
        logger.warning(f"Trace violation in {rho_id}: {trace_error}")
        return False
    
    # 2. PSD check  
    min_eigenval = np.min(np.linalg.eigvals(rho).real)
    if min_eigenval < -1e-10:
        logger.warning(f"PSD violation in {rho_id}: {min_eigenval}")
        return False
        
    return True
```

### **Periodic Checks** (via audit endpoints):
- **Integrability testing**: `/audit/sanity_check/{rho_id}` with test segments
- **Commutator analysis**: Check POVM order effects
- **Residue computation**: For narrative loops and circular references

## Frontend Integration

### **Channel Visualization**
Add to your React components:

```jsx
// In BookReaderTab.jsx or NarrativeTab.jsx
const [channelAudit, setChannelAudit] = useState(null);

const auditChannels = async () => {
  const response = await fetch(`/audit/sanity_check/${rhoId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rho_id: rhoId, check_integrability: true })
  });
  const audit = await response.json();
  setChannelAudit(audit);
};

// UI display:
<div style={{ background: channelAudit?.passes_sanity_check ? '#e8f5e8' : '#ffebee' }}>
  <h4>Channel Health</h4>
  <div>Trace Error: {channelAudit?.trace_preservation_error?.toExponential(2)}</div>
  <div>PSD Status: {channelAudit?.psd_violation < 1e-10 ? '✓' : '✗'}</div>
  {channelAudit?.recommendations?.map((rec, i) => (
    <div key={i} style={{ fontSize: 12, color: '#666' }}>{rec}</div>
  ))}
</div>
```

### **Channel Type Selection UI**
```jsx
<select value={channelType} onChange={(e) => setChannelType(e.target.value)}>
  <option value="rank_one_update">Factual Update</option>
  <option value="coherent_rotation">Perspective Shift</option>
  <option value="dephasing_mixture">Ambiguous Text</option>
</select>
```

## Migration Strategy

### **Phase 1: Parallel Implementation** (Weeks 1-2)
- Deploy new channel system alongside existing convex combination
- Add feature flag to switch between approaches
- Test channel audit endpoints

### **Phase 2: Gradual Migration** (Weeks 3-4)  
- Start using channels for new text reading operations
- Compare results between old and new approaches
- Fix any numerical/behavioral differences

### **Phase 3: Full Deployment** (Week 5)
- Switch all text reading to channel-based approach
- Remove deprecated `blend_states` usage
- Enable full channel auditing in production

### **Phase 4: Advanced Features** (Weeks 6+)
- Implement integrability testing for different segmentations
- Add residue computation for narrative loops  
- Create channel composition interface for complex transformations

## Verification Tests

Before deploying, run these tests:

```python
# Test 1: CPTP verification
def test_channel_cptp():
    embedding = np.random.randn(64)
    channel = text_to_channel(embedding, 0.3, "rank_one_update")
    audit = audit_channel_properties(channel)
    assert audit["passes_audit"], f"Channel failed audit: {audit}"

# Test 2: Integrability 
def test_integrability():
    text = "The knight rode through the forest..."
    segments_a = [text[:50], text[50:]]
    segments_b = [text[:30], text[30:70], text[70:]]
    # Should yield same final ρ (within tolerance)
    
# Test 3: Order effects
def test_povm_commutativity():
    rho = random_density_matrix()
    # Apply POVMs in different orders, check for differences
```

## Benefits of Channel Approach

1. **Safety**: ρ stays on legal manifold (PSD, trace=1)
2. **Composability**: Safe composition of reading operations  
3. **Auditability**: Explicit channel objects for analysis
4. **Mathematical Rigor**: Proper quantum information theory foundation
5. **Debugging**: Clear distinction between unitary (reversible) and dissipative (irreversible) effects

## Backward Compatibility

The `blend_states` function is deprecated but maintained for compatibility. All new code should use `apply_text_channel` with proper quantum channels.

---

**Next Steps**: 
1. Review this integration plan
2. Test the new `text_channels.py` module  
3. Start with Phase 1 parallel implementation
4. Add channel auditing to your development workflow