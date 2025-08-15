"""
Channel Composition Engine for Quantum Narrative Operations

This module implements the mathematical framework for composing quantum channels
in the Analytic Post-Lexical Grammatology system. It allows for the creation
of complex narrative transformations by combining simpler channel operations.

Key Concepts:
- Channel Composition: Φ₂ ∘ Φ₁ where channels are applied sequentially
- Parallel Composition: Multiple channels applied to different subsystems
- Tensor Product Composition: Channels operating on product spaces
- Conditional Composition: Channels applied based on measurement outcomes
- Channel Interpolation: Smooth transitions between different channel types

This enables sophisticated narrative transformations that can model complex
semantic relationships and multi-layered meaning structures.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from core.text_channels import TextChannel, text_to_channel, audit_channel_properties
from core.quantum_state import psd_project, apply_text_channel
from core.embedding import text_to_embedding_vector
import uuid
import time

logger = logging.getLogger(__name__)

class CompositionType(Enum):
    """Types of channel composition operations."""
    SEQUENTIAL = "sequential"  # Φ₂ ∘ Φ₁
    PARALLEL = "parallel"      # Φ₁ ⊗ Φ₂  
    CONVEX = "convex"         # λΦ₁ + (1-λ)Φ₂
    CONDITIONAL = "conditional" # If-then-else channel logic
    INTERPOLATED = "interpolated" # Smooth parameter transitions

@dataclass
class ChannelNode:
    """Represents a node in a channel composition graph."""
    node_id: str
    channel: TextChannel
    channel_type: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: float

@dataclass
class CompositionRule:
    """Defines how channels should be composed."""
    rule_id: str
    composition_type: CompositionType
    input_nodes: List[str]
    output_node: str
    parameters: Dict[str, Any]
    conditions: Optional[Dict[str, Any]] = None

@dataclass
class ComposedChannel:
    """Represents a composed quantum channel."""
    channel_id: str
    composition_graph: Dict[str, Any]
    channel_nodes: Dict[str, ChannelNode]
    composition_rules: List[CompositionRule]
    final_channel: TextChannel
    metadata: Dict[str, Any]
    audit_results: Optional[Dict[str, Any]] = None


class ChannelComposer:
    """
    Core engine for composing quantum channels in complex ways.
    
    The composer allows building sophisticated narrative transformation
    pipelines by combining simpler channel operations using various
    composition rules and logical structures.
    """
    
    def __init__(self):
        """Initialize the channel composer."""
        self.channel_library = {}  # Store of available base channels
        self.composition_templates = {}  # Predefined composition patterns
        self.composed_channels = {}  # Store of composed channels
        self.composition_history = []
    
    def register_base_channel(
        self,
        channel_id: str,
        text: str,
        channel_type: str = "rank_one_update",
        alpha: float = 0.3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelNode:
        """
        Register a base channel in the composition library.
        
        Args:
            channel_id: Unique identifier for the channel
            text: Source text for the channel
            channel_type: Type of quantum channel
            alpha: Channel parameter
            metadata: Additional metadata
            
        Returns:
            ChannelNode representing the registered channel
        """
        try:
            # Create the quantum channel
            embedding = text_to_embedding_vector(text)
            channel = text_to_channel(embedding, alpha, channel_type)
            
            # Create channel node
            node = ChannelNode(
                node_id=channel_id,
                channel=channel,
                channel_type=channel_type,
                parameters={"alpha": alpha, "text": text[:100]},
                metadata=metadata or {},
                created_at=time.time()
            )
            
            # Store in library
            self.channel_library[channel_id] = node
            
            logger.info(f"Registered base channel {channel_id} of type {channel_type}")
            return node
            
        except Exception as e:
            logger.error(f"Failed to register base channel {channel_id}: {e}")
            raise
    
    def create_sequential_composition(
        self,
        channel_ids: List[str],
        composition_id: Optional[str] = None
    ) -> ComposedChannel:
        """
        Create a sequential composition: Φₙ ∘ ... ∘ Φ₂ ∘ Φ₁
        
        Args:
            channel_ids: List of channel IDs to compose sequentially
            composition_id: Optional ID for the composed channel
            
        Returns:
            ComposedChannel representing the sequential composition
        """
        if len(channel_ids) < 2:
            raise ValueError("Sequential composition requires at least 2 channels")
        
        if composition_id is None:
            composition_id = f"seq_comp_{uuid.uuid4().hex[:8]}"
        
        # Verify all channels exist
        missing_channels = [cid for cid in channel_ids if cid not in self.channel_library]
        if missing_channels:
            raise ValueError(f"Missing channels: {missing_channels}")
        
        # Get channel nodes
        channel_nodes = {cid: self.channel_library[cid] for cid in channel_ids}
        
        # Create composition rules
        rules = []
        for i in range(len(channel_ids) - 1):
            rule = CompositionRule(
                rule_id=f"seq_rule_{i}",
                composition_type=CompositionType.SEQUENTIAL,
                input_nodes=[channel_ids[i], channel_ids[i + 1]],
                output_node=f"seq_output_{i}",
                parameters={"order": i}
            )
            rules.append(rule)
        
        # Create the composed channel by sequential application
        composed_channel = self._build_sequential_channel(channel_nodes, channel_ids)
        
        # Create composition graph
        composition_graph = {
            "type": "sequential",
            "nodes": channel_ids,
            "edges": [(channel_ids[i], channel_ids[i + 1]) for i in range(len(channel_ids) - 1)],
            "flow_direction": "left_to_right"
        }
        
        # Create the composed channel object
        result = ComposedChannel(
            channel_id=composition_id,
            composition_graph=composition_graph,
            channel_nodes=channel_nodes,
            composition_rules=rules,
            final_channel=composed_channel,
            metadata={
                "composition_type": "sequential",
                "num_channels": len(channel_ids),
                "created_at": time.time()
            }
        )
        
        # Store the composed channel
        self.composed_channels[composition_id] = result
        self.composition_history.append({
            "timestamp": time.time(),
            "operation": "sequential_composition",
            "channel_id": composition_id,
            "input_channels": channel_ids
        })
        
        logger.info(f"Created sequential composition {composition_id} with {len(channel_ids)} channels")
        return result
    
    def create_convex_combination(
        self,
        channel_weights: Dict[str, float],
        composition_id: Optional[str] = None
    ) -> ComposedChannel:
        """
        Create a convex combination: Σᵢ λᵢ Φᵢ where Σᵢ λᵢ = 1
        
        Args:
            channel_weights: Dictionary mapping channel IDs to weights
            composition_id: Optional ID for the composed channel
            
        Returns:
            ComposedChannel representing the convex combination
        """
        if len(channel_weights) < 2:
            raise ValueError("Convex combination requires at least 2 channels")
        
        # Verify weights sum to 1
        total_weight = sum(channel_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            channel_weights = {cid: w / total_weight for cid, w in channel_weights.items()}
            logger.warning(f"Normalized weights to sum to 1.0")
        
        if composition_id is None:
            composition_id = f"convex_comp_{uuid.uuid4().hex[:8]}"
        
        # Verify all channels exist
        missing_channels = [cid for cid in channel_weights.keys() if cid not in self.channel_library]
        if missing_channels:
            raise ValueError(f"Missing channels: {missing_channels}")
        
        # Get channel nodes
        channel_nodes = {cid: self.channel_library[cid] for cid in channel_weights.keys()}
        
        # Create composition rule
        rule = CompositionRule(
            rule_id="convex_rule",
            composition_type=CompositionType.CONVEX,
            input_nodes=list(channel_weights.keys()),
            output_node="convex_output",
            parameters={"weights": channel_weights}
        )
        
        # Create the composed channel by convex combination
        composed_channel = self._build_convex_channel(channel_nodes, channel_weights)
        
        # Create composition graph
        composition_graph = {
            "type": "convex",
            "nodes": list(channel_weights.keys()),
            "weights": channel_weights,
            "center_node": "convex_combiner"
        }
        
        # Create the composed channel object
        result = ComposedChannel(
            channel_id=composition_id,
            composition_graph=composition_graph,
            channel_nodes=channel_nodes,
            composition_rules=[rule],
            final_channel=composed_channel,
            metadata={
                "composition_type": "convex",
                "num_channels": len(channel_weights),
                "weights": channel_weights,
                "created_at": time.time()
            }
        )
        
        # Store the composed channel
        self.composed_channels[composition_id] = result
        self.composition_history.append({
            "timestamp": time.time(),
            "operation": "convex_combination",
            "channel_id": composition_id,
            "input_channels": list(channel_weights.keys()),
            "weights": channel_weights
        })
        
        logger.info(f"Created convex combination {composition_id} with {len(channel_weights)} channels")
        return result
    
    def create_conditional_composition(
        self,
        condition_channel_id: str,
        true_channel_id: str,
        false_channel_id: str,
        threshold: float = 0.5,
        composition_id: Optional[str] = None
    ) -> ComposedChannel:
        """
        Create a conditional composition based on measurement outcome.
        
        Args:
            condition_channel_id: Channel used for condition evaluation
            true_channel_id: Channel applied if condition is true
            false_channel_id: Channel applied if condition is false
            threshold: Threshold for condition evaluation
            composition_id: Optional ID for the composed channel
            
        Returns:
            ComposedChannel representing the conditional composition
        """
        if composition_id is None:
            composition_id = f"cond_comp_{uuid.uuid4().hex[:8]}"
        
        # Verify all channels exist
        required_channels = [condition_channel_id, true_channel_id, false_channel_id]
        missing_channels = [cid for cid in required_channels if cid not in self.channel_library]
        if missing_channels:
            raise ValueError(f"Missing channels: {missing_channels}")
        
        # Get channel nodes
        channel_nodes = {cid: self.channel_library[cid] for cid in required_channels}
        
        # Create composition rule
        rule = CompositionRule(
            rule_id="conditional_rule",
            composition_type=CompositionType.CONDITIONAL,
            input_nodes=required_channels,
            output_node="conditional_output",
            parameters={"threshold": threshold},
            conditions={
                "condition_channel": condition_channel_id,
                "true_branch": true_channel_id,
                "false_branch": false_channel_id
            }
        )
        
        # Create the composed channel
        composed_channel = self._build_conditional_channel(
            channel_nodes, condition_channel_id, true_channel_id, false_channel_id, threshold
        )
        
        # Create composition graph
        composition_graph = {
            "type": "conditional",
            "condition_node": condition_channel_id,
            "true_branch": true_channel_id,
            "false_branch": false_channel_id,
            "threshold": threshold
        }
        
        # Create the composed channel object
        result = ComposedChannel(
            channel_id=composition_id,
            composition_graph=composition_graph,
            channel_nodes=channel_nodes,
            composition_rules=[rule],
            final_channel=composed_channel,
            metadata={
                "composition_type": "conditional",
                "condition_channel": condition_channel_id,
                "threshold": threshold,
                "created_at": time.time()
            }
        )
        
        # Store the composed channel
        self.composed_channels[composition_id] = result
        self.composition_history.append({
            "timestamp": time.time(),
            "operation": "conditional_composition",
            "channel_id": composition_id,
            "condition_channel": condition_channel_id,
            "branches": [true_channel_id, false_channel_id]
        })
        
        logger.info(f"Created conditional composition {composition_id}")
        return result
    
    def create_interpolated_composition(
        self,
        channel_id_1: str,
        channel_id_2: str,
        interpolation_parameter: float,
        composition_id: Optional[str] = None
    ) -> ComposedChannel:
        """
        Create an interpolated composition between two channels.
        
        Args:
            channel_id_1: First channel ID
            channel_id_2: Second channel ID  
            interpolation_parameter: Parameter t ∈ [0,1] for interpolation
            composition_id: Optional ID for the composed channel
            
        Returns:
            ComposedChannel representing the interpolated composition
        """
        if not 0 <= interpolation_parameter <= 1:
            raise ValueError("Interpolation parameter must be between 0 and 1")
        
        if composition_id is None:
            composition_id = f"interp_comp_{uuid.uuid4().hex[:8]}"
        
        # Verify channels exist
        required_channels = [channel_id_1, channel_id_2]
        missing_channels = [cid for cid in required_channels if cid not in self.channel_library]
        if missing_channels:
            raise ValueError(f"Missing channels: {missing_channels}")
        
        # Create convex combination with interpolation parameter
        weights = {
            channel_id_1: 1 - interpolation_parameter,
            channel_id_2: interpolation_parameter
        }
        
        return self.create_convex_combination(weights, composition_id)
    
    def audit_composed_channel(self, composition_id: str) -> Dict[str, Any]:
        """
        Audit a composed channel for CPTP properties and correctness.
        
        Args:
            composition_id: ID of the composed channel to audit
            
        Returns:
            Audit results
        """
        if composition_id not in self.composed_channels:
            raise ValueError(f"Composed channel {composition_id} not found")
        
        composed = self.composed_channels[composition_id]
        
        try:
            # Audit the final composed channel
            audit_results = audit_channel_properties(composed.final_channel)
            
            # Add composition-specific checks
            audit_results.update({
                "composition_type": composed.metadata.get("composition_type"),
                "num_base_channels": len(composed.channel_nodes),
                "composition_complexity": len(composed.composition_rules),
                "base_channel_audit": {}
            })
            
            # Audit individual base channels
            for channel_id, node in composed.channel_nodes.items():
                try:
                    base_audit = audit_channel_properties(node.channel)
                    audit_results["base_channel_audit"][channel_id] = {
                        "passes_audit": base_audit.get("passes_audit", False),
                        "trace_preservation": base_audit.get("trace_preservation", 0),
                        "cptp_compliance": base_audit.get("cptp_compliance", False)
                    }
                except Exception as e:
                    audit_results["base_channel_audit"][channel_id] = {"error": str(e)}
            
            # Store audit results
            composed.audit_results = audit_results
            
            logger.info(f"Audited composed channel {composition_id}")
            return audit_results
            
        except Exception as e:
            logger.error(f"Failed to audit composed channel {composition_id}: {e}")
            return {"error": str(e)}
    
    def apply_composed_channel(
        self,
        composition_id: str,
        input_state: np.ndarray
    ) -> np.ndarray:
        """
        Apply a composed channel to a quantum state.
        
        Args:
            composition_id: ID of the composed channel
            input_state: Input quantum state
            
        Returns:
            Output quantum state after applying the composed channel
        """
        if composition_id not in self.composed_channels:
            raise ValueError(f"Composed channel {composition_id} not found")
        
        composed = self.composed_channels[composition_id]
        
        try:
            output_state = composed.final_channel.apply(input_state)
            
            logger.debug(f"Applied composed channel {composition_id}")
            return output_state
            
        except Exception as e:
            logger.error(f"Failed to apply composed channel {composition_id}: {e}")
            raise
    
    def get_composition_info(self, composition_id: str) -> Dict[str, Any]:
        """Get detailed information about a composed channel."""
        if composition_id not in self.composed_channels:
            raise ValueError(f"Composed channel {composition_id} not found")
        
        composed = self.composed_channels[composition_id]
        
        return {
            "channel_id": composed.channel_id,
            "composition_type": composed.metadata.get("composition_type"),
            "num_base_channels": len(composed.channel_nodes),
            "base_channels": {
                cid: {
                    "channel_type": node.channel_type,
                    "parameters": node.parameters,
                    "created_at": node.created_at
                } for cid, node in composed.channel_nodes.items()
            },
            "composition_graph": composed.composition_graph,
            "composition_rules": [
                {
                    "rule_id": rule.rule_id,
                    "type": rule.composition_type.value,
                    "input_nodes": rule.input_nodes,
                    "parameters": rule.parameters
                } for rule in composed.composition_rules
            ],
            "metadata": composed.metadata,
            "audit_status": "audited" if composed.audit_results else "not_audited"
        }
    
    def list_compositions(self) -> Dict[str, Dict[str, Any]]:
        """List all composed channels with summary information."""
        return {
            comp_id: {
                "composition_type": comp.metadata.get("composition_type"),
                "num_base_channels": len(comp.channel_nodes),
                "created_at": comp.metadata.get("created_at"),
                "audited": comp.audit_results is not None
            } for comp_id, comp in self.composed_channels.items()
        }
    
    def list_base_channels(self) -> Dict[str, Dict[str, Any]]:
        """List all base channels in the library."""
        return {
            channel_id: {
                "channel_type": node.channel_type,
                "parameters": node.parameters,
                "metadata": node.metadata,
                "created_at": node.created_at
            } for channel_id, node in self.channel_library.items()
        }
    
    # Helper methods for building composed channels
    
    def _build_sequential_channel(
        self,
        channel_nodes: Dict[str, ChannelNode],
        channel_ids: List[str]
    ) -> TextChannel:
        """Build a sequential composition of channels."""
        
        class SequentialChannel(TextChannel):
            def __init__(self, channels: List[TextChannel]):
                self.channels = channels
                # Sequential composition preserves CPTP if all components are CPTP
                super().__init__([])  # Empty Kraus operators, we override apply()
            
            def apply(self, rho: np.ndarray) -> np.ndarray:
                current_state = rho.copy()
                for channel in self.channels:
                    current_state = channel.apply(current_state)
                return current_state
        
        channels = [channel_nodes[cid].channel for cid in channel_ids]
        return SequentialChannel(channels)
    
    def _build_convex_channel(
        self,
        channel_nodes: Dict[str, ChannelNode],
        weights: Dict[str, float]
    ) -> TextChannel:
        """Build a convex combination of channels."""
        
        class ConvexChannel(TextChannel):
            def __init__(self, weighted_channels: List[Tuple[float, TextChannel]]):
                self.weighted_channels = weighted_channels
                super().__init__([])  # Empty Kraus operators, we override apply()
            
            def apply(self, rho: np.ndarray) -> np.ndarray:
                result = np.zeros_like(rho)
                for weight, channel in self.weighted_channels:
                    result += weight * channel.apply(rho)
                return psd_project(result)  # Ensure result is valid density matrix
        
        weighted_channels = [(weights[cid], channel_nodes[cid].channel) for cid in weights.keys()]
        return ConvexChannel(weighted_channels)
    
    def _build_conditional_channel(
        self,
        channel_nodes: Dict[str, ChannelNode],
        condition_id: str,
        true_id: str,
        false_id: str,
        threshold: float
    ) -> TextChannel:
        """Build a conditional channel composition."""
        
        class ConditionalChannel(TextChannel):
            def __init__(self, condition_channel, true_channel, false_channel, threshold):
                self.condition_channel = condition_channel
                self.true_channel = true_channel
                self.false_channel = false_channel
                self.threshold = threshold
                super().__init__([])  # Empty Kraus operators, we override apply()
            
            def apply(self, rho: np.ndarray) -> np.ndarray:
                # Apply condition channel and measure purity as condition
                condition_state = self.condition_channel.apply(rho)
                purity = np.trace(condition_state @ condition_state).real
                
                # Choose branch based on condition
                if purity > self.threshold:
                    return self.true_channel.apply(rho)
                else:
                    return self.false_channel.apply(rho)
        
        return ConditionalChannel(
            channel_nodes[condition_id].channel,
            channel_nodes[true_id].channel,
            channel_nodes[false_id].channel,
            threshold
        )


# Global composer instance
CHANNEL_COMPOSER = ChannelComposer()

# Convenience functions

def create_composition_from_texts(
    texts: List[str],
    composition_type: str = "sequential",
    weights: Optional[List[float]] = None,
    channel_type: str = "rank_one_update",
    alpha: float = 0.3
) -> str:
    """
    Create a channel composition from a list of texts.
    
    Args:
        texts: List of texts to create channels from
        composition_type: Type of composition ('sequential', 'convex')
        weights: Weights for convex combination (if applicable)
        channel_type: Type of quantum channels to create
        alpha: Channel parameter
        
    Returns:
        ID of the created composed channel
    """
    # Register base channels
    channel_ids = []
    for i, text in enumerate(texts):
        channel_id = f"text_channel_{i}_{uuid.uuid4().hex[:8]}"
        CHANNEL_COMPOSER.register_base_channel(
            channel_id, text, channel_type, alpha,
            metadata={"text_preview": text[:50]}
        )
        channel_ids.append(channel_id)
    
    # Create composition
    if composition_type == "sequential":
        composed = CHANNEL_COMPOSER.create_sequential_composition(channel_ids)
    elif composition_type == "convex":
        if weights is None:
            weights = [1.0 / len(texts)] * len(texts)  # Equal weights
        channel_weights = dict(zip(channel_ids, weights))
        composed = CHANNEL_COMPOSER.create_convex_combination(channel_weights)
    else:
        raise ValueError(f"Unsupported composition type: {composition_type}")
    
    return composed.channel_id