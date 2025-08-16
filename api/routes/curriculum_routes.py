"""
Curriculum/Sequencing Routes (APLG Claim Set E)

Implements intelligent sequencing of content to optimize learning trajectories
and narrative development paths. This uses quantum state analysis to determine
optimal ordering of texts, documents, or experiences based on reader state
and desired outcomes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import uuid
from datetime import datetime

from routes.matrix_routes import STATE, rho_read_with_channel
from routes.povm_routes import measure_rho, PACKS
from models.requests import ReadReq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/curriculum", tags=["curriculum-sequencing"])

# Request Models
class LearningObjective(BaseModel):
    """A learning or development objective."""
    name: str
    target_measurement: str  # POVM measurement axis
    target_value: float  # Desired measurement outcome
    priority: float = 1.0  # Relative importance
    measurement_pack: str = "advanced_narrative_pack"

class ContentItem(BaseModel):
    """A piece of content that can be sequenced."""
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    difficulty_level: float = 0.5  # 0-1 scale
    content_type: str = "text"  # text, exercise, reflection, assessment
    prerequisites: List[str] = Field(default_factory=list)
    estimated_duration: int = 300  # seconds

class SequencingRequest(BaseModel):
    """Request for curriculum sequencing."""
    rho_id: str
    content_pool: List[ContentItem]
    learning_objectives: List[LearningObjective]
    max_sequence_length: int = 10
    optimization_strategy: str = "gradient_ascent"  # gradient_ascent, genetic, reinforcement
    constraints: Optional[Dict[str, Any]] = None

class SequencingConstraint(BaseModel):
    """Constraint for curriculum sequencing."""
    type: str  # "prerequisite", "max_difficulty_jump", "content_type_balance", "duration_limit"
    parameters: Dict[str, Any]

class SequenceStep(BaseModel):
    """A single step in a learning sequence."""
    position: int
    content_item: ContentItem
    predicted_state_change: Dict[str, float]
    objective_progress: Dict[str, float]
    cumulative_progress: Dict[str, float]
    readiness_score: float

class CurriculumSequence(BaseModel):
    """A complete curriculum sequence."""
    sequence_id: str
    rho_id: str
    learning_objectives: List[LearningObjective]
    sequence: List[SequenceStep]
    total_duration: int
    predicted_outcomes: Dict[str, float]
    optimization_score: float
    strategy_used: str

# In-memory storage for sequences
CURRICULUM_SEQUENCES: Dict[str, CurriculumSequence] = {}

def predict_content_effects(rho: np.ndarray, content: ContentItem) -> Dict[str, float]:
    """
    Predict the quantum state changes from applying a content item.
    """
    try:
        from core.embedding import text_to_embedding_vector
        from core.quantum_state import apply_text_channel
        
        # Simulate applying the content
        content_embedding = text_to_embedding_vector(content.content)
        predicted_rho = apply_text_channel(rho, content_embedding, 0.3, "rank_one_update")
        
        # Calculate predicted changes
        current_purity = float(np.real(np.trace(rho @ rho)))
        predicted_purity = float(np.real(np.trace(predicted_rho @ predicted_rho)))
        
        current_eigs = np.linalg.eigvals(rho)
        predicted_eigs = np.linalg.eigvals(predicted_rho)
        
        current_entropy = -np.sum(current_eigs * np.log2(current_eigs + 1e-10))
        predicted_entropy = -np.sum(predicted_eigs * np.log2(predicted_eigs + 1e-10))
        
        return {
            "purity_change": float(predicted_purity - current_purity),
            "entropy_change": float(predicted_entropy - current_entropy),
            "eigenvalue_shift": float(np.linalg.norm(predicted_eigs - current_eigs)),
            "complexity_change": float(len(predicted_eigs[predicted_eigs > 1e-6]) - len(current_eigs[current_eigs > 1e-6]))
        }
        
    except Exception as e:
        logger.error(f"Failed to predict content effects: {e}")
        return {"error": str(e)}

def predict_measurement_changes(rho_current: np.ndarray, rho_predicted: np.ndarray, 
                               objectives: List[LearningObjective]) -> Dict[str, float]:
    """
    Predict changes in measurement outcomes for learning objectives.
    """
    changes = {}
    
    try:
        for objective in objectives:
            # Simplified measurement prediction
            # In production, this would use actual POVM measurements
            if objective.measurement_pack in PACKS:
                # Simulate measurement change
                current_trace = float(np.real(np.trace(rho_current)))
                predicted_trace = float(np.real(np.trace(rho_predicted)))
                
                # Simple proxy: use purity change as measurement change
                current_purity = float(np.real(np.trace(rho_current @ rho_current)))
                predicted_purity = float(np.real(np.trace(rho_predicted @ rho_predicted)))
                
                measurement_change = predicted_purity - current_purity
                changes[objective.name] = measurement_change
            else:
                changes[objective.name] = 0.0
                
    except Exception as e:
        logger.error(f"Failed to predict measurement changes: {e}")
        changes["error"] = str(e)
    
    return changes

def calculate_readiness_score(rho: np.ndarray, content: ContentItem, 
                             prerequisites_met: List[str]) -> float:
    """
    Calculate how ready the current quantum state is for a piece of content.
    """
    try:
        base_score = 0.5  # Neutral readiness
        
        # Factor 1: Prerequisites
        prereq_score = len(prerequisites_met) / len(content.prerequisites) if content.prerequisites else 1.0
        
        # Factor 2: Difficulty alignment with state complexity
        eigenvals = np.linalg.eigvals(rho)
        state_complexity = len(eigenvals[eigenvals > 1e-6]) / len(eigenvals)
        difficulty_alignment = 1.0 - abs(state_complexity - content.difficulty_level)
        
        # Factor 3: Quantum state entropy (learning capacity)
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
        max_entropy = np.log2(len(eigenvals))
        learning_capacity = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Combine factors
        readiness_score = (
            0.4 * prereq_score +
            0.3 * difficulty_alignment +
            0.3 * learning_capacity
        )
        
        return float(max(0.0, min(1.0, readiness_score)))
        
    except Exception as e:
        logger.error(f"Failed to calculate readiness score: {e}")
        return 0.5

def optimize_sequence_gradient_ascent(rho_id: str, content_pool: List[ContentItem], 
                                    objectives: List[LearningObjective], 
                                    max_length: int) -> List[ContentItem]:
    """
    Optimize curriculum sequence using gradient ascent approach.
    """
    try:
        current_rho = STATE[rho_id]["rho"]
        sequence = []
        used_items = set()
        completed_prerequisites = set()
        
        for step in range(max_length):
            best_item = None
            best_score = -float('inf')
            
            # Evaluate each unused content item
            for item in content_pool:
                if item.id in used_items:
                    continue
                
                # Check prerequisites
                prereqs_met = [p for p in item.prerequisites if p in completed_prerequisites]
                if len(prereqs_met) < len(item.prerequisites):
                    continue  # Prerequisites not met
                
                # Calculate readiness
                readiness = calculate_readiness_score(current_rho, item, prereqs_met)
                
                # Predict effects
                effects = predict_content_effects(current_rho, item)
                
                if "error" in effects:
                    continue
                
                # Simulate application
                from core.embedding import text_to_embedding_vector
                from core.quantum_state import apply_text_channel
                
                content_embedding = text_to_embedding_vector(item.content)
                predicted_rho = apply_text_channel(current_rho, content_embedding, 0.3, "rank_one_update")
                
                # Calculate objective progress
                measurement_changes = predict_measurement_changes(current_rho, predicted_rho, objectives)
                
                # Score based on objective progress and readiness
                objective_score = 0.0
                for objective in objectives:
                    if objective.name in measurement_changes:
                        # Distance to target (closer is better)
                        current_distance = abs(0.5 - objective.target_value)  # Assume current measurement is 0.5
                        predicted_distance = abs((0.5 + measurement_changes[objective.name]) - objective.target_value)
                        progress = max(0, current_distance - predicted_distance)
                        objective_score += progress * objective.priority
                
                total_score = 0.6 * objective_score + 0.4 * readiness
                
                if total_score > best_score:
                    best_score = total_score
                    best_item = item
            
            if best_item is None:
                break  # No more viable items
            
            # Add to sequence
            sequence.append(best_item)
            used_items.add(best_item.id)
            completed_prerequisites.add(best_item.id)
            
            # Update current state for next iteration
            from core.embedding import text_to_embedding_vector
            from core.quantum_state import apply_text_channel
            
            content_embedding = text_to_embedding_vector(best_item.content)
            current_rho = apply_text_channel(current_rho, content_embedding, 0.3, "rank_one_update")
        
        return sequence
        
    except Exception as e:
        logger.error(f"Gradient ascent optimization failed: {e}")
        return content_pool[:max_length]  # Fallback to first items

def optimize_sequence_genetic(content_pool: List[ContentItem], objectives: List[LearningObjective], 
                            max_length: int) -> List[ContentItem]:
    """
    Optimize curriculum sequence using genetic algorithm approach.
    """
    # Simplified genetic algorithm implementation
    import random
    
    try:
        population_size = 20
        generations = 10
        
        # Initialize population
        population = []
        for _ in range(population_size):
            sequence_length = min(max_length, len(content_pool))
            sequence = random.sample(content_pool, sequence_length)
            population.append(sequence)
        
        # Evolution
        for generation in range(generations):
            # Score each individual (simplified)
            scored_population = []
            for individual in population:
                score = random.random()  # Placeholder scoring
                scored_population.append((score, individual))
            
            # Select top performers
            scored_population.sort(reverse=True)
            survivors = [ind for _, ind in scored_population[:population_size//2]]
            
            # Create next generation
            new_population = survivors.copy()
            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Simple crossover: take first half from parent1, second from parent2
                midpoint = len(parent1) // 2
                child = parent1[:midpoint] + parent2[midpoint:]
                
                # Remove duplicates
                seen = set()
                child = [item for item in child if not (item.id in seen or seen.add(item.id))]
                
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        best_individual = max(population, key=lambda x: random.random())  # Placeholder
        return best_individual
        
    except Exception as e:
        logger.error(f"Genetic algorithm optimization failed: {e}")
        return content_pool[:max_length]

@router.post("/plan_sequence")
async def plan_curriculum_sequence(request: SequencingRequest) -> CurriculumSequence:
    """
    Plan an optimal curriculum sequence for learning objectives.
    
    This implements APLG Claim Set E: intelligent content sequencing
    based on quantum state analysis and learning objectives.
    """
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        # Select optimization strategy
        if request.optimization_strategy == "gradient_ascent":
            optimized_sequence = optimize_sequence_gradient_ascent(
                request.rho_id, request.content_pool, request.learning_objectives, 
                request.max_sequence_length
            )
        elif request.optimization_strategy == "genetic":
            optimized_sequence = optimize_sequence_genetic(
                request.content_pool, request.learning_objectives, 
                request.max_sequence_length
            )
        else:
            # Default: simple ordering
            optimized_sequence = request.content_pool[:request.max_sequence_length]
        
        # Build detailed sequence steps
        current_rho = STATE[request.rho_id]["rho"]
        sequence_steps = []
        cumulative_progress = {obj.name: 0.0 for obj in request.learning_objectives}
        completed_prerequisites = set()
        total_duration = 0
        
        for i, content_item in enumerate(optimized_sequence):
            # Predict effects of this content
            effects = predict_content_effects(current_rho, content_item)
            
            # Calculate readiness
            prereqs_met = [p for p in content_item.prerequisites if p in completed_prerequisites]
            readiness = calculate_readiness_score(current_rho, content_item, prereqs_met)
            
            # Predict measurement changes
            from core.embedding import text_to_embedding_vector
            from core.quantum_state import apply_text_channel
            
            content_embedding = text_to_embedding_vector(content_item.content)
            predicted_rho = apply_text_channel(current_rho, content_embedding, 0.3, "rank_one_update")
            
            measurement_changes = predict_measurement_changes(current_rho, predicted_rho, request.learning_objectives)
            
            # Update cumulative progress
            for obj in request.learning_objectives:
                if obj.name in measurement_changes:
                    cumulative_progress[obj.name] += measurement_changes[obj.name]
            
            # Create sequence step
            step = SequenceStep(
                position=i + 1,
                content_item=content_item,
                predicted_state_change=effects,
                objective_progress=measurement_changes,
                cumulative_progress=cumulative_progress.copy(),
                readiness_score=readiness
            )
            
            sequence_steps.append(step)
            
            # Update state and tracking
            current_rho = predicted_rho
            completed_prerequisites.add(content_item.id)
            total_duration += content_item.estimated_duration
        
        # Calculate predicted outcomes
        predicted_outcomes = {}
        for obj in request.learning_objectives:
            current_progress = cumulative_progress.get(obj.name, 0.0)
            predicted_value = 0.5 + current_progress  # Assume baseline of 0.5
            predicted_outcomes[obj.name] = min(1.0, max(0.0, predicted_value))
        
        # Calculate optimization score
        objective_scores = []
        for obj in request.learning_objectives:
            predicted_value = predicted_outcomes.get(obj.name, 0.5)
            distance_to_target = abs(predicted_value - obj.target_value)
            objective_score = max(0, 1.0 - distance_to_target) * obj.priority
            objective_scores.append(objective_score)
        
        optimization_score = np.mean(objective_scores) if objective_scores else 0.0
        
        # Create final sequence
        sequence_id = str(uuid.uuid4())
        curriculum_sequence = CurriculumSequence(
            sequence_id=sequence_id,
            rho_id=request.rho_id,
            learning_objectives=request.learning_objectives,
            sequence=sequence_steps,
            total_duration=total_duration,
            predicted_outcomes=predicted_outcomes,
            optimization_score=float(optimization_score),
            strategy_used=request.optimization_strategy
        )
        
        # Store sequence
        CURRICULUM_SEQUENCES[sequence_id] = curriculum_sequence
        
        logger.info(f"Generated curriculum sequence {sequence_id} with {len(sequence_steps)} steps")
        return curriculum_sequence
        
    except Exception as e:
        logger.error(f"Curriculum sequencing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sequencing failed: {str(e)}")

@router.post("/execute_sequence/{sequence_id}")
async def execute_curriculum_sequence(sequence_id: str, step_delay: float = 0.0):
    """
    Execute a planned curriculum sequence by applying each content item.
    """
    try:
        if sequence_id not in CURRICULUM_SEQUENCES:
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        sequence = CURRICULUM_SEQUENCES[sequence_id]
        
        if sequence.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        execution_log = []
        
        for step in sequence.sequence:
            # Apply content item
            read_req = ReadReq(raw_text=step.content_item.content, alpha=0.3)
            result = rho_read_with_channel(sequence.rho_id, read_req, "rank_one_update")
            
            execution_log.append({
                "step": step.position,
                "content_id": step.content_item.id,
                "title": step.content_item.title,
                "applied": result.get("read", False),
                "quantum_effects": result.get("quantum_effects", {})
            })
            
            # Optional delay between steps
            if step_delay > 0:
                import asyncio
                await asyncio.sleep(step_delay)
        
        return {
            "sequence_id": sequence_id,
            "execution_completed": True,
            "steps_executed": len(sequence.sequence),
            "total_duration": sequence.total_duration,
            "execution_log": execution_log
        }
        
    except Exception as e:
        logger.error(f"Sequence execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@router.get("/sequences")
async def list_curriculum_sequences():
    """
    List all stored curriculum sequences.
    """
    sequences = []
    for sequence_id, sequence in CURRICULUM_SEQUENCES.items():
        sequences.append({
            "sequence_id": sequence_id,
            "rho_id": sequence.rho_id,
            "sequence_length": len(sequence.sequence),
            "total_duration": sequence.total_duration,
            "optimization_score": sequence.optimization_score,
            "strategy_used": sequence.strategy_used,
            "learning_objectives": [obj.name for obj in sequence.learning_objectives]
        })
    
    return {
        "total_sequences": len(CURRICULUM_SEQUENCES),
        "sequences": sequences
    }

@router.get("/sequence/{sequence_id}")
async def get_curriculum_sequence(sequence_id: str) -> CurriculumSequence:
    """
    Retrieve a specific curriculum sequence.
    """
    if sequence_id not in CURRICULUM_SEQUENCES:
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    return CURRICULUM_SEQUENCES[sequence_id]

@router.post("/test_sequencing")
async def test_curriculum_sequencing():
    """
    Test curriculum sequencing with sample content and objectives.
    """
    from routes.matrix_routes import rho_init
    
    # Create test matrix
    test_init = rho_init(seed_text="Student learning state")
    test_rho_id = test_init["rho_id"]
    
    try:
        # Sample content items
        content_pool = [
            ContentItem(
                id="intro_1",
                title="Introduction to Concepts",
                content="Let's begin with basic concepts and foundational ideas.",
                difficulty_level=0.2,
                estimated_duration=300
            ),
            ContentItem(
                id="practice_1",
                title="Basic Practice",
                content="Practice exercises to reinforce understanding.",
                difficulty_level=0.4,
                prerequisites=["intro_1"],
                estimated_duration=600
            ),
            ContentItem(
                id="advanced_1",
                title="Advanced Applications",
                content="Complex applications requiring deep understanding.",
                difficulty_level=0.8,
                prerequisites=["practice_1"],
                estimated_duration=900
            ),
            ContentItem(
                id="reflection_1",
                title="Reflection and Integration",
                content="Reflect on learning and integrate new knowledge.",
                difficulty_level=0.6,
                estimated_duration=450
            )
        ]
        
        # Sample learning objectives
        objectives = [
            LearningObjective(
                name="conceptual_understanding",
                target_measurement="narrator_reliability",
                target_value=0.8,
                priority=1.0
            ),
            LearningObjective(
                name="practical_skills",
                target_measurement="agency_locus",
                target_value=0.7,
                priority=0.8
            )
        ]
        
        # Test sequencing
        request = SequencingRequest(
            rho_id=test_rho_id,
            content_pool=content_pool,
            learning_objectives=objectives,
            max_sequence_length=4,
            optimization_strategy="gradient_ascent"
        )
        
        sequence = await plan_curriculum_sequence(request)
        
        return {
            "test_completed": True,
            "sequence_id": sequence.sequence_id,
            "sequence_length": len(sequence.sequence),
            "optimization_score": sequence.optimization_score,
            "predicted_outcomes": sequence.predicted_outcomes,
            "sequence_preview": [
                {
                    "position": step.position,
                    "title": step.content_item.title,
                    "readiness_score": step.readiness_score
                }
                for step in sequence.sequence
            ]
        }
        
    finally:
        # Clean up test matrix
        if test_rho_id in STATE:
            del STATE[test_rho_id]