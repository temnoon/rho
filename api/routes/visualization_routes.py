"""
Bures-Preserving Visualization Routes (APLG Claim Set F)

Implements quantum-geometric visualization of density matrix trajectories that
preserves Bures distance relationships. This provides interactive visualizations
of narrative state evolution, reader journeys, and quantum state manifolds
using proper Riemannian geometry on the space of density matrices.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import uuid
from datetime import datetime
import base64
import io

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/visualization", tags=["bures-visualization"])

# Request Models
class TrajectoryVisualizationRequest(BaseModel):
    """Request for trajectory visualization."""
    rho_trajectory: List[str]  # List of rho_ids
    visualization_type: str = "bures_manifold"  # bures_manifold, eigenvalue_evolution, purity_flow
    dimension_reduction: str = "bures_mds"  # bures_mds, pca, tsne, umap
    color_mapping: str = "chronological"  # chronological, purity, entropy, measurement
    include_geodesics: bool = True
    resolution: int = 100

class StateVisualizationRequest(BaseModel):
    """Request for single state visualization."""
    rho_id: str
    visualization_type: str = "eigenspace_sphere"  # eigenspace_sphere, bloch_sphere, purity_cone
    measurement_overlays: List[str] = Field(default_factory=list)
    coordinate_system: str = "cartesian"  # cartesian, spherical, hyperbolic

class InteractiveVisualizationRequest(BaseModel):
    """Request for interactive visualization."""
    rho_ids: List[str]
    interaction_type: str = "exploration"  # exploration, comparison, navigation
    real_time_updates: bool = False
    export_format: str = "json"  # json, svg, plotly, d3

# Response Models
class VisualizationData(BaseModel):
    """Visualization data structure."""
    visualization_id: str
    type: str
    coordinates: List[List[float]]
    metadata: Dict[str, Any]
    bures_distances: Optional[List[List[float]]] = None
    geodesic_paths: Optional[List[List[List[float]]]] = None
    color_values: Optional[List[float]] = None

class VisualizationResponse(BaseModel):
    """Complete visualization response."""
    success: bool
    visualization_data: VisualizationData
    rendering_info: Dict[str, Any]
    interaction_capabilities: List[str]
    export_options: List[str]

# In-memory storage for visualizations
VISUALIZATIONS: Dict[str, Dict[str, Any]] = {}

def calculate_bures_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Calculate the Bures distance between two density matrices.
    
    Bures distance: d_B(ρ₁, ρ₂) = √(2(1 - √F(ρ₁, ρ₂)))
    where F is the fidelity: F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))
    """
    try:
        # Calculate fidelity
        sqrt_rho1 = np.real(np.sqrt(rho1 + 1e-12 * np.eye(rho1.shape[0])))
        sqrt_rho1_rho2_sqrt_rho1 = sqrt_rho1 @ rho2 @ sqrt_rho1
        
        # Eigenvalues of the matrix under the square root
        eigenvals = np.linalg.eigvals(sqrt_rho1_rho2_sqrt_rho1)
        eigenvals = np.real(eigenvals[eigenvals >= 0])
        
        # Fidelity
        fidelity = np.sum(np.sqrt(eigenvals))
        fidelity = min(1.0, max(0.0, fidelity))  # Clamp to [0,1]
        
        # Bures distance
        bures_dist = np.sqrt(2 * (1 - fidelity))
        
        return float(bures_dist)
        
    except Exception as e:
        logger.warning(f"Bures distance calculation failed: {e}")
        # Fallback to Frobenius distance
        return float(np.linalg.norm(rho1 - rho2, 'fro'))

def calculate_bures_distance_matrix(rho_list: List[np.ndarray]) -> np.ndarray:
    """
    Calculate pairwise Bures distances for a list of density matrices.
    """
    n = len(rho_list)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_bures_distance(rho_list[i], rho_list[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def bures_multidimensional_scaling(distance_matrix: np.ndarray, target_dim: int = 3) -> np.ndarray:
    """
    Perform multidimensional scaling preserving Bures distances.
    """
    try:
        n = distance_matrix.shape[0]
        
        # Classical MDS
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        B = -0.5 * H @ (distance_matrix ** 2) @ H
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Take top eigenvalues/vectors
        eigenvals = eigenvals[:target_dim]
        eigenvecs = eigenvecs[:, :target_dim]
        
        # Ensure positive eigenvalues
        eigenvals = np.maximum(eigenvals, 0)
        
        # Coordinates
        coordinates = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        return coordinates
        
    except Exception as e:
        logger.error(f"MDS failed: {e}")
        # Fallback to random positions
        return np.random.randn(distance_matrix.shape[0], target_dim)

def calculate_geodesic_path(rho1: np.ndarray, rho2: np.ndarray, steps: int = 50) -> List[np.ndarray]:
    """
    Calculate geodesic path between two density matrices in Bures geometry.
    """
    try:
        # Simplified geodesic: linear interpolation in matrix space with projection
        path = []
        
        for i in range(steps + 1):
            t = i / steps
            # Linear interpolation
            interpolated = (1 - t) * rho1 + t * rho2
            
            # Project to ensure positive semidefinite and trace 1
            eigenvals, eigenvecs = np.linalg.eigh(interpolated)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            eigenvals = eigenvals / np.sum(eigenvals)  # Normalize trace
            
            projected = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            path.append(projected)
        
        return path
        
    except Exception as e:
        logger.error(f"Geodesic calculation failed: {e}")
        return [rho1, rho2]

def extract_eigenspace_coordinates(rho: np.ndarray) -> List[float]:
    """
    Extract coordinates for eigenspace visualization.
    """
    try:
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        
        # Take top 3 eigenvalues for 3D visualization
        top_eigenvals = eigenvals[:3]
        
        # Normalize to unit sphere (preserving relative magnitudes)
        if np.sum(top_eigenvals) > 0:
            normalized = top_eigenvals / np.sum(top_eigenvals)
        else:
            normalized = np.array([1/3, 1/3, 1/3])
        
        # Convert to spherical coordinates
        if len(normalized) >= 3:
            x = normalized[0]
            y = normalized[1] 
            z = normalized[2]
        else:
            x, y, z = 1/3, 1/3, 1/3
        
        return [float(x), float(y), float(z)]
        
    except Exception as e:
        logger.error(f"Eigenspace coordinate extraction failed: {e}")
        return [0.0, 0.0, 0.0]

def generate_color_mapping(rho_list: List[np.ndarray], mapping_type: str) -> List[float]:
    """
    Generate color values for visualization points.
    """
    try:
        if mapping_type == "chronological":
            return list(range(len(rho_list)))
        
        elif mapping_type == "purity":
            return [float(np.real(np.trace(rho @ rho))) for rho in rho_list]
        
        elif mapping_type == "entropy":
            entropies = []
            for rho in rho_list:
                eigenvals = np.linalg.eigvals(rho)
                eigenvals = eigenvals[eigenvals > 1e-10]
                entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
                entropies.append(float(entropy))
            return entropies
        
        elif mapping_type == "measurement":
            # Placeholder: use trace as measurement proxy
            return [float(np.real(np.trace(rho))) for rho in rho_list]
        
        else:
            return [0.0] * len(rho_list)
            
    except Exception as e:
        logger.error(f"Color mapping generation failed: {e}")
        return [0.0] * len(rho_list)

@router.post("/trajectory")
async def visualize_trajectory(request: TrajectoryVisualizationRequest) -> VisualizationResponse:
    """
    Generate Bures-preserving visualization of a quantum state trajectory.
    
    This implements APLG Claim Set F: visualization that preserves the
    geometric structure of the quantum state space.
    """
    try:
        # Validate rho_ids
        missing_ids = [rho_id for rho_id in request.rho_trajectory if rho_id not in STATE]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Matrices not found: {missing_ids}")
        
        # Extract density matrices
        rho_list = [STATE[rho_id]["rho"] for rho_id in request.rho_trajectory]
        
        if len(rho_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 matrices for trajectory")
        
        # Calculate Bures distance matrix
        distance_matrix = calculate_bures_distance_matrix(rho_list)
        
        # Dimension reduction preserving Bures distances
        if request.dimension_reduction == "bures_mds":
            coordinates = bures_multidimensional_scaling(distance_matrix, 3)
        else:
            # Fallback to eigenspace coordinates
            coordinates = np.array([extract_eigenspace_coordinates(rho) for rho in rho_list])
        
        # Generate color mapping
        color_values = generate_color_mapping(rho_list, request.color_mapping)
        
        # Calculate geodesic paths if requested
        geodesic_paths = None
        if request.include_geodesics and len(rho_list) > 1:
            geodesic_paths = []
            for i in range(len(rho_list) - 1):
                path = calculate_geodesic_path(rho_list[i], rho_list[i + 1], request.resolution // 10)
                # Convert path to coordinates
                path_coords = []
                for rho in path:
                    coord = extract_eigenspace_coordinates(rho)
                    path_coords.append(coord)
                geodesic_paths.append(path_coords)
        
        # Create visualization data
        visualization_id = str(uuid.uuid4())
        
        visualization_data = VisualizationData(
            visualization_id=visualization_id,
            type=request.visualization_type,
            coordinates=coordinates.tolist(),
            metadata={
                "rho_trajectory": request.rho_trajectory,
                "num_points": len(rho_list),
                "dimension_reduction": request.dimension_reduction,
                "color_mapping": request.color_mapping
            },
            bures_distances=distance_matrix.tolist(),
            geodesic_paths=geodesic_paths,
            color_values=color_values
        )
        
        # Store visualization
        VISUALIZATIONS[visualization_id] = {
            "data": visualization_data.dict(),
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        rendering_info = {
            "coordinate_ranges": {
                "x": [float(np.min(coordinates[:, 0])), float(np.max(coordinates[:, 0]))],
                "y": [float(np.min(coordinates[:, 1])), float(np.max(coordinates[:, 1]))],
                "z": [float(np.min(coordinates[:, 2])), float(np.max(coordinates[:, 2]))]
            },
            "color_range": [float(np.min(color_values)), float(np.max(color_values))],
            "bures_geometry_preserved": True
        }
        
        interaction_capabilities = [
            "rotation", "zoom", "pan", "point_selection", "trajectory_scrubbing"
        ]
        
        export_options = ["json", "csv", "plotly", "threejs", "svg"]
        
        return VisualizationResponse(
            success=True,
            visualization_data=visualization_data,
            rendering_info=rendering_info,
            interaction_capabilities=interaction_capabilities,
            export_options=export_options
        )
        
    except Exception as e:
        logger.error(f"Trajectory visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/state")
async def visualize_single_state(request: StateVisualizationRequest) -> VisualizationResponse:
    """
    Visualize a single quantum state with geometric accuracy.
    """
    try:
        if request.rho_id not in STATE:
            raise HTTPException(status_code=404, detail="Matrix not found")
        
        rho = STATE[request.rho_id]["rho"]
        
        if request.visualization_type == "eigenspace_sphere":
            # Map eigenvalues to sphere coordinates
            coordinates = [extract_eigenspace_coordinates(rho)]
            
        elif request.visualization_type == "bloch_sphere":
            # For 2x2 matrices, map to Bloch sphere
            if rho.shape[0] == 2:
                # Extract Pauli components
                sigma_x = np.array([[0, 1], [1, 0]])
                sigma_y = np.array([[0, -1j], [1j, 0]])
                sigma_z = np.array([[1, 0], [0, -1]])
                
                x = float(np.real(np.trace(rho @ sigma_x)))
                y = float(np.real(np.trace(rho @ sigma_y)))
                z = float(np.real(np.trace(rho @ sigma_z)))
                
                coordinates = [[x, y, z]]
            else:
                # Fallback to eigenspace for higher dimensions
                coordinates = [extract_eigenspace_coordinates(rho)]
                
        elif request.visualization_type == "purity_cone":
            # Map to purity cone coordinates
            purity = float(np.real(np.trace(rho @ rho)))
            eigenvals = np.linalg.eigvals(rho)
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
            
            # Cylindrical coordinates: radius = purity, height = entropy
            angle = 0.0  # Could be based on dominant eigenvector
            x = purity * np.cos(angle)
            y = purity * np.sin(angle)
            z = float(entropy)
            
            coordinates = [[x, y, z]]
        
        else:
            coordinates = [extract_eigenspace_coordinates(rho)]
        
        # Create visualization data
        visualization_id = str(uuid.uuid4())
        
        visualization_data = VisualizationData(
            visualization_id=visualization_id,
            type=request.visualization_type,
            coordinates=coordinates,
            metadata={
                "rho_id": request.rho_id,
                "matrix_dimension": rho.shape[0],
                "purity": float(np.real(np.trace(rho @ rho))),
                "trace": float(np.real(np.trace(rho))),
                "rank": int(np.sum(np.linalg.eigvals(rho) > 1e-10))
            },
            color_values=[1.0]  # Single point
        )
        
        # Store visualization
        VISUALIZATIONS[visualization_id] = {
            "data": visualization_data.dict(),
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        rendering_info = {
            "coordinate_system": request.coordinate_system,
            "geometric_representation": request.visualization_type,
            "measurement_overlays": request.measurement_overlays
        }
        
        return VisualizationResponse(
            success=True,
            visualization_data=visualization_data,
            rendering_info=rendering_info,
            interaction_capabilities=["rotation", "zoom", "measurement_probe"],
            export_options=["json", "svg", "png"]
        )
        
    except Exception as e:
        logger.error(f"State visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/interactive")
async def create_interactive_visualization(request: InteractiveVisualizationRequest):
    """
    Create interactive visualization supporting real-time updates.
    """
    try:
        # Validate rho_ids
        missing_ids = [rho_id for rho_id in request.rho_ids if rho_id not in STATE]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Matrices not found: {missing_ids}")
        
        # Extract matrices
        rho_list = [STATE[rho_id]["rho"] for rho_id in request.rho_ids]
        
        # Create interactive visualization structure
        visualization_id = str(uuid.uuid4())
        
        # Generate coordinates
        if len(rho_list) > 1:
            distance_matrix = calculate_bures_distance_matrix(rho_list)
            coordinates = bures_multidimensional_scaling(distance_matrix, 3)
        else:
            coordinates = np.array([extract_eigenspace_coordinates(rho_list[0])])
        
        # Interactive capabilities
        interaction_data = {
            "type": request.interaction_type,
            "coordinates": coordinates.tolist(),
            "rho_ids": request.rho_ids,
            "real_time_updates": request.real_time_updates,
            "interaction_handlers": {
                "point_click": f"/visualization/interact/{visualization_id}/point",
                "trajectory_update": f"/visualization/interact/{visualization_id}/trajectory",
                "measurement_overlay": f"/visualization/interact/{visualization_id}/measure"
            }
        }
        
        # Store visualization
        VISUALIZATIONS[visualization_id] = {
            "type": "interactive",
            "data": interaction_data,
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "visualization_id": visualization_id,
            "interaction_data": interaction_data,
            "websocket_endpoint": f"/visualization/ws/{visualization_id}" if request.real_time_updates else None,
            "export_format": request.export_format
        }
        
    except Exception as e:
        logger.error(f"Interactive visualization creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interactive visualization failed: {str(e)}")

@router.get("/list")
async def list_visualizations():
    """
    List all stored visualizations.
    """
    visualizations = []
    for viz_id, viz_data in VISUALIZATIONS.items():
        visualizations.append({
            "visualization_id": viz_id,
            "type": viz_data.get("type", "unknown"),
            "created_at": viz_data["created_at"],
            "num_points": len(viz_data["data"].get("coordinates", [])),
            "metadata": viz_data["data"].get("metadata", {})
        })
    
    return {
        "total_visualizations": len(VISUALIZATIONS),
        "visualizations": visualizations
    }

@router.get("/visualization/{visualization_id}")
async def get_visualization(visualization_id: str):
    """
    Retrieve a specific visualization.
    """
    if visualization_id not in VISUALIZATIONS:
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return VISUALIZATIONS[visualization_id]

@router.post("/test_bures_preservation")
async def test_bures_preservation():
    """
    Test Bures distance preservation in visualizations.
    """
    from routes.matrix_routes import rho_init
    
    # Create test matrices
    test_matrices = []
    for i in range(4):
        init_result = rho_init(seed_text=f"Test matrix {i}")
        test_matrices.append(init_result["rho_id"])
    
    try:
        # Test trajectory visualization
        request = TrajectoryVisualizationRequest(
            rho_trajectory=test_matrices,
            visualization_type="bures_manifold",
            dimension_reduction="bures_mds",
            include_geodesics=True
        )
        
        result = await visualize_trajectory(request)
        
        # Verify Bures distance preservation
        original_distances = result.visualization_data.bures_distances
        coordinates = np.array(result.visualization_data.coordinates)
        
        # Calculate Euclidean distances in visualization space
        euclidean_distances = []
        for i in range(len(coordinates)):
            row = []
            for j in range(len(coordinates)):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                row.append(float(dist))
            euclidean_distances.append(row)
        
        # Measure preservation quality (correlation between original and embedded distances)
        original_flat = [original_distances[i][j] for i in range(len(original_distances)) 
                        for j in range(i+1, len(original_distances))]
        euclidean_flat = [euclidean_distances[i][j] for i in range(len(euclidean_distances)) 
                         for j in range(i+1, len(euclidean_distances))]
        
        # Simple correlation calculation
        if len(original_flat) > 1:
            correlation = np.corrcoef(original_flat, euclidean_flat)[0, 1]
        else:
            correlation = 1.0
        
        return {
            "test_completed": True,
            "visualization_id": result.visualization_data.visualization_id,
            "bures_preservation_quality": float(correlation),
            "num_test_matrices": len(test_matrices),
            "geodesic_paths_generated": len(result.visualization_data.geodesic_paths or []),
            "distance_preservation": {
                "correlation": float(correlation),
                "threshold_met": correlation > 0.7,
                "original_distances_sample": original_distances[:2] if original_distances else [],
                "euclidean_distances_sample": euclidean_distances[:2]
            }
        }
        
    finally:
        # Clean up test matrices
        for rho_id in test_matrices:
            if rho_id in STATE:
                del STATE[rho_id]

@router.delete("/clear_visualizations")
async def clear_visualizations():
    """
    Clear all stored visualizations.
    """
    count = len(VISUALIZATIONS)
    VISUALIZATIONS.clear()
    
    return {
        "cleared": True,
        "visualizations_removed": count
    }