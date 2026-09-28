from pydantic import BaseModel, Field
from typing import List, Optional, Any

class ProblemDef(BaseModel):
    num_vars: int
    num_constrs: int
    obj_coeffs: List[float]
    row_ptr: List[int]
    col_idx: List[int]
    values: List[float]
    row_senses: str
    rhs: List[float]
    var_lower_bounds: Optional[List[float]] = None
    var_upper_bounds: Optional[List[float]] = None
    is_integer: Optional[List[bool]] = None
    method: str = "auto"
    gpu: bool = True

class SolveRequest(BaseModel):
    mps_content: Optional[str] = None
    problem_def: Optional[ProblemDef] = None
    filename: Optional[str] = None
    method: str = "auto"
    gpu: bool = True

class SolveResponse(BaseModel):
    status: str
    solver_used: Optional[str] = None
    objective: Optional[float] = None
    solution: Optional[List[float]] = None
    wall_time_ms: Optional[float] = None
    iterations: Optional[int] = None
    mock: Optional[bool] = None
    message: Optional[str] = None
    trace: Optional[List[dict]] = None

class InspectResponse(BaseModel):
    vars: int
    constrs: int
    is_milp: bool
    sense: str
    problem_summary: str

class BenchmarkResult(BaseModel):
    problem: str
    status: str
    solver_used: Optional[str] = None
    objective: Optional[float] = None
    reference: Optional[float] = None
    reference_time_ms: Optional[float] = None
    difference: Optional[float] = None
    passed: Optional[bool] = None
    wall_time_ms: Optional[float] = None
    iterations: Optional[int] = None
    mock: Optional[bool] = None
    error_message: Optional[str] = None

class BenchmarkResponse(BaseModel):
    benchmark_results: List[BenchmarkResult]

class StreamUpdate(BaseModel):
    type: str = "update"
    iteration: int
    primal_obj: float
    dual_obj: float
    elapsed_ms: float
    mock: Optional[bool] = None

class StreamResult(BaseModel):
    type: str = "result"
    status: str
    solver_used: Optional[str] = None
    objective: Optional[float] = None
    solution: Optional[List[float]] = None
    wall_time_ms: Optional[float] = None
    iterations: Optional[int] = None
    mock: Optional[bool] = None

class StreamError(BaseModel):
    type: str = "error"
    message: str
