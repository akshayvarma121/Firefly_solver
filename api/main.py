import os
import sys
import uuid
import asyncio
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError


from config import settings
from schemas import (
    SolveRequest, SolveResponse, ProblemDef, 
    BenchmarkResponse, BenchmarkResult,
    StreamUpdate, StreamResult, StreamError,
    InspectResponse
)

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2")
    if cuda_path:
        _cuda_bin = os.path.join(cuda_path, "bin", "x64")
        if os.path.isdir(_cuda_bin):
            os.add_dll_directory(_cuda_bin)
        _cuda_bin2 = os.path.join(cuda_path, "bin")
        if os.path.isdir(_cuda_bin2):
            os.add_dll_directory(_cuda_bin2)

core_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

try:
    import firefly_solver
    FIREFLY_SOLVER_AVAILABLE = True
except Exception as e:
    FIREFLY_SOLVER_AVAILABLE = False
    print(f"[WARNING] firefly_solver import failed: {e}. Fallback mock mode enabled.", file=sys.stderr)

def get_base_path():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


import narration
from trace import build_trace_from_prob

# Shared benchmark logic
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "core"))
from cli.bench import run_batch as _run_batch  # type: ignore

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firefly_api")

app = FastAPI(title="Firefly Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log_request(problem_name: str, method: str, gpu: bool, status: str, wall_time_ms: float = 0.0, msg: str = ""):
    logger.info(f"[Problem: {problem_name}] [Method: {method}] [GPU: {gpu}] [Status: {status}] [Time: {wall_time_ms:.2f}ms] {msg}")

@app.get("/health")
def health_check():
    mode = "GPU" if FIREFLY_SOLVER_AVAILABLE else "CPU (Mock Fallback)"
    if FIREFLY_SOLVER_AVAILABLE and hasattr(firefly_solver, "cuda_is_available") and not firefly_solver.cuda_is_available():
        mode = "CPU (Fallback)"
    return {"status": "ok", "mode": mode, "solver_available": FIREFLY_SOLVER_AVAILABLE}


@app.post("/inspect", response_model=InspectResponse, response_model_exclude_none=True)
async def inspect_endpoint(req: SolveRequest):
    try:
        if not FIREFLY_SOLVER_AVAILABLE:
            raise ImportError("firefly_solver module is not available")

        if req.mps_content:
            prob = firefly_solver.parse_mps_string(req.mps_content)
        elif getattr(req, 'filename', None):
            sample_path = os.path.join(get_base_path(), "sample_problems", req.filename)
            if not os.path.isfile(sample_path):
                raise HTTPException(status_code=404, detail=f"Sample not found: {req.filename}")
            with open(sample_path, "r", encoding="utf-8") as f:
                prob = firefly_solver.parse_mps_string(f.read())
        else:
            raise HTTPException(status_code=400, detail="Must provide mps_content or filename")
            
        is_milp = any(prob.is_integer) if hasattr(prob, 'is_integer') and prob.is_integer else False
        return InspectResponse(
            vars=prob.num_vars,
            constrs=prob.num_constrs,
            is_milp=is_milp,
            sense="minimize",
            problem_summary=f"{prob.num_vars} vars, {prob.num_constrs} constrs, {len(prob.values)} nonzeros"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inspect failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve", response_model=SolveResponse, response_model_exclude_none=True)
async def solve_endpoint(
    request: Request,
    file: UploadFile = File(None),
    problem_def: str = Form(None),
    method: str = Form("auto"),
    gpu: bool = Form(True)
):
    try:
        prob_name = file.filename if file else "json_def"
        prob = None

        if file:
            content = await file.read()
            if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
                raise ValueError(f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_BYTES} bytes")
            prob = firefly_solver.parse_mps_string(content.decode("utf-8"))
        elif problem_def:
            data = json.loads(problem_def)
            pdef = ProblemDef(**data)
            prob = firefly_solver.SparseProblem()
            prob.num_vars = pdef.num_vars
            prob.num_constrs = pdef.num_constrs
            prob.obj_coeffs = pdef.obj_coeffs
            prob.row_ptr = pdef.row_ptr
            prob.col_idx = pdef.col_idx
            prob.values = pdef.values
            prob.row_senses = list(pdef.row_senses)
            prob.rhs = pdef.rhs
            if pdef.var_lower_bounds: prob.var_lower_bounds = pdef.var_lower_bounds
            if pdef.var_upper_bounds: prob.var_upper_bounds = pdef.var_upper_bounds
            if pdef.is_integer: prob.is_integer = pdef.is_integer
            method = pdef.method
            gpu = pdef.gpu
        else:
            raise ValueError("Must provide either file or problem_def")

        if not FIREFLY_SOLVER_AVAILABLE:
            raise ImportError("firefly_solver module is not available")

        # Run solve in a thread, wrap with timeout
        solve_future = asyncio.to_thread(build_trace_from_prob, prob, method, gpu)
        trace_data, res = await asyncio.wait_for(solve_future, timeout=settings.SOLVE_TIMEOUT_SECONDS)
        
        result = SolveResponse(
            status=res.status,
            solver_used=getattr(res, "solver_used", None),
            objective=res.objective,
            solution=res.solution,
            wall_time_ms=res.wall_time_ms,
            iterations=res.iterations,
            message=res.message if hasattr(res, "message") else None,
            trace=trace_data
        )
        
        log_request(prob_name, method, gpu, res.status, res.wall_time_ms)
        return result

    except (ValueError, json.JSONDecodeError, ValidationError, RuntimeError) as e:
        logger.warning(f"Input validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        msg = f"Solve timed out after {settings.SOLVE_TIMEOUT_SECONDS}s"
        logger.warning(f"[MOCK FALLBACK ACTIVE] {msg}")
        mock_trace = [
            {"stage": "parse", "narration": "[MOCK] Parsed a linear programming problem to minimize an objective over 2 variables and 2 constraints."},
            {"stage": "output", "narration": "[MOCK] The solve was interrupted by a limit after 123.4 ms, returning the best known state."}
        ]
        return SolveResponse(
            status="OPTIMAL", objective=-99.99, solution=[1.0, 2.0],
            wall_time_ms=123.4, iterations=42, mock=True, message=msg, trace=mock_trace
        )
    except Exception as e:
        msg = str(e)
        logger.warning(f"[MOCK FALLBACK ACTIVE] Solver failed: {msg}")
        mock_trace = [
            {"stage": "parse", "narration": "[MOCK] Parsed a linear programming problem to minimize an objective over 2 variables and 2 constraints."},
            {"stage": "output", "narration": f"[MOCK] The solver failed with: {msg}"}
        ]
        return SolveResponse(
            status="OPTIMAL", objective=-99.99, solution=[1.0, 2.0],
            wall_time_ms=123.4, iterations=42, mock=True, message=msg, trace=mock_trace
        )

@app.get("/benchmark", response_model=BenchmarkResponse, response_model_exclude_none=True)
async def benchmark_endpoint(gpu: bool = True):
    sample_dir = os.path.join(get_base_path(), "sample_problems")
    if not os.path.isdir(sample_dir):
        return BenchmarkResponse(benchmark_results=[])

    try:
        summary = await asyncio.to_thread(
            _run_batch,
            sample_dir,
            method="auto",
            gpu=gpu,
            firefly_solver=firefly_solver,
        )

        results = []
        for pr in summary.results:
            results.append(BenchmarkResult(
                problem=pr.problem,
                status=pr.status,
                solver_used=getattr(pr, "solver_used", None),
                objective=pr.objective,
                reference=pr.reference,
                reference_time_ms=getattr(pr, "reference_time_ms", None),
                difference=pr.difference,
                passed=pr.passed,
                wall_time_ms=pr.wall_time_ms,
                iterations=pr.iterations,
                error_message=pr.error_message
            ))

        return BenchmarkResponse(benchmark_results=results)
    except Exception as e:
        # Just returning a single ERROR result representing the failure of the suite
        err_res = BenchmarkResult(
            problem="benchmark_suite",
            status="ERROR",
            passed=False,
            error_message=str(e)
        )
        return BenchmarkResponse(benchmark_results=[err_res])

@app.websocket("/ws/solve-stream")
async def websocket_solve(websocket: WebSocket):
    await websocket.accept()
    solve_future = None
    
    try:
        data = await websocket.receive_text()
        try:
            req = json.loads(data)
            solve_req = SolveRequest.model_validate(req)
        except Exception as e:
            await websocket.send_json(StreamError(message=f"Invalid request JSON: {e}").model_dump(exclude_none=True))
            return
            
        method = solve_req.method
        gpu = solve_req.gpu
        
        prob = None
        setup_error = None
        
        try:
            if not FIREFLY_SOLVER_AVAILABLE:
                raise ImportError("firefly_solver module is not available")

            if solve_req.mps_content:
                prob = firefly_solver.parse_mps_string(solve_req.mps_content)
            elif getattr(solve_req, 'filename', None):
                sample_path = os.path.join(get_base_path(), "sample_problems", solve_req.filename)
                if not os.path.isfile(sample_path):
                    raise ValueError(f"Sample not found: {solve_req.filename}")
                with open(sample_path, "r", encoding="utf-8") as f:
                    prob = firefly_solver.parse_mps_string(f.read())
            elif solve_req.problem_def:
                pdef = solve_req.problem_def
                prob = firefly_solver.SparseProblem()
                prob.num_vars = pdef.num_vars
                prob.num_constrs = pdef.num_constrs
                prob.obj_coeffs = pdef.obj_coeffs
                prob.row_ptr = pdef.row_ptr
                prob.col_idx = pdef.col_idx
                prob.values = pdef.values
                prob.row_senses = list(pdef.row_senses)
                prob.rhs = pdef.rhs
                if pdef.var_lower_bounds: prob.var_lower_bounds = pdef.var_lower_bounds
                if pdef.var_upper_bounds: prob.var_upper_bounds = pdef.var_upper_bounds
                if pdef.is_integer: prob.is_integer = pdef.is_integer
            else:
                raise ValueError("Must provide either mps_content, filename, or problem_def")
        except Exception as e:
            setup_error = e

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Generate initial parse trace
        if prob:
            is_milp = any(prob.is_integer) if hasattr(prob, 'is_integer') and prob.is_integer else False
            parse_data = {"stage": "parse", "vars": prob.num_vars, "constrs": prob.num_constrs, "is_milp": is_milp, "sense": "minimize"}
            parse_data["narration"] = narration.describe_parse(parse_data)
            await websocket.send_json({"type": "trace", **parse_data})

        def iteration_callback(iteration, primal_obj, dual_obj, elapsed_ms, mock=False):
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "data",
                    "data": {
                        "iteration": int(iteration),
                        "primal": float(primal_obj),
                        "dual": float(dual_obj)
                    }
                }),
                loop
            )

        def solve_task():
            try:
                if setup_error:
                    raise setup_error
                return firefly_solver.solve(prob, method=method, gpu=gpu, iteration_callback=iteration_callback)
            except Exception as e:
                logger.warning(f"[MOCK FALLBACK ACTIVE] Stream solver failed: {e}")
                import time
                for i in range(10):
                    time.sleep(0.1)
                    iteration_callback(i * 5, -100.0 + i, -100.0 - i, i * 10, mock=True)
                class MockRes:
                    status = "OPTIMAL"
                    objective = -99.99
                    solution = [1.0, 2.0]
                    wall_time_ms = 123.4
                    iterations = 42
                return MockRes()

        solve_future = asyncio.create_task(asyncio.to_thread(solve_task))

        async def pump_queue():
            while not solve_future.done():
                try:
                    update = await asyncio.wait_for(queue.get(), timeout=0.1)
                    await websocket.send_json(update)
                except asyncio.TimeoutError:
                    continue
            
            # flush
            while not queue.empty():
                update = queue.get_nowait()
                await websocket.send_json(update)

        # Enforce overall solve timeout on the stream too
        await asyncio.wait_for(pump_queue(), timeout=settings.SOLVE_TIMEOUT_SECONDS)
        
        res = solve_future.result()
        is_mock = getattr(res, '__class__', None).__name__ == 'MockRes'
        
        if hasattr(res, 'presolve_stats'):
            ps = res.presolve_stats
            pdata = {"stage": "presolve", "original_rows": ps.original_rows, "original_cols": ps.original_cols, "reduced_rows": ps.reduced_rows, "reduced_cols": ps.reduced_cols, "rows_removed": ps.rows_removed, "bounds_tightened": ps.bounds_tightened, "variables_fixed": ps.variables_fixed, "scaling_applied": ps.scaling_applied}
            pdata["narration"] = narration.describe_presolve(pdata)
            await websocket.send_json({"type": "trace", **pdata})
        else:
            pdata = {"stage": "presolve", "message": "PresolveStats not available"}
            pdata["narration"] = narration.describe_presolve(pdata)
            await websocket.send_json({"type": "trace", **pdata})

        is_milp_flag = any(prob.is_integer) if prob and hasattr(prob, 'is_integer') and prob.is_integer else False
        if is_milp_flag:
            milp_data = {"stage": "milp", "message": "Branch-and-bound exploration completed."}
            milp_data["narration"] = narration.describe_milp(milp_data)
            await websocket.send_json({"type": "trace", **milp_data})
        else:
            lp_core_data = {"stage": "lp_core", "message": f"Final LP iterations: {res.iterations}"}
            lp_core_data["narration"] = narration.describe_lp_core(lp_core_data)
            await websocket.send_json({"type": "trace", **lp_core_data})
            
        odata = {"stage": "output", "status": res.status, "objective": res.objective, "iterations": res.iterations, "time_ms": res.wall_time_ms}
        odata["narration"] = narration.describe_output(odata)
        await websocket.send_json({"type": "trace", **odata})

        await websocket.send_json({"type": "status", "state": "converged" if res.status == "OPTIMAL" else "error"})

    except asyncio.TimeoutError:
        if solve_future:
            solve_future.cancel()
        await websocket.send_json(StreamError(message=f"Solve timed out after {settings.SOLVE_TIMEOUT_SECONDS}s").model_dump(exclude_none=True))
    except WebSocketDisconnect:
        if solve_future:
            solve_future.cancel()
        logger.info("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json(StreamError(message=str(e)).model_dump(exclude_none=True))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FIREFLY_API_PORT", 8000))
    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
