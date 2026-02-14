"""
Modal script for Gemma-2-2B humor experiment.
Extended timeout for full experiment with Dataset A/B comparison.
"""
import modal

app = modal.App("humor-gemma-experiment")

# ---------------------------------------------------------------------------
# Volumes (persistent storage)
# ---------------------------------------------------------------------------
results_vol = modal.Volume.from_name("humor-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("humor-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: Include local code directly
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zip")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.45.0",
        "datasets==3.0.0",
        "transformer-lens==2.8.0",
        "typeguard",              # <-- add this
        "scikit-learn==1.5.0",
        "matplotlib==3.9.0",
        "seaborn==0.13.0",
        "pandas==2.2.0",
        "openpyxl==3.1.0",
        "numpy==1.26.4",
    )
    .add_local_dir("src", remote_path="/root/humor_project/src")
    .add_local_dir("datasets", remote_path="/root/humor_project/datasets")
)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
SHARED_KWARGS = dict(
    image=image,
    gpu="A100-80GB",  # A100 40GB - plenty for Gemma-2-2B
    cpu=8.0,
    memory=65536,  # 48 GB (increased for safety)
    timeout=5 * 3600,  # 5 hours max (was 3, now extended)
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/humor_project/results": results_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)

DOWNLOAD_KWARGS = dict(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=30 * 60,  # 30 min
    volumes={
        "/root/humor_project/results": results_vol,
    },
)


# ---------------------------------------------------------------------------
# Experiment function (runs on GPU)
# ---------------------------------------------------------------------------
@app.function(**SHARED_KWARGS)
def run_experiment(model_name: str = "gemma-2-2b"):
    """
    Run the complete humor experiment.
    
    Timeline (Gemma-2-2B):
    - Model loading: ~5 min
    - Dataset A extraction + probing: ~60-75 min
    - Dataset B extraction + probing: ~60-75 min
    - Rank analysis: ~10 min
    - Interventions: ~45-60 min
    Total: ~3-4 hours (5 hour timeout for safety)
    """
    import sys
    import os
    from datetime import datetime
    
    start_time = datetime.now()
    print("="*70)
    print(f"GEMMA-2-2B HUMOR EXPERIMENT")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeout: 5 hours")
    print("="*70)
    
    # Set working directory and path
    os.chdir("/root/humor_project")
    sys.path.insert(0, "/root/humor_project/src")

    
    # Import and run
    from experiment import run_experiment as exp
    from intervention_tests import run_interventions
    
    # Step 1: Probing + Rank Analysis (includes Dataset A & B)
    print("\n[1/2] Running experiment (probing + Dataset A/B comparison)...")
    print("      This will take ~2-2.5 hours")
    exp_results = exp(model_name=model_name)
    
    # Step 2: Interventions
    print("\n[2/2] Running interventions (steering + ablation)...")
    print("      This will take ~45-60 minutes")
    int_results = run_interventions(model_name=model_name)
    
    # Commit results to volume
    print("\nCommitting results to volume...")
    results_vol.commit()
    hf_cache_vol.commit()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print(f"   Duration: {duration:.1f} minutes ({duration/60:.1f} hours)")
    print(f"   Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return {
        "status": "success",
        "model": model_name,
        "duration_minutes": duration
    }

# ---------------------------------------------------------------------------
# Download results function
# ---------------------------------------------------------------------------
@app.function(**DOWNLOAD_KWARGS)
def download_results(model_name: str = "gemma-2-2b"):
    """Create a zip and return it to the local machine."""
    import os
    import subprocess

    model_safe = model_name.replace("/", "_")
    results_dir = f"/root/humor_project/results/{model_name}"
    if not os.path.exists(results_dir):
        print(f"❌ No results found at {results_dir}")
        print("Available:", os.listdir("/root/humor_project/results"))
        return None

    zip_name = f"{model_safe}_complete_results.zip"
    tmp_zip = f"/tmp/{zip_name}"

    print(f"📦 Creating {zip_name}...")
    subprocess.run(
        ["zip", "-r", tmp_zip, model_name],
        cwd="/root/humor_project/results",
        check=True
    )

    with open(tmp_zip, "rb") as f:
        zip_bytes = f.read()

    print(f"✅ Created {len(zip_bytes)/1e6:.1f} MB archive")
    return {"zip_name": zip_name, "zip_data": zip_bytes}

# ---------------------------------------------------------------------------
# Check status function
# ---------------------------------------------------------------------------
@app.function(**DOWNLOAD_KWARGS)
def check_status(model_name: str = "gemma-2-2b"):
    """Check if results exist and show key findings."""
    import os
    import json
    
    results_path = f"/root/humor_project/results/{model_name}/rank_analysis.json"
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        
        print("\n✅ EXPERIMENT COMPLETE!")
        print(f"   Best layer: {data.get('best_layer')}")
        print(f"   Best accuracy: {data.get('best_probe_accuracy', 0):.1%}")
        print(f"   Rank for 90% variance: {data.get('rank_90')}")
        print(f"   Fisher ratio: {data.get('fisher_ratio', 0):.2f}")
        
        return {"status": "complete", "results": data}
    else:
        return {"status": "not_found", "message": "Experiment still running or not started"}

# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(model: str = "gemma-2-2b", detach: bool = False):
    """
    Run experiment.
    
    Usage:
        modal run modal_gemma.py --detach              # Run Gemma-2-2B in background
        modal run modal_gemma.py --model gpt2          # Run GPT-2 (faster)
        modal run modal_gemma.py                       # Run and wait
        modal run modal_gemma.py::check_status         # Check if complete
        modal run modal_gemma.py::download             # Download results
    """
    print("🚀 MODAL HUMOR EXPERIMENT")
    print("="*70)
    print(f"Model: {model}")
    print(f"GPU: A100 80GB")
    print(f"Timeout: 5 hours")
    print(f"Mode: {'Detached (background)' if detach else 'Interactive'}")
    print("="*70)
    
    if model == "gemma-2-2b":
        print("\n⏱️  ESTIMATED TIME:")
        print("   • Dataset A extraction + probing: 60-75 min")
        print("   • Dataset B extraction + probing: 60-75 min")
        print("   • Rank analysis: 10 min")
        print("   • Interventions: 45-60 min")
        print("   • TOTAL: ~3-4 hours")
        print("\n💰 ESTIMATED COST: $9-12 (A100 @ $3/hr for 3-4 hrs)")
    elif model == "gpt2":
        print("\n⏱️  ESTIMATED TIME: 25-30 minutes")
        print("💰 ESTIMATED COST: $1.50")
    
    print("="*70)
    print()
    
    if detach:
        # Spawn and return immediately
        print("⏳ Launching in detached mode...")
        call = run_experiment.spawn(model_name=model)
        print(f"✅ Spawned! Function call ID: {call.object_id}")
        print()
        print("Next steps:")
        print("  1. Check status: modal run modal_gemma.py::check_status")
        print("  2. Download when done: modal run modal_gemma.py::download")
        print()
    else:
        # Run and wait
        print("⏳ Running experiment (will take 3-4 hours)...")
        print("   You can Ctrl+C to detach (experiment continues)")
        print()
        result = run_experiment.remote(model_name=model)
        print(f"\n✅ Complete: {result}")
        print("\nDownload results with: modal run modal_gemma.py::download")

# ---------------------------------------------------------------------------
# Download entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def download(model: str = "gemma-2-2b"):
    """Download results to local machine."""
    print(f"📥 Downloading results for {model}...")

    result = download_results.remote(model_name=model)
    if result is None:
        print("❌ No results found. Has the experiment completed?")
        return

    zip_name = result.get("zip_name", f"{model.replace('/', '_')}_complete_results.zip")
    zip_bytes = result["zip_data"]

    with open(zip_name, "wb") as f:
        f.write(zip_bytes)

    import os
    print(f"\n✅ Saved to: {os.path.abspath(zip_name)}")
