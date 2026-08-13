import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PROJECT PATHS AND SHARED CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

sys.path.insert(0, str(TRAINING_DIR))
from training_config import MODEL_NAME, SEED  # noqa: E402


DATASET_SPECS = {
    "gptplus": {
        "dataset_label": "GPTPlus",
        "model_suffix": "GPTPlus-DS",
        "merged_dir": PROJECT_ROOT / "models" / "lfm2_700m_gptplus_merged",
    },
    "claude": {
        "dataset_label": "Claude",
        "model_suffix": "Claude-DS",
        "merged_dir": PROJECT_ROOT / "models" / "lfm2_700m_claude_merged",
    },
}

OUTPUT_DIR = PROJECT_ROOT / "models" / "gguf"
DEFAULT_LLAMA_CPP_DIR = Path.home() / "llama.cpp"
DEFAULT_QUANTIZATION = "Q4_K_M"

SUPPORTED_QUANTIZATIONS = (
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
)

DEFAULT_SMOKE_PROMPT = (
    "Generate two concise bank-transfer descriptions using only the provided "
    "information.\n"
    "Category: rent\n"
    "Beneficiary: Oak Residence\n"
    "Amount: EUR 850\n"
    "Reference period: August 2026\n\n"
)


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the selected merged LFM2-700M Hugging Face model to an "
            "F16 GGUF file, quantize it with llama.cpp, and run a final "
            "llama-cli smoke test."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["gptplus", "claude"],
        required=True,
        help=(
            "Fine-tuning dataset associated with the merged model to convert. "
            "After evaluation, pass the dataset corresponding to the best model."
        ),
    )

    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=DEFAULT_LLAMA_CPP_DIR,
        help=(
            "Path to the external llama.cpp repository. "
            "Default: ~/llama.cpp"
        ),
    )

    parser.add_argument(
        "--quantization",
        choices=SUPPORTED_QUANTIZATIONS,
        default=DEFAULT_QUANTIZATION,
        help=(
            "Final GGUF quantization. Project default: Q4_K_M."
        ),
    )

    parser.add_argument(
        "--smoke-max-tokens",
        type=int,
        default=48,
        help="Maximum tokens generated during the llama-cli smoke test. Default: 48.",
    )

    parser.add_argument(
        "--smoke-prompt",
        default=DEFAULT_SMOKE_PROMPT,
        help="Prompt used for the final llama-cli smoke test.",
    )

    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the final llama-cli smoke test. Not recommended for the final run.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of already existing GGUF/report files.",
    )

    return parser.parse_args()


# ============================================================
# NAMING
# ============================================================


def sanitize_filename_component(value):
    safe = []

    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")

    return "".join(safe).strip("_")



def get_base_model_short_name():
    # Example: LiquidAI/LFM2-700M -> LFM2-700M
    name = str(MODEL_NAME).rstrip("/").split("/")[-1]
    return sanitize_filename_component(name)



def build_output_paths(dataset_name, quantization):
    spec = DATASET_SPECS[dataset_name]
    base_name = get_base_model_short_name()
    model_name = f"{base_name}_{spec['model_suffix']}"
    quant_label = sanitize_filename_component(quantization.upper())

    return {
        "display_name": model_name,
        "f16": OUTPUT_DIR / f"{model_name}_F16.gguf",
        "quantized": OUTPUT_DIR / f"{model_name}_{quant_label}.gguf",
        "report": OUTPUT_DIR / f"{model_name}_conversion_report.json",
    }


# ============================================================
# VALIDATION
# ============================================================


def resolve_llama_cpp_tools(llama_cpp_dir):
    llama_cpp_dir = llama_cpp_dir.expanduser().resolve()

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    cli_binary = llama_cpp_dir / "build" / "bin" / "llama-cli"

    return {
        "root": llama_cpp_dir,
        "convert_script": convert_script,
        "quantize_binary": quantize_binary,
        "cli_binary": cli_binary,
    }



def validate_inputs(dataset_name, tools, output_paths, overwrite, skip_smoke_test):
    merged_dir = DATASET_SPECS[dataset_name]["merged_dir"]

    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Merged model directory not found:\n{merged_dir}\n\n"
            "Run post_training/merge_lora.py before conversion."
        )

    if not (merged_dir / "config.json").exists():
        raise FileNotFoundError(
            f"config.json not found in merged model directory:\n{merged_dir}"
        )

    if not any(merged_dir.glob("*.safetensors")):
        raise FileNotFoundError(
            f"No .safetensors model weights found in:\n{merged_dir}"
        )

    if not tools["root"].exists():
        raise FileNotFoundError(
            f"llama.cpp repository not found:\n{tools['root']}\n\n"
            "Clone/build llama.cpp on the EC2 instance first, or pass "
            "--llama-cpp-dir with the correct path."
        )

    if not tools["convert_script"].is_file():
        raise FileNotFoundError(
            f"llama.cpp conversion script not found:\n{tools['convert_script']}"
        )

    if not tools["quantize_binary"].is_file():
        raise FileNotFoundError(
            f"llama-quantize binary not found:\n{tools['quantize_binary']}\n\n"
            "Build llama.cpp before running this script."
        )

    if not os.access(tools["quantize_binary"], os.X_OK):
        raise PermissionError(
            f"llama-quantize is not executable:\n{tools['quantize_binary']}"
        )

    if not skip_smoke_test:
        if not tools["cli_binary"].is_file():
            raise FileNotFoundError(
                f"llama-cli binary not found:\n{tools['cli_binary']}\n\n"
                "Build llama.cpp before running the smoke test."
            )

        if not os.access(tools["cli_binary"], os.X_OK):
            raise PermissionError(
                f"llama-cli is not executable:\n{tools['cli_binary']}"
            )

    existing = [
        path
        for key, path in output_paths.items()
        if key in {"f16", "quantized", "report"} and Path(path).exists()
    ]

    if existing and not overwrite:
        existing_text = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "One or more output files already exist:\n"
            f"{existing_text}\n\n"
            "Use --overwrite only if you intentionally want to replace them."
        )


# ============================================================
# SUBPROCESS / REPRODUCIBILITY HELPERS
# ============================================================


def command_as_text(command):
    return " ".join(shlex.quote(str(part)) for part in command)



def run_command(command, cwd=None):
    print("\n$", command_as_text(command))

    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if completed.stdout:
        print(completed.stdout.rstrip())

    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)

    return {
        "command": [str(part) for part in command],
        "command_text": command_as_text(command),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }



def require_success(result, stage_name):
    if result["return_code"] != 0:
        raise RuntimeError(
            f"{stage_name} failed with return code {result['return_code']}."
        )



def get_llama_cpp_commit(llama_cpp_dir):
    try:
        completed = subprocess.run(
            ["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if completed.returncode == 0:
            return completed.stdout.strip()
    except OSError:
        pass

    return None



def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()



def file_metadata(path):
    path = Path(path)

    if not path.exists():
        return None

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }



def truncate_for_report(text, max_chars=12000):
    text = str(text or "")

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated in report]..."


# ============================================================
# CONVERSION / QUANTIZATION / SMOKE TEST
# ============================================================


def convert_to_f16(merged_dir, f16_output, tools):
    command = [
        sys.executable,
        tools["convert_script"],
        merged_dir,
        "--outfile",
        f16_output,
        "--outtype",
        "f16",
    ]

    result = run_command(command, cwd=tools["root"])
    require_success(result, "Hugging Face -> GGUF F16 conversion")

    if not f16_output.exists() or f16_output.stat().st_size == 0:
        raise RuntimeError(
            f"F16 conversion finished without a valid output file:\n{f16_output}"
        )

    return result



def quantize_gguf(f16_input, quantized_output, quantization, tools):
    command = [
        tools["quantize_binary"],
        f16_input,
        quantized_output,
        quantization,
    ]

    result = run_command(command, cwd=tools["root"])
    require_success(result, f"GGUF {quantization} quantization")

    if not quantized_output.exists() or quantized_output.stat().st_size == 0:
        raise RuntimeError(
            "Quantization finished without a valid output file:\n"
            f"{quantized_output}"
        )

    return result



def run_smoke_test(quantized_model, prompt, max_tokens, tools):
    command = [
        tools["cli_binary"],
        "-m",
        quantized_model,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--seed",
        str(SEED),
    ]

    result = run_command(command, cwd=tools["root"])
    require_success(result, "llama-cli smoke test")

    return result


# ============================================================
# REPORT
# ============================================================


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()



def save_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)



def compact_command_result(result):
    if result is None:
        return None

    return {
        "command": result["command"],
        "command_text": result["command_text"],
        "return_code": result["return_code"],
        "stdout": truncate_for_report(result["stdout"]),
        "stderr": truncate_for_report(result["stderr"]),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    if args.smoke_max_tokens <= 0:
        raise ValueError("--smoke-max-tokens must be positive.")

    tools = resolve_llama_cpp_tools(args.llama_cpp_dir)
    output_paths = build_output_paths(args.dataset, args.quantization)
    merged_dir = DATASET_SPECS[args.dataset]["merged_dir"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_inputs(
        dataset_name=args.dataset,
        tools=tools,
        output_paths=output_paths,
        overwrite=args.overwrite,
        skip_smoke_test=args.skip_smoke_test,
    )

    # If overwrite is enabled, remove only the outputs this run owns.
    if args.overwrite:
        for key in ("f16", "quantized", "report"):
            path = Path(output_paths[key])
            if path.exists():
                path.unlink()

    print("=" * 72)
    print("LFM2-700M GGUF CONVERSION + QUANTIZATION")
    print("=" * 72)
    print(f"\nBase model configuration: {MODEL_NAME}")
    print(f"Selected fine-tuned model: {output_paths['display_name']}")
    print(f"Merged Hugging Face input:\n{merged_dir}")
    print(f"llama.cpp repository:\n{tools['root']}")
    print(f"Intermediate GGUF: F16")
    print(f"Final quantization: {args.quantization}")
    print(f"Output directory:\n{OUTPUT_DIR}")

    report = {
        "status": "running",
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "project": "Edge AI for Smart Bank Transfers",
        "pipeline_stage": "I+J - GGUF conversion and quantization",
        "base_model": str(MODEL_NAME),
        "selected_dataset": args.dataset,
        "selected_dataset_label": DATASET_SPECS[args.dataset]["dataset_label"],
        "selected_model_label": output_paths["display_name"],
        "input_merged_model": str(merged_dir),
        "intermediate_format": "GGUF F16",
        "final_quantization": args.quantization,
        "llama_cpp": {
            "repository_path": str(tools["root"]),
            "git_commit": get_llama_cpp_commit(tools["root"]),
            "convert_script": str(tools["convert_script"]),
            "quantize_binary": str(tools["quantize_binary"]),
            "cli_binary": str(tools["cli_binary"]),
        },
        "outputs": {
            "f16_gguf": str(output_paths["f16"]),
            "quantized_gguf": str(output_paths["quantized"]),
            "conversion_report": str(output_paths["report"]),
        },
        "conversion": None,
        "quantization": None,
        "smoke_test": {
            "enabled": not args.skip_smoke_test,
            "prompt": args.smoke_prompt if not args.skip_smoke_test else None,
            "max_tokens": args.smoke_max_tokens if not args.skip_smoke_test else None,
            "temperature": 0 if not args.skip_smoke_test else None,
            "seed": SEED if not args.skip_smoke_test else None,
            "result": None,
        },
        "file_metadata": {},
        "error": None,
    }

    try:
        conversion_result = convert_to_f16(
            merged_dir=merged_dir,
            f16_output=output_paths["f16"],
            tools=tools,
        )
        report["conversion"] = compact_command_result(conversion_result)
        report["file_metadata"]["f16_gguf"] = file_metadata(output_paths["f16"])

        quantization_result = quantize_gguf(
            f16_input=output_paths["f16"],
            quantized_output=output_paths["quantized"],
            quantization=args.quantization,
            tools=tools,
        )
        report["quantization"] = compact_command_result(quantization_result)
        report["file_metadata"]["quantized_gguf"] = file_metadata(
            output_paths["quantized"]
        )

        if not args.skip_smoke_test:
            smoke_result = run_smoke_test(
                quantized_model=output_paths["quantized"],
                prompt=args.smoke_prompt,
                max_tokens=args.smoke_max_tokens,
                tools=tools,
            )
            report["smoke_test"]["result"] = compact_command_result(smoke_result)

        report["status"] = "success"

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise

    finally:
        report["finished_at_utc"] = utc_now_iso()
        save_report(output_paths["report"], report)

    print("\n" + "=" * 72)
    print("CONVERSION + QUANTIZATION COMPLETED")
    print("=" * 72)
    print(f"\nF16 GGUF:\n{output_paths['f16']}")
    print(f"\n{args.quantization} GGUF:\n{output_paths['quantized']}")
    print(f"\nTechnical report:\n{output_paths['report']}")

    if not args.skip_smoke_test:
        print("\nSmoke test: PASSED")


if __name__ == "__main__":
    main()
