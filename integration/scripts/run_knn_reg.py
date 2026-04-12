import os
import io
import sys
import time
import shutil
import subprocess
import pandas as pd

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
name_model = "knn_reg"
workspace  = os.environ["GITHUB_WORKSPACE"]
repository = os.environ["GITHUB_REPOSITORY"]

base_input_dir  = os.path.join(workspace, f"integration/input_{name_model}")
base_output_dir = os.path.join(workspace, f"integration/output_{name_model}")

input_versions = [
          "0.21.3",
          "0.22.0",
          "0.22.1",
          "0.23.0",
          "0.23.1",
          "0.23.2",
          "0.24.0",
          "0.24.1",
          "0.24.2",
          "1.0.0",
          "1.0.1",
          "1.0.2",
          "1.1.0",
          "1.1.1",
          "1.1.2",
          "1.1.3",
          "1.2.0",
          "1.2.1",
          "1.2.2",
          "1.3.0",
          "1.3.1",
          "1.3.2",
          "1.4.0",
          "1.4.2",
          "1.5.0",
          "1.5.1",
          "1.5.2",
          "1.6.0",
          "1.6.1",
          "1.7.0",
          "1.7.1",
          "1.7.2"
        ]

output_versions = [
          "0.21.3",
          "0.22.0",
          "0.22.1",
          "0.23.0",
          "0.23.1",
          "0.23.2",
          "0.24.0",
          "0.24.1",
          "0.24.2",
          "1.0.0",
          "1.0.1",
          "1.0.2",
          "1.1.0",
          "1.1.1",
          "1.1.2",
          "1.1.3",
          "1.2.0",
          "1.2.1",
          "1.2.2",
          "1.3.0",
          "1.3.1",
          "1.3.2",
          "1.4.0",
          "1.4.2",
          "1.5.0",
          "1.5.1",
          "1.5.2",
          "1.6.0",
          "1.6.1",
          "1.7.0",
          "1.7.1",
          "1.7.2"
        ]

REGISTRY = f"ghcr.io/{repository}"

# -----------------------------------------------------------------------
# Tee
# -----------------------------------------------------------------------
class Tee:
    def __init__(self, file):
        self.file   = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# -----------------------------------------------------------------------
# container_execution
# -----------------------------------------------------------------------
def container_execution(inp_ver, out_ver):
    print(f"INPUT: {inp_ver}")
    print(f"OUTPUT: {out_ver}")

    name_input_image  = f"{REGISTRY}/sklearn-migrator-input-{inp_ver}:latest"
    name_output_image = f"{REGISTRY}/sklearn-migrator-output-{out_ver}:latest"

    shutil.rmtree(base_input_dir,  ignore_errors=True)
    shutil.rmtree(base_output_dir, ignore_errors=True)
    os.makedirs(base_input_dir,  exist_ok=True, mode=0o777)
    os.makedirs(base_output_dir, exist_ok=True, mode=0o777)

    # 1. Run INPUT container
    try:
        result_input = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{base_input_dir}:/input",
            name_input_image,
            f"input_{name_model}.py"
        ], check=True, capture_output=True, text=True)
        print(result_input.stdout)
        print(result_input.stderr)
        print("INPUT execution: ✅")
    except subprocess.CalledProcessError as e:
        print("INPUT execution: ❌")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

    # 2. Copy serialized_model
    try:
        src = os.path.join(base_input_dir,  "serialized_model.json")
        dst = os.path.join(base_output_dir, "serialized_model.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    except Exception as e:
        print(f"COPY serialized_model: ❌\nERROR: {e}")
        return None

    # 3. Run OUTPUT container
    try:
        result_output = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{base_output_dir}:/output",
            name_output_image,
            f"output_{name_model}.py"
        ], check=True, capture_output=True, text=True)
        print(result_output.stdout)
        print(result_output.stderr)
        print("OUTPUT execution: ✅")
    except subprocess.CalledProcessError as e:
        print("OUTPUT execution: ❌")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

    # 4. Comparison
    try:
        df_input  = pd.read_csv(f"{base_input_dir}/y_pred_input.csv")
        df_output = pd.read_csv(f"{base_output_dir}/y_pred_output.csv")
        return float(((df_input - df_output)["0"]).max())
    except Exception as e:
        print(f"COMPARISON: ❌\nERROR: {e}")
        return None

# -----------------------------------------------------------------------
# container_execution_with_retry
# -----------------------------------------------------------------------
def container_execution_with_retry(inp_ver, out_ver, max_retries=3, wait_seconds=10):
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")
        error = container_execution(inp_ver, out_ver)
        time.sleep(1)
        if error is not None:
            return error
        if attempt < max_retries:
            print(f"Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    print(f"All {max_retries} attempts failed ❌")
    return None

# -----------------------------------------------------------------------
# run_all
# -----------------------------------------------------------------------
def run_all():
    os.makedirs(base_input_dir,  exist_ok=True, mode=0o777)
    os.makedirs(base_output_dir, exist_ok=True, mode=0o777)

    df_performance = pd.DataFrame()

    for inp_ver in input_versions:
        all_output = []

        for out_ver in output_versions:
            buffer     = io.StringIO()
            sys.stdout = Tee(buffer)

            error = container_execution_with_retry(inp_ver, out_ver)

            if error is None:
                print(f"SKIPPING: input={inp_ver}, output={out_ver} due to error ❌")
                all_output.append(None)
                log_dir = f"logs/name_model={name_model}/error/input_version={inp_ver}/output_version={out_ver}"
            else:
                all_output.append(error)
                log_dir = f"logs/name_model={name_model}/successful/input_version={inp_ver}/output_version={out_ver}"

            print("-" * 70)
            sys.stdout = sys.stdout.stdout

            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/logs.txt", "w") as log_file:
                log_file.write(buffer.getvalue())

        df_performance[inp_ver] = all_output

    os.makedirs("performance", exist_ok=True)
    df_performance.index = output_versions
    df_performance.to_csv(f"performance/performance_{name_model}.csv")

    return df_performance


if __name__ == "__main__":
    df = run_all()
    print(df)