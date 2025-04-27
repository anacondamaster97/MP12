from kubernetes import client, config, utils
from flask import Flask, request, jsonify # Use jsonify for Flask responses
from os import path
import yaml # Need PyYAML to load yaml files: pip install pyyaml
import random, string
import sys
import json # Keep json for potential loading if needed, but jsonify is better for responses

# --- Kubernetes API Setup ---
try:
    # Load configuration from default location (~/.kube/config)
    # This is expected to work if the server runs on the same machine (EC2 instance)
    # where 'eksctl' was used to create the cluster.
    config.load_kube_config()
    print("INFO: Kubernetes configuration loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Kubernetes configuration: {e}")
    print("       Ensure ~/.kube/config is present and valid, or run inside a cluster.")
    # Depending on requirements, you might exit or try in-cluster config
    # For this MP, running on the EC2 instance where eksctl ran is expected.
    # config.load_incluster_config() # Use this if running the server *inside* the K8s cluster
    sys.exit(1) # Exit if config loading fails

# Create API client instances
# CoreV1Api for pods, nodes, namespaces, services, etc.
v1 = client.CoreV1Api()
# BatchV1Api for jobs, cronjobs, etc.
batch_v1 = client.BatchV1Api()

# --- Flask App Setup ---
app = Flask(__name__)

# --- Helper Function to Create Kubernetes Job ---
def create_k8s_job(job_yaml_file, namespace):
    """
    Loads a Job definition from a YAML file and creates it in the specified namespace.

    Args:
        job_yaml_file (str): Path to the Kubernetes Job YAML file.
        namespace (str): The namespace where the job should be created.

    Returns:
        tuple: (bool, str) indicating success (True/False) and a message/error detail.
    """
    # Construct absolute path relative to this script file
    yaml_path = path.abspath(path.join(path.dirname(__file__), job_yaml_file))

    if not path.isfile(yaml_path):
        error_msg = f"Job YAML file not found at: {yaml_path}"
        print(f"ERROR: {error_msg}")
        return False, error_msg

    try:
        # Load the YAML file into a Python dictionary
        with open(yaml_path, 'r') as f:
            job_manifest = yaml.safe_load(f)

        if not job_manifest:
            error_msg = f"Job YAML file is empty or invalid: {yaml_path}"
            print(f"ERROR: {error_msg}")
            return False, error_msg

        # Use the Kubernetes client to create the job
        # Ensure the manifest KIND is Job and apiVersion is batch/v1
        print(f"INFO: Attempting to create job from '{job_yaml_file}' in namespace '{namespace}'...")
        api_response = batch_v1.create_namespaced_job(
            body=job_manifest,
            namespace=namespace
        )
        job_name = api_response.metadata.name
        print(f"INFO: Successfully initiated job creation. Job name: '{job_name}' in namespace '{namespace}'.")
        return True, job_name # Return success and the generated job name

    except client.ApiException as e:
        # Handle API errors (e.g., invalid manifest, permissions, quota exceeded *at creation*)
        error_body = e.body
        try:
            # Try to parse the error body as JSON for more details
            error_details = json.loads(error_body)
            error_msg = f"Kubernetes API Error: {error_details.get('message', error_body)}"
        except json.JSONDecodeError:
            error_msg = f"Kubernetes API Error: {error_body}"
        print(f"ERROR: {error_msg}")
        return False, error_msg

    except yaml.YAMLError as e:
        # Handle errors during YAML parsing
        error_msg = f"Error parsing YAML file '{yaml_path}': {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg

    except Exception as e:
        # Handle other unexpected errors
        error_msg = f"An unexpected error occurred: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


# --- API Endpoints ---

@app.route('/config', methods=['GET'])
def get_config():
    """
    Retrieves information about all pods running across all namespaces.
    """
    print("INFO: Received request for /config")
    pods_list = []
    try:
        # List pods in all namespaces
        # You might want to add a timeout_seconds parameter
        ret = v1.list_pod_for_all_namespaces(watch=False, timeout_seconds=10)

        for pod in ret.items:
            # Extract required information for each pod
            pod_info = {
                "name": pod.metadata.name,
                # Pod IP might not be assigned immediately
                "ip": pod.status.pod_ip if pod.status.pod_ip else "N/A",
                "namespace": pod.metadata.namespace,
                # Node name might not be assigned immediately (if unscheduled)
                "node": pod.spec.node_name if pod.spec.node_name else "Pending/Unscheduled",
                "status": pod.status.phase # e.g., Pending, Running, Succeeded, Failed, Unknown
            }
            pods_list.append(pod_info)

        print(f"INFO: Found {len(pods_list)} pods across all namespaces.")
        # Return the list of pods formatted as JSON
        return jsonify({"pods": pods_list}), 200

    except client.ApiException as e:
        error_msg = f"Kubernetes API Error listing pods: {e.body}"
        print(f"ERROR: {error_msg}")
        return jsonify({"error": "Failed to list pods", "details": error_msg}), 500
    except Exception as e:
        error_msg = f"An unexpected error occurred while listing pods: {e}"
        print(f"ERROR: {error_msg}")
        return jsonify({"error": "An internal server error occurred", "details": error_msg}), 500


@app.route('/img-classification/free',methods=['POST'])
def post_free():
    """
    Handles requests for the free image classification service.
    Creates a Kubernetes Job in the 'free-service' namespace using 'free-job.yaml'.
    """
    print("INFO: Received request for /img-classification/free")

    # Request body parsing is not needed as per instructions (dataset='mnist', type='ff' are fixed in YAML)
    # Example if you needed it:
    # if not request.is_json:
    #     return jsonify({"error": "Request must be JSON"}), 400
    # data = request.get_json()
    # dataset = data.get('dataset') # But we hardcode this in the YAML

    # Attempt to create the job
    success, result = create_k8s_job('free-job.yaml', 'free-service')

    if success:
        # Return 200 OK if the job creation request was accepted by the K8s API
        # 'result' contains the generated job name
        return jsonify({"message": "Free tier job creation request accepted.", "job_name": result}), 200
    else:
        # Return 500 Internal Server Error if the API call failed
        # 'result' contains the error message
        return jsonify({"error": "Failed to create free tier job.", "details": result}), 500


@app.route('/img-classification/premium', methods=['POST'])
def post_premium():
    """
    Handles requests for the premium image classification service.
    Creates a Kubernetes Job in the 'default' namespace using 'premium-job.yaml'.
    """
    print("INFO: Received request for /img-classification/premium")

    # Request body parsing not needed (dataset='kmnist', type='cnn' are fixed in YAML)

    # Attempt to create the job in the 'default' namespace
    success, result = create_k8s_job('premium-job.yaml', 'default')

    if success:
        # Return 200 OK if the job creation request was accepted
        return jsonify({"message": "Premium tier job creation request accepted.", "job_name": result}), 200
    else:
        # Return 500 Internal Server Error if the API call failed
        return jsonify({"error": "Failed to create premium tier job.", "details": result}), 500


# --- Main Execution ---
if __name__ == "__main__":
    print("INFO: Starting Flask server...")
    # Run Flask development server
    # Listen on all available network interfaces (0.0.0.0) on port 5000
    # Set debug=False for production/testing as per instructions (debug=True reloads on code change)
    app.run(host='0.0.0.0', port=5000, debug=False)
    # Note: To run on port 80, you might need root privileges (sudo python server.py)
    # and ensure no other service (like Apache/Nginx) is using port 80.
    # Port 5000 is generally easier for non-root execution.