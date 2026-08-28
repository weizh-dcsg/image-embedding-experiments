#!/usr/bin/env python3

"""
This script deploys the model to an ES cluster.
✅ How to Run
python deploy_model.py --model-name intfloat/e5-small --env QA
--model-name: (required) Hugging Face model ID (e.g., intfloat/e5-small)

--env: (optional) One of: DEV, QA, PROD (default = DEV)
"""

import argparse
import os
from pathlib import Path

from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy HF transformer model to Elasticsearch")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model name (e.g., intfloat/e5-small)"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["DEV", "QA", "PROD"],
        default="DEV",
        help="Deployment environment: DEV (default), QA, or PROD"
    )
    return parser.parse_args()


def get_es_connection(env):
    env_var_mapping = {"DEV": "ELASTIC_HOST_DEV", "QA": "ELASTIC_HOST_QA", "PROD": "ELASTIC_HOST_PROD"}
    ES_HOST = os.getenv(env_var_mapping.get(env, f"ELASTIC_HOST_{env}"))
    ES_USER = os.getenv("ELASTIC_USERNAME")
    ES_PASS = os.getenv("ELASTIC_PASSWORD")
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER,ES_PASS),
        request_timeout=600,
        verify_certs=True  # Optional: Set to False only if using self-signed certs
    )

    # Test connection
    print(es.info().body)
    return es 

def is_model_registered(es, model_id):
    try:
        es.ml.get_trained_models(model_id=model_id)
        print(f"✅ Model '{model_id}' is registered in the cluster.")
        return True
    except Exception as e:
        if hasattr(e, 'info') and e.info.get('status') == 404:
            print(f"ℹ️ Model '{model_id}' is not registered.")
            return False
        else:
            print(f"⚠️ Error checking model registration: {e}")
            return False

def main():
    args = parse_args()
    hf_model_id = args.model_name
    env = args.env

    # 2. Define model name from Hugging Face`
    eland_model_id = hf_model_id.replace("/", "__")  # e.g., intfloat__e5-small
    print(f"Deploying model '{hf_model_id}' to {env} environment")

    # 1. Connect to Elasticsearch
    es = get_es_connection(env)
    
    if not is_model_registered(es,eland_model_id):

        # 3. Load a Hugging Face transformers model directly from the model hub
        tm = TransformerModel(model_id=hf_model_id, task_type="text_embedding")
        # 4. Export the model in a TorchScript representation which Elasticsearch uses
        tmp_path = "my_vs_models"
        full_path = os.path.abspath(tmp_path)
        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        print(f"Model will be saved to: {full_path}")

        model_path, config, vocab_path = tm.save(tmp_path)
        # 5. Register & import the model into Elasticsearch
        # This automatically registers the model if it doesn't already exist

        # Import the model to Elasticsearch
        ptm = PyTorchModel(es,tm.elasticsearch_model_id())
        ptm.import_model(
            model_path=model_path,
            config_path=None,
            vocab_path=vocab_path,
            config=config
        )
        # Start the trained model deployment
        es.ml.start_trained_model_deployment(
            model_id=tm.elasticsearch_model_id(),
            wait_for="started",   # optional: blocks until deployment is ready
            timeout="2m"   # wait up to 2 minutes
        )

        print(f"✅ Successfully deployed model: '{hf_model_id}' to {env} environment")

if __name__ == "__main__":
    main()