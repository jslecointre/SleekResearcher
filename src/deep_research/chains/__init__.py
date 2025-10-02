"""
Model Configuration for Deep Research
"""
import os

from deep_research import logger
from deep_research.utils import create_model

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "watsonx")

# Model configurations
if MODEL_PROVIDER == "watsonx":
    # WATSONX MODELS
    SCOPING_MODEL = "ibm:meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    RESEARCH_MODEL = "ibm:openai/gpt-oss-120b"
    SUMMARIZATION_MODEL = "ibm:meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    COMPRESS_MODEL = "ibm:meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    SUPERVISOR_MODEL = "ibm:openai/gpt-oss-120b"  # "ibm:meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    MAX_TOKEN_SUPERVISOR = 32000
    WRITER_MODEL = "ibm:openai/gpt-oss-120b"
    MAX_TOKEN_WRITER_MODEL = 32000

elif MODEL_PROVIDER == "openai_anthropic":
    # OPENAI/ANTHROPIC MODELS
    SCOPING_MODEL = "openai:gpt-4.1"
    RESEARCH_MODEL = "anthropic:claude-sonnet-4-20250514"  # Main research model
    SUMMARIZATION_MODEL = "openai:gpt-4.1-mini"  # Fast summarization
    COMPRESS_MODEL = "openai:gpt-4.1"  # High-quality compression
    SUPERVISOR_MODEL = "anthropic:claude-sonnet-4-20250514"
    WRITER_MODEL = "openai:gpt-4.1"  # max_tokens=32000
    MAX_TOKEN_WRITER_MODEL = 32000
    MAX_TOKEN_SUPERVISOR = None

else:
    raise ValueError(f"Invalid MODEL_PROVIDER: {MODEL_PROVIDER}. Must be 'watsonx' or 'openai_anthropic'")

try:
    logger.info("=" * 50)
    logger.info(f"USING {MODEL_PROVIDER.upper()} MODEL PROVIDER")
    logger.info("=" * 50)

    # SCOPING
    scoping_model = create_model(SCOPING_MODEL)
    logger.info(f"Scoping model created: {SCOPING_MODEL}")
    summarization_model = create_model(model_name=SUMMARIZATION_MODEL)
    logger.info(f"Summarization model created: {SUMMARIZATION_MODEL}")

    # RESEARCH
    research_model = create_model(RESEARCH_MODEL)
    logger.info(f"Main model created: {RESEARCH_MODEL}")

    compress_model = create_model(COMPRESS_MODEL, max_tokens=32000)
    logger.info(f"Compression model created: {COMPRESS_MODEL}")

    # SUPERVISOR MODEL
    supervisor_model = create_model(model_name=SUPERVISOR_MODEL, max_tokens=MAX_TOKEN_SUPERVISOR)
    logger.info(f"Supervisor model created: {SUPERVISOR_MODEL}")

    # WRITER MODEL
    writer_model = create_model(model_name=WRITER_MODEL, max_tokens=MAX_TOKEN_WRITER_MODEL)
    logger.info(f"Writer model created: {WRITER_MODEL}")

    logger.info("=" * 50)
    logger.info("All models initialized successfully!")
    logger.info("=" * 50)
except Exception as e:
    logger.info(f"Error creating models: {e}")
    logger.info("Using default models from research_agent.py...")
