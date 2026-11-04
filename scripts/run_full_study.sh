#!/bin/bash
# Runs the complete study on one Apple-silicon machine with Ollama. Safe to re-run: every stage resumes.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
source .venv/bin/activate

AGENT_MODELS="qwen3:8b qwen3:14b qwen3:32b llama3.1:8b gemma3:12b phi4:14b mistral-small3.2:24b gpt-oss:20b"
GENERATOR_MODELS="olmo2:13b granite3.3:8b"

if ! curl -s localhost:11434/api/tags > /dev/null; then
  OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > logs/ollama_server.log 2>&1 &
  sleep 5
fi

AVAILABLE=""
for m in $GENERATOR_MODELS $AGENT_MODELS; do
  if ollama pull "$m" > /dev/null 2>&1; then echo "pulled $m"; else echo "COULD NOT PULL $m"; fi
done
for m in $AGENT_MODELS; do
  if ollama list | awk '{print $1}' | grep -qx "$m"; then AVAILABLE="$AVAILABLE $m"; fi
done
GENS=""
for m in $GENERATOR_MODELS; do
  if ollama list | awk '{print $1}' | grep -qx "$m"; then GENS="$GENS $m"; fi
done
echo "agent models: $AVAILABLE"
echo "paraphrase generators: $GENS"

python benchmark_construction/build_lakehouse_request_benchmark.py > logs/benchmark_build.log
python benchmark_construction/generate_paraphrased_requests.py --generators $GENS
python agent_evaluation/run_keyword_router_baseline.py
python agent_evaluation/run_finetuned_encoder_baseline.py || echo "ENCODER BASELINE FAILED"
python agent_evaluation/run_local_model_agent_evaluation.py --models $AVAILABLE --concurrency 4
python analysis/analyze_agent_governance_results.py > logs/analysis.log
echo "STUDY_COMPLETE"
