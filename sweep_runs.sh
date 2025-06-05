#!/bin/bash

# check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "Please activate your virtual environment first."
  exit 1
fi

for model in resnet18 resnet34 resnet50; do
  for skip_kind in learnable_orth identity; do
    for update_rule_iters in 1, 5, 10; do
      run_name="model-${model}_skip-${skip_kind}_ns-it${update_rule_iters}"
      python -m orthogonal_skip_connections.experiments.run_experiments \
        model.model_type=$model \
        model.skip_kind=$skip_kind \
        model.update_rule_iters=$update_rule_iters \
        wandb.run_name=$run_name
    done
  done
done