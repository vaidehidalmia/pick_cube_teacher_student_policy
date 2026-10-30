cd "$(dirname "$0")/.."
mkdir -p logs

CONDITIONS="demos100 demos25 demos50 demos200 demos500"
EVAL_SEEDS="1000 2000 3000"

for cond in $CONDITIONS; do
    cfg="configs/student_${cond}.yaml"

    if [ -f "checkpoints/student_${cond}/policy.pt" ]; then
        echo "[$(date +%H:%M)] skip training ${cond} (checkpoint exists)"
    else
        echo "[$(date +%H:%M)] train ${cond}"
        if ! uv run python code/train_student.py "$cfg" \
            > "logs/train_${cond}.log" 2>&1; then
            echo "  FAILED, see logs/train_${cond}.log"
            continue
        fi
        grep "training for\|using .* episodes" "logs/train_${cond}.log"
    fi

    for s in $EVAL_SEEDS; do
        result="results/student_${cond}_es${s}_eval.json"
        log="logs/eval_${cond}_es${s}.log"

        if [ -f "$result" ]; then
            echo "[$(date +%H:%M)] skip eval ${cond} eval seed ${s}"
            continue
        fi

        echo "[$(date +%H:%M)] eval ${cond} eval seed ${s}"

        if uv run python code/eval_student.py "$cfg" \
            --eval-seed "$s" \
            > "$log" 2>&1; then
            grep "^success_rate" "$log"
        else
            echo "  FAILED, see $log"
        fi
    done
done