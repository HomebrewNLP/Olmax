CLEANUP="${1}"
BASE_PREFIX="homebrew"

run () {
  zone=$1
  tpu_version=$2
  data_path=$3
  tpu_count=$4
  preemptible=$5
  prefix=$zone
  if [ "$preemptible" = "1" ]; then
    prefix="$prefix-preemptible"
  fi
  args="launch_multiple_runs.py --tpus $tpu_count --zone $zone --tpu-version $tpu_version --data-path $data_path --prefix $BASE_PREFIX-$prefix --preemptible $preemptible"
  if [ "$CLEANUP" == "1" ]; then
    python3 "$args" --cleanup 1
  else
    screen -dmS "$prefix" python3 "$args" --cleanup 0
  fi
}

# TPUv3 (Total: 330 v3-8's - 2640  v3-cores)
run europe-west4-a 3 "gs://ggpt4/the-big-char-pile/" 250 1
run europe-west4-b 3 "gs://ggpt4/the-big-char-pile/" 15 1
run europe-west4-c 3 "gs://ggpt4/the-big-char-pile/" 15 1
run us-central1-a 3 "gs://ggpt4-us/the-big-char-pile/" 200 1
run us-central1-c 3 "gs://ggpt4-us/the-big-char-pile/" 15 1
# non-preemptible; not gonna use euw4a as that zone is actively used
run europe-west4-b 3 "gs://ggpt4/the-big-char-pile/" 5 0
run europe-west4-c 3 "gs://ggpt4/the-big-char-pile/" 5 0
run us-central1-c 3 "gs://ggpt4-us/the-big-char-pile/" 5 0

# TPUv2 (Total: 470 v2-8's - 3760  v2-cores)
run europe-west4-b 2 "gs://ggpt4/the-big-char-pile/" 15 1
run europe-west4-c 2 "gs://ggpt4/the-big-char-pile/" 15 1
run us-central1-b 2 "gs://ggpt4-us/the-big-char-pile/" 150 1
run us-central1-c 2 "gs://ggpt4-us/the-big-char-pile/" 150 1
run us-central1-f 2 "gs://ggpt4-us/the-big-char-pile/" 100 1
# non-preemptible; not gonna use euw4a as that zone is actively used
run europe-west4-b 2 "gs://ggpt4/the-big-char-pile/" 5 0
run europe-west4-c 2 "gs://ggpt4/the-big-char-pile/" 5 0
run us-central1-f 2 "gs://ggpt4-us/the-big-char-pile/" 25 0
run us-central1-a 2 "gs://ggpt4-us/the-big-char-pile/" 5 0