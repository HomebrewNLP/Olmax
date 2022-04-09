wandb sweep sweep.yaml 2> .tmp.txt
sweep=`cat .tmp.txt`
sweep=($sweep)
sweep="${sweep[-1]}"
python3 scripts/launch_on_tensorfork.py --sweep $sweep --us-service-account lucas-american-bucket@gpt-2-15b-poetry.iam.gserviceaccount.com --eu-service-account lucas-europe@gpt-2-15b-poetry.iam.gserviceaccount.com --prefix homebrewnlp --use-us 1 --dry 0 --cleanup 0