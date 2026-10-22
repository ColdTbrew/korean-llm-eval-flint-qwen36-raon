#!/bin/bash
# sequential: one model server at a time; runs 2 and 3 for each model
# NOTE: in this repo the server scripts are named serve_flint.sh / serve_qwen_community.sh / serve_qwen_official.sh / serve_raon.sh;
# on the original machine serve_llama.sh = serve_flint.sh and serve_qwen.sh = serve_qwen_community.sh.
umask 077
cd ~/flint
log=~/flint/runs/driver.log; mkdir -p ~/flint/runs; chmod 700 ~/flint/runs
stop_all() { for s in flint-server qwen-server qwen-off-server raon-server; do tmux send-keys -t $s C-c 2>/dev/null; done; sleep 6; for s in flint-server qwen-server qwen-off-server raon-server; do tmux kill-session -t $s 2>/dev/null; done; sleep 3; }
run_model() { # name session script port alias extra
  name=$1; sess=$2; script=$3; port=$4; alias=$5; extra=$6
  stop_all
  echo "$(date '+%F %T') start $name avail=$(free -g | awk '/Mem/{print $7}')G" >> $log
  tmux new -d -s $sess "bash -c \"umask 077; bash ~/flint/$script 2>&1 | tee ~/flint/runs/${name}_server.log\""
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/health | grep -q ok && break; sleep 4; done
  echo "$(date '+%F %T') $name healthy avail=$(free -g | awk '/Mem/{print $7}')G" >> $log
  for r in 2 3; do
    d=~/flint/runs/$name/run$r; mkdir -p $d; cd $d; cp ~/flint/qwen36off/{run_eval.py,bench.py,long_ko.txt} .
    ex=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);d['seed']=$r;print(json.dumps(d))" "$extra")
    export FLINT_URL=http://127.0.0.1:$port/v1/chat/completions EVAL_MODEL=$alias EVAL_EXTRA="$ex"
    echo "$ex" > eval_extra.json
    python3 bench.py > bench.log 2>&1
    python3 run_eval.py ~/flint/qwen36off/prompts.json results.jsonl > eval.log 2>&1
    echo "$(date '+%F %T') $name run$r done n=$(wc -l < results.jsonl) avail=$(free -g | awk '/Mem/{print $7}')G" >> $log
    cd ~/flint
  done
}
run_model flint flint-server serve_llama.sh 8080 flint '{}'
run_model qwen_community qwen-server serve_qwen.sh 8081 qwen3.6-35b-a3b '{"presence_penalty":1.5,"min_p":0.0}'
run_model qwen_official qwen-off-server serve_qwen_official.sh 8082 qwen3.6-35b-a3b-official '{"presence_penalty":1.5,"min_p":0.0}'
run_model raon raon-server serve_raon.sh 8083 raon-1-mini-preview '{"presence_penalty":1.5,"min_p":0.0}'
stop_all
tmux new -d -s flint-server "bash -c \"umask 077; bash ~/flint/serve_llama.sh 2>&1 | tee ~/flint/server.log\""
for i in $(seq 1 60); do curl -s 127.0.0.1:8080/health | grep -q ok && break; sleep 4; done
echo "$(date '+%F %T') ALLDONE flint restored" >> $log
