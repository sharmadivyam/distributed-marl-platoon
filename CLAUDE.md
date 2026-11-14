# CLAUDE.md

PyTorch implementation of networked multi-agent A2C (NMARL) from Chu, Chinchali & Katti,
"Multi-agent Reinforcement Learning for Networked System Control" (ICLR 2020). Each agent
can only observe and communicate with its graph neighbours. Environments: ATSC (SUMO
traffic signals, large grid / Monaco) and CACC (platoon catch-up / slow-down, pure numpy).

The current branch `neurcomm-repro` is for reproducing the NeurComm (`ma2c_nc`) results.

## Commands

```bash
# train (writes <base>/log, <base>/data, <base>/model)
python3 main.py --base-dir <base> train --config-dir config/config_ma2c_nc_catchup.ini
# evaluate: <base> is a *single trained agent dir*, not a parent of several
python3 main.py --base-dir <base> evaluate --evaluation-seeds 2000,2010   # --checkpoint best for model/best/
# SUMO GUI (ATSC only, writes no data)
python3 main.py --base-dir <base> evaluate --evaluation-seeds 2000 --demo
tensorboard --logdir <base>/log
# demo pipeline (parallel train+eval, report): see demo/DEMO_GUIDE.md
python3 demo/run_demo.py --steps 6e4 --algos ma2c_nc,ia2c --scenarios catchup
python3 demo/make_report.py
```

There are no tests. ATSC grid needs `envs/large_grid_data/build_file.py` run first, plus SUMO.

## Entry flow

### Train: `main.py:train` (main.py:80)
1. `init_dir` creates `log/ data/ model/` under `--base-dir` (utils.py:28), and the config
   is copied into `data/` with `shutil.copy` (utils.py:16-17).
2. `init_env(config['ENV_CONFIG'])` (main.py:42) picks the env from `scenario`:
   `atsc*large_grid` → `LargeGridEnv`, other `atsc*` → `RealNetEnv`, **anything else** →
   `CACCEnv`. The ATSC env modules (which need SUMO's `traci`/`sumolib`) are imported lazily
   inside `init_env`, so CACC runs without SUMO.
3. `Counter(total_step, test_interval, log_interval)` (main.py:94-97).
4. `init_agent` (main.py:56) maps `env.agent` (the `ENV_CONFIG.agent` string) to a model
   class and passes `env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask,
   env.coop_gamma`. An unknown string, including `greedy`, returns `None`.
5. `model.load(dirs['model'], train_mode=True)` resumes from the highest
   `checkpoint-<step>.pt` if one exists (models.py:62). If none exists, it logs
   "Can not find checkpoint", which is harmless.
6. `Trainer.run()` (utils.py:230), per episode:
   - `env.reset()`, `done=True`, `model.reset()` (this resets the LSTM states).
   - Repeat `explore()` (utils.py:168) for `n_step = MODEL_CONFIG.batch_size` steps:
     `_get_policy` → `_get_value` → `env.update_fingerprint(policy)` → `env.step` →
     `model.add_transition`. The bootstrap value `R` comes from the critic, or is 0 on
     `done`.
   - Then `model.backward(R, dt, ...)` (one RMSprop update per rollout).
   - Agents whose name starts with `ma2c` get `env.get_fingerprint()` (the previous step's
     policies) as extra input (utils.py:119-121). `ia2c*` agents get neighbour actions for
     the critic instead (utils.py:136).
   - **CACC only:** after every training episode, a deterministic (argmax) test episode is
     run, and *its* reward is what gets logged as `train_reward` (utils.py:253-260). ATSC
     logs the training-episode reward and tests stochastically (utils.py:214-219).
   - **Best checkpoint (CACC):** `Trainer` gets `model_dir` from `main.py` and, whenever the
     mean of the last 5 test rewards is a new high, replaces `model/best/checkpoint-<step>.pt`
     (`_save_if_best`, utils.py:152-166).
   - At the end, the run writes `data/train_reward.csv`, and `main.py` saves
     `model/checkpoint-<final_step>.pt`.

### Evaluate: `main.py:evaluate` (main.py:144)
- Creates `eva_data/ eva_log/` (skipped with `--demo`). `evaluate_fn(base_dir, ...)` reads
  the first `*.ini` in `<base>/data/` (utils.py:20), builds the env with SUMO `port=1`, and
  overrides the env's `test_seeds` with `--evaluation-seeds` (main.py:130).
- Builds the model with `total_step=0`, so there is no optimizer or buffer
  (models.py:142). It loads the latest checkpoint in `model/` in eval mode, or in `model/best/` with
  `--checkpoint best` (main.py:136).
- `Evaluator.run()` (utils.py:326) calls `perform(test_ind)` once per seed. Output goes to
  `eva_data/<scenario>_<agent>_{control,traffic}.csv` (cacc_env.py:131-137 for CACC).

`Tester`, `init_test_flag`, `threading`, and `TRAIN_CONFIG.test_interval` are imported or
parsed but never used by the train or evaluate paths.

## Algorithms

Model classes are in `agents/models.py`. Each one only overrides `_init_policy` (and, for
the MA2C family, the buffer and forward signature). Policy networks are in
`agents/policies.py`. All of them use FC encoder → LSTMCell → per-agent actor head (softmax)
plus a critic head that also takes one-hot neighbour actions.

| `agent =` (config) | Model class | Policy class | Paper | Notes |
|---|---|---|---|---|
| `ia2c` | `IA2C` models.py:17 | `LstmPolicy` policies.py:78 | Baseline independent A2C. README's "PolicyInferring" (Lowe et al. 2017, MADDPG) is the closest match: the critic is conditioned on neighbour actions (policies.py:30-57) | One `nn.ModuleList` of per-agent policies, one optimizer. On CACC, the state includes neighbour states (cacc_env.py:71-73) |
| `ia2c_fp` | `IA2C_FP` models.py:208 | `FPPolicy` policies.py:138 | Fingerprints, Foerster et al. 2017 (arXiv:1702.08887) | The env appends neighbour policies to the **end** of the state (cacc_env.py:74-77). `FPPolicy` splits them into a separate FC (policies.py:144-166) |
| `ma2c_nc` | `MA2C_NC` models.py:238 | `NCMultiAgentPolicy` policies.py:169 | **NeurComm**, Chu et al. ICLR 2020 (message passing inspired by Gilmer et al. 2017 MPNN) | One centralized meta-module holds all agents. Heterogeneous dims are zero-padded (models.py:276-282). Uses `MultiAgentOnPolicyBuffer` |
| `ma2c_cu` | `IA2C_CU` models.py:302 | `ConsensusPolicy` policies.py:380 | Consensus update, Zhang et al. 2018 (arXiv:1802.08757) | **Naming trap:** the class is `IA2C_*`, but the agent string starts with `ma2c`, so the Trainer and env treat it as MA2C (fingerprint path, own-state-only obs, cacc_env.py:279). After each backward, it averages **LSTM** params with neighbours (policies.py:394-400, 424-435). The helper is called `_get_critic_wts`, but the LSTM is shared by actor and critic. Config files are named `config_ia2c_cu_*.ini` |
| `ma2c_cnet` | `MA2C_CNET` models.py:323 | `CommNetMultiAgentPolicy` policies.py:450 | CommNet, Sukhbaatar et al. 2016 (ref impl: IC3Net) | `self.name = 'ma2c_ic3'` (models.py:326). The message is the **mean** of neighbour `h`, **added** to the obs encoding without ReLU (policies.py:481-495). Ignores fingerprints |
| `ma2c_dial` | `MA2C_DIAL` models.py:340 | `DIALMultiAgentPolicy` policies.py:498 | DIAL, Foerster et al. 2016 | Message is the concatenated neighbour `h` through an FC, plus a one-hot of the agent's own argmax fingerprint, padded to `n_fc` (policies.py:526-539) |
| `greedy` | — | — | — | Only `config/config_greedy.ini` uses it, and it is stale: `init_agent` returns `None`, and `scenario = large_grid` lacks the `atsc_` prefix, so it would route to `CACCEnv` |

Shared mechanics in `Policy` (policies.py:8): the loss is `-logπ·A + v_coef·(R-v)² −
e_coef·H` (policies.py:59-64). The MA2C family sums this over agents (policies.py:202-210).
LSTM state is carried across rollouts but detached at rollout boundaries
(`states_fw`/`states_bw`). It is zeroed whenever the pre-step `done` is set (utils.py:38-39,
policies.py:350).

## NeurComm message construction

All of it is in `NCMultiAgentPolicy`, agents/policies.py:

- **Message per agent `i`: `_get_comm_s`, policies.py:229-250**
  - `m_i`: neighbours' previous-step LSTM hidden states `h_j`, **concatenated**
    (policies.py:231).
  - `p_i`: neighbours' fingerprints, i.e. their previous-step policies π_j
    (policies.py:232).
  - `nx_i`: neighbours' current observations s_j (policies.py:233).
  - For heterogeneous nets, each neighbour slice is narrowed back to its true dim
    (policies.py:238-246).
  - Encoding: `relu(fc_x([x_i, nx_i]))`, `relu(fc_p(p_i))`, and `relu(fc_m(m_i))` are
    concatenated into a 3·`n_fc` LSTM input (policies.py:247-250; layer sizes in
    `_init_comm_layer`, policies.py:270-288).
- **One communication round per time step: `_run_comm_layers`, policies.py:329-357.** It
  calls `_get_comm_s` at policies.py:343 using `h` from t-1. Agents with no neighbours use
  only `fc_x(x_i)` (policies.py:344-349).
- **Critic:** `_run_critic_heads` concatenates `h_i` with one-hot neighbour actions
  (policies.py:359-377).
- **Where the fingerprints come from:** `Trainer._get_policy` calls `env.get_fingerprint()`
  (utils.py:120), and the env stores the last policy via `env.update_fingerprint(policy)`
  (utils.py:177; cacc_env.py:244-248, atsc_env.py:126/212). CACC initializes them uniform
  on reset (cacc_env.py:186). The same `ps` are stored in the buffer (utils.py:184,
  models.py:245-256) and replayed in `backward`.
- `CommNetMultiAgentPolicy` and `DIALMultiAgentPolicy` override only `_init_comm_layer` and
  `_get_comm_s`, so they share this loop.

## Spatial discount factor (α, `coop_gamma`)

- **Config:** `ENV_CONFIG.coop_gamma`. It is read by the env (cacc_env.py:340,
  atsc_env.py:90), exposed as `env.coop_gamma`, and passed by `init_agent` (main.py:58-75)
  → `_init_algo` → `_init_train` → `_init_trans_buffer` as the buffer's `alpha`
  (models.py:199 for IA2C-style per-agent `OnPolicyBuffer`, models.py:299 for the MA2C
  `MultiAgentOnPolicyBuffer`).
- **Distance matrix:** `env.distance_mask` holds hop distances, `|i-j|` for CACC
  (cacc_env.py:255-259). For ATSC, see large_grid_env.py:105 and real_net_env.py:155-166.
- **Discounted return (the code that actually runs):**
  - `OnPolicyBuffer._add_s_R_Adv`, agents/utils.py:157-173.
  - `MultiAgentOnPolicyBuffer._add_s_R_Adv`, agents/utils.py:245-269.
  - Per step: `R = γ·R·(1-done) + Σ_d α^d · Σ_{j: dist(i,j)=d} r_j`
    (utils.py:162-166 / 256-260).
- **`coop_gamma < 0` disables spatial discounting.** The env then returns the scalar global
  reward to every agent (cacc_env.py:236-237; atsc_env.py:205-206, train mode only), and
  the buffer uses plain `_add_R_Adv` (utils.py:106-107, 120-132 / 181-182, 194-212).
  **Every shipped `ma2c_*` config uses `coop_gamma = -1`** except
  `config_ma2c_{cnet,dial}_net.ini` (0.9). The IA2C configs use 0.8–0.95 on most
  scenarios.
- **Dead code:** `_add_st_R_Adv` (utils.py:134, 214) is a spatio-temporal variant with a
  `(γα)^d` weight that uses `dt`. `sample_transition` never calls it, so the `dt` passed
  into `backward` is unused.

## Config → code map

`[MODEL_CONFIG]` is read in `IA2C._init_algo` / `_init_train` / `_init_scheduler`
(models.py:104-199):

| key | used as | where |
|---|---|---|
| `batch_size` | `n_step`: rollout length per update. `env.T % n_step == 0` is asserted | models.py:122, utils.py:107 |
| `num_fc` / `num_lstm` | FC width / LSTM hidden size (`n_h` in MA2C policies) | models.py:123-124 |
| `reward_norm` / `reward_clip` | reward divided by `reward_norm` (if >0), then clipped to ±`reward_clip` (if >0) | models.py:29-32, 241-244 |
| `gamma` | temporal discount in the buffer | models.py:192 |
| `lr_init`, `lr_decay`, `lr_min` | RMSprop LR. `linear` decay needs `lr_min` (all configs use `constant`) | models.py:166-172, agents/utils.py:299 |
| `rmsp_alpha`, `rmsp_epsilon` | RMSprop `alpha`, `eps` | models.py:182-191 |
| `optimizer` (optional) | `rmsprop` (default, `torch.optim.RMSprop`) or `rmsprop_tf` (`TFRMSprop`, agents/utils.py:274: TF1 semantics, ms init 1 and eps inside sqrt) | models.py:184-191 |
| `entropy_coef`, `value_coef` | loss weights | models.py:178-179, policies.py:59 |
| `max_grad_norm` | global grad clip (if >0) | models.py:47-48 |

`[TRAIN_CONFIG]` (main.py:94-96): `total_step` is the stop condition (utils.py:86).
`log_interval` sets how often step logging happens (utils.py:83, 188). `test_interval` is
unused.

`[ENV_CONFIG]`:
- **Common:**
  - `agent` selects the algorithm (main.py:56) and changes what the env puts in the state
    (cacc_env.py:71-77, 279).
  - `scenario` selects the env (main.py:43). For CACC, `split('_')[1]` gives
    `catchup`/`slowdown` (cacc_env.py:334).
  - `coop_gamma` is described in the spatial-discount section above.
  - `seed` seeds torch (main.py:100, models.py:133) and the env's numpy (cacc_env.py:22).
    It increments every reset (cacc_env.py:176).
  - `test_seeds` is overridden at evaluation time.
- **CACC (cacc_env.py:320-343):**
  - `control_interval_sec` → `dt`.
  - `episode_length_sec / dt` → `T`.
  - `batch_size`: after a collision, the episode ends at the next multiple of it
    (cacc_env.py:232). Keep it equal to `MODEL_CONFIG.batch_size`.
  - `headway_*`, `speed_*`, `accel_*` → OVM and constraints.
  - `reward_v`, `reward_u`, `collision_penalty` → reward weights a, b, G
    (cacc_env.py:40-52). Training also adds a collision-headway shaping term.
  - `n_vehicle` → number of agents.
  - `norm_headway`, `norm_speed` are read but unused.
- **ATSC (atsc_env.py:79-98):** `control_interval_sec`, `yellow_interval_sec`,
  `episode_length_sec`, `objective`, `data_path`, `norm_*`, `clip_*`, `coef_wait`,
  `peak_flow*`, `init_density`.

## Gotchas

- `main.py:19-20` defaults to the original author's local paths. Always pass `--base-dir`
  and `--config-dir`.
- **Training is unstable on CACC** with the paper's settings (single seed): rewards swing a lot
  and NeurComm tends to settle in a near-"do nothing" policy within 60k–250k steps. Evaluate
  `--checkpoint best`, not the last checkpoint. Details: demo/DEMO_GUIDE.md ("What we found").
- CACC seeds: `env.seed` grows by 2 per episode (train + test reset), reaching ~2000 after
  250k steps, so use evaluation seeds far above that (the demo uses 100000+).
- **Possible heterogeneous-dim bug in DIAL:** `DIALMultiAgentPolicy._get_comm_s` feeds the
  padded `x[i]` (policies.py:538) instead of narrowing it to `n_s_ls[i]` like NC and
  CommNet do (policies.py:246, 493). On heterogeneous nets (e.g. Monaco), this mismatches
  `fc_x_layers[i]`'s input size unless agent `i` has the max state dim. Not yet verified at
  runtime.
- The working tree has `total_step = 1e4` (instead of 1e6) in
  `config/config_ma2c_nc_catchup.ini`. That is a local smoke-test edit, not the paper
  setting.
