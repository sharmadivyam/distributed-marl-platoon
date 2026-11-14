# Distributed Platoon Simulator (Phase 1)

A live, visual simulation of the CACC platoon as a distributed system. Three platoons drive
side by side, one per controller: the fixed controller, IA2C and NeurComm. Every car is an
independent agent. It measures its own speed and its radar gap to the car in front, and it gets
everything else from other vehicles over vehicle-to-vehicle (V2V) messages, which are drawn as
moving dots.

Everything runs on your laptop. The server only listens on `127.0.0.1` and uses the Python
standard library, so nothing extra has to be installed.

## Run it

```bash
cd ~/projects/marl/deeprl_network
source .venv/bin/activate
python -m simulator.server --open      # opens http://localhost:8765
```

Stop it with `Ctrl+C` in that terminal. If port 8765 is busy, add `--port 8770`.

## Controls

| Control | What it does |
|---|---|
| Play / Pause (`Space`) | Start or pause the simulation |
| Reset (`R`) | Restart all three platoons from the same starting position |
| Scenario | **Catch-up** (car 1 starts too far back; the policies were trained on this) or **Slow-down** (all cars start fast and the leader slows down; the policies never saw this) |
| Seed | Starting position. These are the report's unseen test seeds, so seed 100000 is the episode shown in the report's trajectory figure |
| Speed | 0.5x to 4x real time |
| Leader brakes! (`B`) | The leader brakes from cruising speed to 8 m/s at 2 m/s², holds for 4 s, then speeds back up. None of the controllers trained on this |
| Auto demo | Scripted run for demos and recordings: catch-up, then the leader brakes at 30 s |

## What you see

- **Cars:** coloured by their gap to the car in front. Green is within 2 m of the 20 m target,
  amber is 2 to 6 m off, red is more than 6 m off or closer than 10 m. The number under each
  gap is the gap in metres. `L` is the leader.
- **Messages:** blue dots are beacons (the front car's speed, 4 bytes). Orange dots are the
  extra data the learned controllers need: IA2C shares each car's 5-number observation
  (20 bytes), NeurComm sends observation + policy + LSTM memory (292 bytes, drawn bigger).
  Messages to the car behind travel on the upper line, messages to the car in front on the lower.
- **Chips above each platoon:** collision status, average reward per step (the report's metric),
  the closest gap so far, and the V2V traffic in messages and kilobytes per second.
- **Chart on the right:** the gap of every car over the last 30 s (car 1 light, car 8 dark).
  The dashed line is the 20 m target and the pink band is closer than 10 m.

## Is it faithful to the trained models?

Yes. `python -m simulator.check` replays the report's evaluation through the simulator. With the
perfect network, all three platoons match the original environment step for step (same rewards,
same gaps), for example IA2C scores exactly −22.96 on seed 100000, like in the evaluation.

## What happens (seed 100000)

| | Catch-up | Catch-up + leader brakes at 30 s | Slow-down (never seen) |
|---|---|---|---|
| Fixed controller | no collision, a wave grows down the platoon | no collision, closest gap 11.5 m | no collision |
| IA2C | no collision, calm platoon | no collision, closest gap 6.2 m | **collision at 26.7 s** |
| NeurComm | no collision, but car 2 never closes its gap | **collision at 37.3 s** | **collision at 8.7 s** |

V2V traffic per 0.1 s step: fixed 8 messages / 32 bytes, IA2C 22 messages / 312 bytes,
NeurComm 22 messages / 4,120 bytes (13 times IA2C).

## Suggested demo for the professor (about 4 minutes)

1. **Before pressing Play:** each coloured box is an independent agent; the dots are the V2V
   messages it depends on. Point at the traffic chips: NeurComm sends 13 times more data than IA2C.
2. **Auto demo, first 30 s (catch-up, trained scenario):** car 1 has to close a big gap. The
   fixed controller's disturbance grows down the platoon (cars at the back turn amber and red);
   IA2C keeps the platoon calm; NeurComm's car 2 never reacts.
3. **At 30 s the leader brakes (never trained on):** fixed and IA2C absorb it, NeurComm crashes.
4. **Switch the scenario to Slow-down:** now IA2C crashes too, while the fixed controller copes.
   The learned policies work in the situation they trained on and fail outside it.
5. **What comes next:** Phase 2 breaks the network on purpose (lost and delayed messages, cut
   links, crashed cars) with sliders on this same screen. Phase 3 runs every car as its own
   process.

A recorded run of the auto demo is in `demo/videos/` as a backup in case a live demo fails.

## How it is built

| File | Role |
|---|---|
| `network.py` | The V2V network layer. Every message goes through it, so it can be counted, drawn, and (in Phase 2) dropped or delayed |
| `world.py` | The CACC physics, but each car only uses its own measurements plus received messages. Also the leader and one platoon ("lane") |
| `controllers.py` | Fixed controller, and the trained IA2C and NeurComm policies from `demo/runs` |
| `server.py` | Runs the simulation in real time and streams it to the browser (Server-Sent Events) |
| `static/` | The dashboard (HTML, CSS, JavaScript) |
| `check.py` | Equivalence check and the brake test |

Notes for questions:
- The physics (positions and speeds) runs in one "world" object. It stands for the real road,
  not a central controller: each car's decision only uses its own sensors and its messages.
- NeurComm's message contents are still computed inside one shared policy network; the bytes
  shown are what each car would send. Phase 3 splits it so each car runs its own part.
