# Running the platoon across two laptops

Laptop **A** (where you type `mpirun`) runs the road and cars 1 to 4. Laptop **B** runs cars 5 to 8.
Car 4 and car 5 then talk over the real Wi-Fi network: a truly distributed run on two machines.

MPI starts the processes on laptop B through SSH, so:
- the project must be at the **same path** on both laptops: `/Users/Shared/platoon`
  (that folder exists on every Mac, whatever the username), and
- A must be able to SSH into B **without a password**.

Plan about 30 minutes the first time. Do it before the demo day.

## 1. Both laptops on the same network

Use the same Wi-Fi. **Campus Wi-Fi often blocks laptops from reaching each other** ("client
isolation"). If step 4 fails, use a **phone hotspot** for both laptops, or a cable between them.

## 2. Install on laptop A

```bash
cd ~/projects/marl/deeprl_network
bash mpi/two_laptops/setup_laptop.sh          # copies the project to /Users/Shared/platoon
bash mpi/two_laptops/make_bundle.sh           # makes ~/Downloads/platoon_bundle.zip
```
AirDrop `platoon_bundle.zip` and `mpi/two_laptops/setup_laptop.sh` to laptop B.

## 3. Install on laptop B

1. Homebrew must be installed (https://brew.sh).
2. In Terminal: `bash ~/Downloads/setup_laptop.sh ~/Downloads/platoon_bundle.zip`
3. **System Settings > General > Sharing > Remote Login: ON** (this allows SSH).
4. Note the "Local hostname" shown there, e.g. `Anshikas-MacBook-Air.local`, and the username
   (`whoami` in Terminal).
5. If the firewall is on (System Settings > Network > Firewall), turn it off for the demo, or
   allow incoming connections for Python when macOS asks.

## 4. Passwordless SSH from A to B (on laptop A)

```bash
ssh-keygen -t ed25519                          # press Enter for all questions
ssh-copy-id USERNAME_B@HOSTNAME_B.local        # asks for B's password one last time
ssh USERNAME_B@HOSTNAME_B.local hostname       # must print B's name WITHOUT asking a password
```
If the usernames on A and B differ, add this to `~/.ssh/config` on A:
```
Host laptopB
    HostName HOSTNAME_B.local
    User USERNAME_B
```
and use `laptopB` as the name in the hosts file.

## 5. Hosts file (on laptop A)

```bash
cd /Users/Shared/platoon
cp mpi/two_laptops/hosts.example mpi/two_laptops/hosts
# edit mpi/two_laptops/hosts: line 1 = laptop A's name (slots=5), line 2 = laptop B (slots=4)
```

## 6. Run (on laptop A)

```bash
cd /Users/Shared/platoon
bash mpi/two_laptops/run_two_laptops.sh        # check: should print both machine names
bash mpi/two_laptops/run_two_laptops.sh --controller ia2c --log mpi/logs/two_laptops_ia2c.jsonl
bash mpi/two_laptops/run_two_laptops.sh --controller fixed --delay 5 --fallback radar \
                                        --log mpi/logs/two_laptops_fixed_delay_radar.jsonl
```
The output's `placement:` line shows which ranks ran on which laptop, e.g.
`ranks 0-4 on Divyams-MacBook-Air-2, ranks 5-8 on Anshikas-MacBook-Air`.
With no faults, the result must again be **−22.96** for IA2C: the same numbers, now over real Wi-Fi.

Watch it in the browser:
```bash
.venv/bin/python -m simulator.server --port 8766 --open --replay mpi/logs/two_laptops_ia2c.jsonl
```
The lane title says "2 machines".

## If something fails

| Problem | Fix |
|---|---|
| `ssh: connect ... timed out` / host not found | Not on the same network, or campus Wi-Fi isolation: use a phone hotspot. Check `ping HOSTNAME_B.local` |
| SSH asks for a password | Step 4 was not completed: run `ssh-copy-id` again |
| `prted: command not found` / `orted` not found | On laptop B, Homebrew is not on the SSH path: rerun the setup script there (it fixes `~/.zshenv`) |
| `No such file .../Users/Shared/platoon/...` | The project is not at `/Users/Shared/platoon` on both laptops |
| Run hangs at the start | Firewall on B is blocking MPI: turn it off for the demo |
| Many "stale reads" | Wi-Fi is slower than one machine: add `--timeout-ms 200` |
| Different results on B | Both laptops must use the same package versions (the setup script installs them) |

## What to tell the professor

- Same program, same results, now on **two physical machines** connected by Wi-Fi.
- Cars 4 and 5 exchange their messages over the real network, so real network latency is part of
  the run. The `ms per step` in the output shows how much the network costs compared with one laptop.
