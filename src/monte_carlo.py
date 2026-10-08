"""First-visit Monte Carlo control: MCTable, the agent, train/evaluate helpers, the 3 training phases and the evaluation run.

Copied cell by cell, without changes, from the Zeppelin note Nhom08_MultiFloorMaze_Code.json so the code can be read on GitHub.
Some helpers call Zeppelin's z object for the game UI, so run the code in the notebook (or the Colab notebook for Double DQN).
"""

# ===== Cell 21: 7.1. Định nghĩa MCTable =====
class MCTable(dict):
    # Lưu Q(s,a) dưới dạng trung bình mẫu Monte Carlo
    # dict bên trong giữ SUM các return G đã cộng dồn
    # nCounts[k] = số mẫu đã thêm
    # __getitem__(k) -> SUM / count (ước lượng Q). Chưa có key -> default
    # __setitem__(k, v): không ghi đè Q; mà cộng thêm một return v và tăng count

    def __init__(self, default):
        dict.__init__(self)
        self.nCounts = {}
        self.default = default

    def __getitem__(self, k):
        if super().__contains__(k):
            return super().__getitem__(k) / self.nCounts[k]
        return self.default

    def __setitem__(self, k, v):
        if super().__contains__(k):
            oldV = super().__getitem__(k)
        else:
            oldV = 0
            self.nCounts[k] = 0
        super().__setitem__(k, oldV + v)
        self.nCounts[k] += 1


# ===== Cell 22: 7.2. Định nghĩa MonteCarloControl =====
# MC Control + epsilon-greedy
class MonteCarloControl:
    # Sau mỗi episode kết thúc: tính return G đã chiết khấu γ, first-visit cập nhật MCTable
    def __init__(self, env, discount, epsilon, epsilon_min, decay_rate, defaultQ, maxSteps=None, seed=None):
        self.env = env
        self.delta = float(discount)  # γ — hệ số chiết khấu cho return
        self.epsilon = float(epsilon)  # xác suất explore (epsilon-greedy)
        self.epsilon_min = float(epsilon_min)  # sàn epsilon sau decay
        self.decay_rate = float(decay_rate)  # nhân epsilon mỗi episode (exponential decay)
        self.table = MCTable(defaultQ)  # Q(s,a) ~ trung bình return
        self.seed = seed

        # Map cố định: lần đầu random layout 1 lần, các ep sau chỉ reset trạng thái
        self.fixedMapRandom1Time = False
        self.agent = Agent()

        if maxSteps is not None:
            self.env.maxSteps = maxSteps

        if self.seed is not None:
            self.randGen = np.random.default_rng(seed=seed)
        else:
            self.randGen = np.random.default_rng()

    def decay_epsilon(self):
        # Sau mỗi episode: giảm epsilon nhưng không thấp hơn epsilon_min
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def getAgentState(self):
        return self.env.getState(self.agent)

    def play(self, n=1, fixed_map=False):
        for i in range(n):
            if fixed_map:
                if not self.fixedMapRandom1Time:
                    if self.seed is not None:
                        env_seed = int(self.randGen.integers(1_000_000_000))
                        self.env.reset(seed=env_seed, randomize_map=True)
                    else:
                        self.env.reset(seed=None, randomize_map=True)
                    self.fixedMapRandom1Time = True
                else:
                    self.env.reset(seed=None, randomize_map=False)
            elif self.seed is not None:
                env_seed = int(self.randGen.integers(1_000_000_000))
                self.env.reset(seed=env_seed, randomize_map=True)
            else:
                self.env.reset(seed=None, randomize_map=True)

            trajectory = []
            done = False
            score = 0.0

            while not done:
                s = self.getAgentState()
                actions = self.env.getPosActions()
                a = self.policy(s, actions)

                _raw, r, done = self.env.step(a)
                score += float(r)
                trajectory.append((s, a, float(r)))

            self.improve(trajectory)
            self.decay_epsilon()
            yield done, score, self.env.floor
            
    def improve(self, trajectory):
        # Cập nhật Q bằng first-visit Monte Carlo
        # trajectory[t] = (s_t, a_t, r_{t+1}):
        # s_t: state trước khi gọi step(a_t)
        # r_{t+1}: reward env trả về sau step đó

        # Return tại bước t: G_t = r_{t+1} + γ r_{t+2} + γ² … đến hết episode
        # First-visit: với mỗi (s,a), chỉ dùng G của lần đầu tiên (s,a) xuất hiện trong episode

        n = len(trajectory)
        if n == 0:
            return

        # Bước 1: tính G_t cho mọi t bằng công thức lùi G_t = r_{t+1} + γ * G_{t+1}
        G = 0.0
        Gs = [0.0] * n
        gamma = self.delta
        for t in range(n - 1, -1, -1):
            G = trajectory[t][2] + gamma * G
            Gs[t] = G

        # Bước 2: first-visit — chỉ table[s,a] += G_t khi (s,a) chưa gặp trong episode
        seen = set()
        for t in range(n):
            s, a, _ = trajectory[t]
            key = (s, a)
            if key in seen:
                continue
            seen.add(key)
            self.table[key] = Gs[t]

    def policy(self, s, actions):
        # Epsilon-greedy trên Q(s,·) với các action hợp lệ tại s
        # Các action cùng Q max: chia xác suất greedy đều; các action khác nhận phần ε còn lại
        # Nếu tất cả Q bằng nhau -> chọn ngẫu nhiên đều
        if not actions:
            actions = [0, 1, 2, 3, 8]

        qs = [(a, self.table[s, a]) for a in actions]
        if len(qs) == 1:
            return int(actions[0])

        max_q = max(q for _, q in qs)
        indexes = [i for i, (_a, q) in enumerate(qs) if q == max_q]

        if len(indexes) == len(qs):
            p = [1.0 / len(indexes)] * len(qs)
        else:
            p = [
                (1.0 - self.epsilon) / len(indexes) if i in indexes
                else self.epsilon / (len(qs) - len(indexes))
                for i in range(len(qs))
            ]
        return int(self.randGen.choice(actions, p=p))


# ===== Cell 23: 7.3. Các hàm trợ giúp train, evaluate MonteCarloControl =====
def save_mctable(table, path, protocol=4):
    # Lưu MCTable ra .pkl (SUM return + nCounts)
    _dir = os.path.dirname(os.path.abspath(path))
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    payload = {
        "version": 1,
        "default": table.default,
        "sums": {k: dict.__getitem__(table, k) for k in table},
        "nCounts": {k: int(table.nCounts[k]) for k in table},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=protocol)


def load_mctable(path, default_q=None):
    # Đọc file do save_mctable ghi -> MCTable
    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        payload = _U(f).load()

    if not isinstance(payload, dict) or "sums" not in payload or "nCounts" not in payload:
        raise ValueError("Không phải file MC (.pkl)")

    if default_q is not None:
        payload = {**payload, "default": float(default_q)}

    t = MCTable(float(payload["default"]))
    for k, sum_v in payload["sums"].items():
        dict.__setitem__(t, k, float(sum_v))
        t.nCounts[k] = int(payload["nCounts"][k])
    return t


def evaluate_mctable_on_ui(maze, load_path=None, episodes=1, fixed_map=True, sleep_time=0.1, seed=None, use_ui=True, default_q=0.0,):
    # Xem policy greedy trên UI (file từ save_mctable)
    global lastReward

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if load_path is None:
        mctable = MCTable(float(default_q))
    else:
        mctable = load_mctable(load_path, default_q=default_q)

    results = []
    first_reset = True

    for _ep in range(episodes):
        if fixed_map:
            if first_reset:
                env_seed = int(rng.integers(1_000_000_000))
                maze.reset(seed=env_seed, randomize_map=True)
                first_reset = False
            else:
                maze.reset(seed=None, randomize_map=False)
        else:
            env_seed = int(rng.integers(1_000_000_000))
            maze.reset(seed=env_seed, randomize_map=True)

        lastReward = 0.0
        if use_ui:
            z.z.angularBind("gameStatus", "Đang chơi")
            _updateUI()

        done = False
        total_reward = 0.0
        steps = 0
        agent = Agent()

        while not done:
            s = maze.getState(agent)
            actions = maze.getPosActions()
            if not actions:
                actions = [0, 1, 2, 3, 8]

            qs = [(a, mctable[s, a]) for a in actions]
            max_q = max(q for _, q in qs)
            best_actions = [a for a, q in qs if q == max_q]
            a = int(rng.choice(best_actions))

            _, r, done = maze.step(a)
            lastReward = float(r)
            total_reward += float(r)
            steps += 1

            if use_ui:
                _updateUI()
                time.sleep(sleep_time)

        if use_ui:
            _setGameStatus()

        # Format: (reward, số bước, tầng kết thúc, end reason)
        er = maze.end_reason
        if er is None:
            raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")
        results.append((float(total_reward), steps, int(maze.floor), er))

    return results


# ===== Cell 24: 7.4.1. Train MC với Decayed Epsilon Greedy | Phase 1 =====
# PHASE TRAINING 1
# SOFT WALL = 0
# BOMB = 0
# MAP CỐ ĐỊNH SEED 42
# Phase 1: 50k episode

MapCoDinhHayKhong_MC = True  # True = giữ 1 layout sau ep đầu; False = random layout mỗi episode

saveMCModel = True  # True = lưu checkpoint + best + final; False = không lưu file train
save_dir_mc = "MC_Phase1_F"

loadExistingMCTable = False  # True = train tiếp từ MCTable đã lưu
load_path_mc = os.path.join(save_dir_mc, "best_mctable.pkl")

# CHỈNH EPISODE / LOG TRƯỚC KHI TRAIN
episodeAmount_MC = 50000
printAfterEpisode_MC = 2000

if saveMCModel:
    os.makedirs(save_dir_mc, exist_ok=True)

# Chọn MonteCarloControl theo chế độ map (cố định vs random)
if MapCoDinhHayKhong_MC:
    # Map cố định:
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,  # explore đầu train; giảm dần theo decay_rate mỗi episode
        epsilon_min=0.05,
        # decay_rate: decay = (epsilon_min / epsilon_start) ** (1 / (episodeAmount * 75%))
        # epsilon chạm min quanh 75% tổng episode
        decay_rate=0.99992,
        defaultQ=0.0,
        maxSteps=250,
        seed=42,  # RNG epsilon-greedy + chuỗi env_seed khi random map / ep đầu fixed map
    )
else:
    # Map random:
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.1,  # random map: giữ explore cho trạng thái mới
        decay_rate=0.999979,
        defaultQ=0.0,
        maxSteps=300,
        seed=42,
    )

if loadExistingMCTable:
    try:
        agent_MC.table = load_mctable(load_path_mc)
        print(
            f"Đã load MCTable: {os.path.abspath(load_path_mc)} "
            f"({len(agent_MC.table)} (s,a))"
        )
    except FileNotFoundError:
        print(f"Không thấy file {load_path_mc}, train từ đầu.")

currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}
episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []
best_success = 0.0

for _done, score, end_floor in agent_MC.play(episodeAmount_MC, fixed_map=MapCoDinhHayKhong_MC):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)
    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode_MC == 0:
        score_tb = float(np.mean(episodeScoreResult[-printAfterEpisode_MC:]))
        recentEnds = endHistory[-printAfterEpisode_MC:]
        recentFloors = endFloorHistory[-printAfterEpisode_MC:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = float(np.mean(stepHistory[-printAfterEpisode_MC:]))
        goal_r = recentEnds.count("goal")
        death_r = recentEnds.count("death")
        timeout_r = recentEnds.count("timeout")
        stag_r = recentEnds.count("stagnation")
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode + 1}/{episodeAmount_MC} | "
            f"avg_reward={score_tb:.2f} | success_rate={success_rate:.2f} | avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} "
            f"timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode_MC} ep]: goal={goal_r} death={death_r} timeout={timeout_r} stagnation={stag_r} | "
            f"epsilon={agent_MC.epsilon:.3f}"
        )
        print(f"Tầng kết thúc [{printAfterEpisode_MC} ep]: {floor_stat}")
        print(f"Số (s,a) trong MCTable: {len(agent_MC.table)}")

        if saveMCModel:
            ckpt = os.path.join(save_dir_mc, f"mctable_ep{currentEpisode + 1}.pkl")
            save_mctable(agent_MC.table, ckpt)
            print("[MC] Saved checkpoint:", os.path.abspath(ckpt))
            if success_rate > best_success:
                best_success = success_rate
                best_p = os.path.join(save_dir_mc, "best_mctable.pkl")
                save_mctable(agent_MC.table, best_p)
                print("New BEST:", os.path.abspath(best_p))

    currentEpisode += 1

print(f"\n[Xong. Episodes: {episodeAmount_MC}")
print(f"Điểm TB: {float(np.mean(episodeScoreResult)):.2f}")
print(
    f"goal={endType['goal']} death={endType['death']} "
    f"timeout={endType['timeout']} stagnation={endType['stagnation']}"
)

if saveMCModel:
    final_p = os.path.join(save_dir_mc, "mctable_final.pkl")
    save_mctable(agent_MC.table, final_p)
    print("Final:", os.path.abspath(final_p))

# Biểu đồ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50
max_floor = int(TOTAL_FLOORS) - 1

plt.figure(figsize=(14, 16))

plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    sm = np.convolve(episodeScoreResult, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sm, linewidth=2, label="Moving Avg")
plt.title("Return (Monte Carlo FV)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory], dtype=float)
if len(success) >= window:
    sr = np.convolve(success, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sr)
else:
    plt.plot(episodes, success)
plt.title("Success Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
rates = []
for f in range(max_floor + 1):
    is_f = (end_floors == f).astype(float)
    if len(is_f) >= window:
        rates.append(np.convolve(is_f, np.ones(window) / window, mode="valid"))
    else:
        rates.append(is_f)
x_stack = episodes[window - 1 :] if len(end_floors) >= window else episodes
plt.stackplot(
    x_stack,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8,
)
plt.title("End Floor Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    st = np.convolve(stepHistory, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], st, linewidth=2)
plt.title("Episode Length (MC)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 25: 7.4.2. Train MC với Decayed Epsilon Greedy | Phase 2 =====
# PHASE TRAINING 2
# SOFT WALL = 5
# BOMB = 2
# MAP CỐ ĐỊNH SEED 42
# Phase 2: 50k episode

MapCoDinhHayKhong_MC = True  # True = giữ 1 layout sau ep đầu; False = random layout mỗi episode

saveMCModel = True  # True = lưu checkpoint + best + final; False = không lưu file train
save_dir_mc = "MC_Phase2_F"

loadExistingMCTable = True  # True = train tiếp từ MCTable đã lưu
old_dir_mc = "MC_Phase1_F"
load_path_mc = os.path.join(old_dir_mc, "mctable_final.pkl")

# CHỈNH EPISODE / LOG TRƯỚC KHI TRAIN
episodeAmount_MC = 50000
printAfterEpisode_MC = 2000

if saveMCModel:
    os.makedirs(save_dir_mc, exist_ok=True)

# Chọn MonteCarloControl theo chế độ map (cố định vs random)
if MapCoDinhHayKhong_MC:
    # Map cố định:
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,  # explore đầu train; giảm dần theo decay_rate mỗi episode
        epsilon_min=0.05,
        # decay_rate: decay = (epsilon_min / epsilon_start) ** (1 / (episodeAmount * 75%))
        # epsilon chạm min quanh 75% tổng episode
        decay_rate=0.99992,
        defaultQ=0.0,
        maxSteps=250,
        seed=42,  # RNG epsilon-greedy + chuỗi env_seed khi random map / ep đầu fixed map
    )
else:
    # Map random:
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.1,  # random map: giữ explore cho trạng thái mới
        decay_rate=0.999979,
        defaultQ=0.0,
        maxSteps=300,
        seed=42,
    )

if loadExistingMCTable:
    try:
        agent_MC.table = load_mctable(load_path_mc)
        print(
            f"Đã load MCTable: {os.path.abspath(load_path_mc)} "
            f"({len(agent_MC.table)} (s,a))"
        )
    except FileNotFoundError:
        print(f"Không thấy file {load_path_mc}, train từ đầu.")

currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}
episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []
best_success = 0.0

for _done, score, end_floor in agent_MC.play(episodeAmount_MC, fixed_map=MapCoDinhHayKhong_MC):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)
    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode_MC == 0:
        score_tb = float(np.mean(episodeScoreResult[-printAfterEpisode_MC:]))
        recentEnds = endHistory[-printAfterEpisode_MC:]
        recentFloors = endFloorHistory[-printAfterEpisode_MC:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = float(np.mean(stepHistory[-printAfterEpisode_MC:]))
        goal_r = recentEnds.count("goal")
        death_r = recentEnds.count("death")
        timeout_r = recentEnds.count("timeout")
        stag_r = recentEnds.count("stagnation")
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode + 1}/{episodeAmount_MC} | "
            f"avg_reward={score_tb:.2f} | success_rate={success_rate:.2f} | avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} "
            f"timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode_MC} ep]: goal={goal_r} death={death_r} timeout={timeout_r} stagnation={stag_r} | "
            f"epsilon={agent_MC.epsilon:.3f}"
        )
        print(f"Tầng kết thúc [{printAfterEpisode_MC} ep]: {floor_stat}")
        print(f"Số (s,a) trong MCTable: {len(agent_MC.table)}")

        if saveMCModel:
            ckpt = os.path.join(save_dir_mc, f"mctable_ep{currentEpisode + 1}.pkl")
            save_mctable(agent_MC.table, ckpt)
            print("[MC] Saved checkpoint:", os.path.abspath(ckpt))
            if success_rate > best_success:
                best_success = success_rate
                best_p = os.path.join(save_dir_mc, "best_mctable.pkl")
                save_mctable(agent_MC.table, best_p)
                print("New BEST:", os.path.abspath(best_p))

    currentEpisode += 1

print(f"\nXong. Episodes: {episodeAmount_MC}")
print(f"Điểm TB: {float(np.mean(episodeScoreResult)):.2f}")
print(
    f"goal={endType['goal']} death={endType['death']} "
    f"timeout={endType['timeout']} stagnation={endType['stagnation']}"
)

if saveMCModel:
    final_p = os.path.join(save_dir_mc, "mctable_final.pkl")
    save_mctable(agent_MC.table, final_p)
    print("Final:", os.path.abspath(final_p))

# Biểu đồ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50
max_floor = int(TOTAL_FLOORS) - 1

plt.figure(figsize=(14, 16))

plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    sm = np.convolve(episodeScoreResult, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sm, linewidth=2, label="Moving Avg")
plt.title("Return (Monte Carlo FV)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory], dtype=float)
if len(success) >= window:
    sr = np.convolve(success, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sr)
else:
    plt.plot(episodes, success)
plt.title("Success Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
rates = []
for f in range(max_floor + 1):
    is_f = (end_floors == f).astype(float)
    if len(is_f) >= window:
        rates.append(np.convolve(is_f, np.ones(window) / window, mode="valid"))
    else:
        rates.append(is_f)
x_stack = episodes[window - 1 :] if len(end_floors) >= window else episodes
plt.stackplot(
    x_stack,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8,
)
plt.title("End Floor Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    st = np.convolve(stepHistory, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], st, linewidth=2)
plt.title("Episode Length (MC)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 26: 7.4.3. Train MC với Decayed Epsilon Greedy | Phase 3 =====
# PHASE TRAINING 3
# SOFT WALL = 5
# BOMB = 2
# MAP RANDOM SEED 42
# Phase 3: 50k episode

MapCoDinhHayKhong_MC = False  # True = giữ 1 layout sau ep đầu; False = random layout mỗi episode

saveMCModel = True  # True = lưu checkpoint + best + final; False = không lưu file train
save_dir_mc = "MC_Phase3_F"

loadExistingMCTable = True  # True = train tiếp từ MCTable đã lưu
old_dir_mc = "MC_Phase2_F"
load_path_mc = os.path.join(old_dir_mc, "mctable_final.pkl")

# CHỈNH EPISODE / LOG TRƯỚC KHI TRAIN
episodeAmount_MC = 50000
printAfterEpisode_MC = 2000

if saveMCModel:
    os.makedirs(save_dir_mc, exist_ok=True)

# Chọn MonteCarloControl theo chế độ map (cố định vs random)
if MapCoDinhHayKhong_MC:
    # Map cố định: 50k
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,  # explore đầu train; giảm dần theo decay_rate mỗi episode
        epsilon_min=0.05,
        # decay_rate: decay = (epsilon_min / epsilon_start) ** (1 / (episodeAmount * 75%))
        # epsilon chạm min quanh 75% tổng episode
        decay_rate=0.99992,
        defaultQ=0.0,
        maxSteps=250,
        seed=42,  # RNG epsilon-greedy + chuỗi env_seed khi random map / ep đầu fixed map
    )
else:
    # Map random: 50k
    agent_MC = MonteCarloControl(
        maze,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.1,  # random map: giữ explore cho trạng thái mới
        decay_rate=0.9999385,
        defaultQ=0.0,
        maxSteps=300,
        seed=42,
    )

if loadExistingMCTable:
    try:
        agent_MC.table = load_mctable(load_path_mc)
        print(
            f"Đã load MCTable: {os.path.abspath(load_path_mc)} "
            f"({len(agent_MC.table)} (s,a))"
        )
    except FileNotFoundError:
        print(f"Không thấy file {load_path_mc}, train từ đầu.")

currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}
episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []
best_success = 0.0

for _done, score, end_floor in agent_MC.play(episodeAmount_MC, fixed_map=MapCoDinhHayKhong_MC):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)
    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode_MC == 0:
        score_tb = float(np.mean(episodeScoreResult[-printAfterEpisode_MC:]))
        recentEnds = endHistory[-printAfterEpisode_MC:]
        recentFloors = endFloorHistory[-printAfterEpisode_MC:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = float(np.mean(stepHistory[-printAfterEpisode_MC:]))
        goal_r = recentEnds.count("goal")
        death_r = recentEnds.count("death")
        timeout_r = recentEnds.count("timeout")
        stag_r = recentEnds.count("stagnation")
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode + 1}/{episodeAmount_MC} | "
            f"avg_reward={score_tb:.2f} | success_rate={success_rate:.2f} | avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} "
            f"timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode_MC} ep]: goal={goal_r} death={death_r} timeout={timeout_r} stagnation={stag_r} | "
            f"epsilon={agent_MC.epsilon:.3f}"
        )
        print(f"Tầng kết thúc [{printAfterEpisode_MC} ep]: {floor_stat}")
        print(f"Số (s,a) trong MCTable: {len(agent_MC.table)}")

        if saveMCModel:
            ckpt = os.path.join(save_dir_mc, f"mctable_ep{currentEpisode + 1}.pkl")
            save_mctable(agent_MC.table, ckpt)
            print("[MC] Saved checkpoint:", os.path.abspath(ckpt))
            if success_rate > best_success:
                best_success = success_rate
                best_p = os.path.join(save_dir_mc, "best_mctable.pkl")
                save_mctable(agent_MC.table, best_p)
                print("New BEST:", os.path.abspath(best_p))

    currentEpisode += 1

print(f"\nXong. Episodes: {episodeAmount_MC}")
print(f"Điểm TB: {float(np.mean(episodeScoreResult)):.2f}")
print(
    f"goal={endType['goal']} death={endType['death']} "
    f"timeout={endType['timeout']} stagnation={endType['stagnation']}"
)

if saveMCModel:
    final_p = os.path.join(save_dir_mc, "mctable_final.pkl")
    save_mctable(agent_MC.table, final_p)
    print("Final:", os.path.abspath(final_p))

# Biểu đồ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50
max_floor = int(TOTAL_FLOORS) - 1

plt.figure(figsize=(14, 16))

plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    sm = np.convolve(episodeScoreResult, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sm, linewidth=2, label="Moving Avg")
plt.title("Return (Monte Carlo FV)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory], dtype=float)
if len(success) >= window:
    sr = np.convolve(success, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], sr)
else:
    plt.plot(episodes, success)
plt.title("Success Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
rates = []
for f in range(max_floor + 1):
    is_f = (end_floors == f).astype(float)
    if len(is_f) >= window:
        rates.append(np.convolve(is_f, np.ones(window) / window, mode="valid"))
    else:
        rates.append(is_f)
x_stack = episodes[window - 1 :] if len(end_floors) >= window else episodes
plt.stackplot(
    x_stack,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8,
)
plt.title("End Floor Rate (MC)")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    st = np.convolve(stepHistory, np.ones(window) / window, mode="valid")
    plt.plot(episodes[window - 1 :], st, linewidth=2)
plt.title("Episode Length (MC)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 27: 7.5. Evaluate MonteCarloControl với Decayed Epsilon Greedy =====
mc_eval_results = evaluate_mctable_on_ui(
    maze,
    load_path=os.path.join("FinalModel/MC2", "mctable_final.pkl"),
    episodes=1000,
    fixed_map=True,
    sleep_time=0.1,
    seed=42,
    use_ui=False,
    default_q=0.0,
)

print("\nChi tiết từng episode:")
for ep_i, (total_r, n_steps, end_floor, end_type) in enumerate(mc_eval_results, start=1):
    print(
        f"  Ep {ep_i:03d} | reward={total_r:8.2f} | steps={n_steps:4d} | "
        f"end_floor={end_floor} | end_type={end_type}"
    )

if len(mc_eval_results) > 0:
    rewards = np.array([r for r, _, _, _ in mc_eval_results], dtype=float)
    steps_a = np.array([s for _, s, _, _ in mc_eval_results], dtype=float)
    ends = [t for _, _, _, t in mc_eval_results]
    print(
        f"\nTổng quan | avg_reward={rewards.mean():.2f} | avg_steps={steps_a.mean():.1f} | "
        f"goal={ends.count('goal')} death={ends.count('death')} "
        f"timeout={ends.count('timeout')} stagnation={ends.count('stagnation')}"
    )
