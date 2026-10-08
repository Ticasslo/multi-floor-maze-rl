"""Tabular Q-learning: Q-table, the agent, train/evaluate helpers, the 3 training phases and the evaluation run.

Copied cell by cell, without changes, from the Zeppelin note Nhom08_MultiFloorMaze_Code.json so the code can be read on GitHub.
Some helpers call Zeppelin's z object for the game UI, so run the code in the notebook (or the Colab notebook for Double DQN).
"""

# ===== Cell 29: 8.1. Định nghĩa QTable_QLearning =====
# QTable của QLearning được lưu ở dạng dict gồm key-value với key = (state, action) || value = Q(s,a)
# State bao gồm x số, action là 1 số đại diện cho 1 hành động -> Trả về value của hành động đó là Q
class QTable_QLearning(dict):
    def __init__(self, default):
        dict.__init__(self)
        self.default = default
        
    def __getitem__(self, k):
        return super().get(k, self.default)
    
    def __setitem__(self, k, v):
        super().__setitem__(k, v)
    
    # Ở trạng thái state s, thì các actions mỗi action sẽ có một Q
    # Chọn ra và trả về giá trị Q = value lớn nhất -> Dùng cập nhật Q value khi train theo policy
    def max(self, s, actions):
        if not actions:
            return self.default
        return max(self[s, a] for a in actions)


# ===== Cell 30: 8.2. Định nghĩa class QLearning =====
# QLearning dùng cho MazeEnv: state = getState(Agent()), action chỉ trong getPosActions()
class QLearning:
    def __init__(self, env, lrate, discount, epsilon, epsilon_min, decay_rate, defaultQ, maxSteps=None, seed=None):
        self.env = env
        # Learning rate (α)
        self.alpha = lrate
        # Discount factor (γ)
        self.delta = discount
        
        # Exponential Decayed Epsilon-greedy: bắt đầu từ epsilon, decay xuống epsilon_min sau mỗi episode
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        
        self.table = QTable_QLearning(defaultQ)
        self.seed = seed

        # Dùng cho chế độ map cố định: đánh dấu đã random map 1 lần đầu tiên chưa, để thực hiện random map theo seed 1 lần rồi dùng map đó cố định luôn
        self.fixedMapRandom1Time = False

        self.agent = Agent()
        
        if maxSteps is not None:
            self.env.maxSteps = maxSteps
            
        if self.seed is not None:
            self.randGen = np.random.default_rng(seed=seed)
        else:
            self.randGen = np.random.default_rng()


    def decay_epsilon(self):
        # Giảm epsilon nhưng không được thấp hơn mức tối thiểu
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
            
    def getAgentState(self):
        return self.env.getState(self.agent)

    def play(self, n=1, fixed_map=False):
        for i in range(n):
            if fixed_map:
                # Map cố định:
                # - Lần đầu: random map theo seed (nếu có) để layout phụ thuộc QLearning.seed
                # - Các lần sau: giữ nguyên layout, chỉ reset vị trí/máu/monster
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
                # Map random nhưng chuỗi sinh map là một chuỗi cố định, kể cả layout lẫn vị trí item random theo seed chuỗi qua từng episode
                env_seed = int(self.randGen.integers(1_000_000_000))
                self.env.reset(seed=env_seed, randomize_map=True)
            else:
                # Map layout random hoàn toàn, cả item cũng random mỗi episode
                self.env.reset(seed=None, randomize_map=True)
            
            # Lấy state sau khi reset môi trường, st
            s = self.getAgentState()
            
            done = False
            score = 0.0

            while not done:
                # B0: Xét các actions khả thi có thể thực hiện tại state st
                actions = self.env.getPosActions()
                
                # B1: chọn action at tại state st theo policy (epsilon-greedy)
                # env hiện đang ở st nên các actions possible sẽ theo st
                a = self.policy(s, actions)
                
                # B2: thực hiện action at -> môi trường chuyển sang trạng thái mới st+1 nhận reward r và flag done kết thúc episode
                _, r, done = self.env.step(a)
                # B3: lấy state st+1 mới sau khi thực hiện action at
                s_next = self.getAgentState()
                score += r
                    
                # B4: cập nhật Q-value cho cặp (s, a) dùng reward hiện tại và ước lượng giá trị tương lai tại s_next
                # env hiện đang ở s_t+1 nên các actions possible sẽ theo s_t+1
                # Lấy danh sách hành động khả thi tại state s_t+1
                actions_next = self.env.getPosActions()
                self.improve(s, a, s_next, actions_next, r, done)
                
                # B5: chuyển state st hiện tại sang state mới st+1
                s = s_next
            
            # Thực hiện Exponential Decayed Epsilon-greedy
            self.decay_epsilon()
            
            # Ghi lại tầng kết thúc
            end_floor = self.env.floor
            yield done, score, end_floor


    # Cập nhật Q(st,at) theo công thức Q-learning
    def improve(self, st, at, s_next, actions_next, r, done):
        if done:
            # Nếu game kết thúc thì giá trị tương lai/ reward tương lai = 0 nên ko cần xét
            tdTarget = r
        else:
            # env hiện đang ở s_t+1 nên các actions possible at+1 sẽ theo s_t+1
            # Tìm Max Q của trạng thái kế tiếp/ giá trị tương lai
            tdTarget = r + self.delta * self.table.max(s_next, actions_next)
            
        old_q = self.table[st, at]
        self.table[st, at] = (1 - self.alpha) * old_q + self.alpha * tdTarget


    # Epsilon-greedy policy
    def policy(self, s, actions):
        # Nếu mà kẹt/ ko có hành động nào khả thi thì random action move: 0 tới 3,
        # Không thêm búa ở đây vì búa chỉ hợp lệ khi hammerTimer == 0 (đang cooldown thì không có trong actions).
        if not actions:
            # fallback khi hiếm khi actions rỗng: cho phép agent WAIT để tránh invalid loop
            actions = [0, 1, 2, 3, 8]
            
        # Lấy Q-value cho từng action tại st
        qs = [(a, self.table[s, a]) for a in actions]
    
        # Nếu chỉ có 1 action thì chọn luôn
        if len(qs) == 1:
            return actions[0]

        # Tìm max Q-value
        max_q = max(q for _, q in qs)
        indexes = [i for i, (a, q) in enumerate(qs) if q == max_q]
    
        # Nếu tất cả action có Q giống nhau -> chọn ngẫu nhiên đều
        if len(indexes) == len(qs):
            p = [1 / len(indexes)] * len(indexes)

        # Ngược lại: epsilon-greedy
        else:
            p = [
                (1 - self.epsilon) / len(indexes) if i in indexes
                else self.epsilon / (len(qs) - len(indexes))
                for i in range(len(qs))
            ]
    
        # Chọn action theo phân phối xác suất p
        return self.randGen.choice(actions, p=p)


# ===== Cell 31: 8.3. Các hàm trợ giúp train, evaluate QLearning =====
def loadQTable(path, default_q=0.0):
    # Load Q-table từ file pickle (dict[(state, action) -> Q]) và trả về đối tượng QTable_QLearning tương ứng
    table = QTable_QLearning(default_q)
    with open(path, "rb") as f:
        # Một số Q-table pickle có thể được tạo từ môi trường numpy khác
        # Dùng unpickler compat để map numpy._core -> numpy.core
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core")
                return super().find_class(module, name)
        raw = _CompatUnpickler(f).load()
        
    if not isinstance(raw, dict):
        raise ValueError("Q-table sai định dạng (phải là dict[(state, action) -> Q])")
        
    table.update(raw)
    return table

def evaluate_qtable_on_ui(load_path=None, episodes=1, fixed_map=True, sleep_time=0.1, seed=None, use_ui=True, default_q=0.0):
    # Dùng Q-table đã train để cho agent chạy tự động trên UI.
    # Tham số:
    # - load_path  : đường dẫn tới file qtable.pkl; None -> dùng Q-table rỗng (baseline chưa train)
    # - episodes   : số episode chạy thử
    # - fixed_map  : True -> dùng 1 layout map cố định, False -> mỗi episode random layout mới
    # - sleep_time : thời gian giữa các step để nhìn UI (giây)
    # - seed       : seed chung cho RNG; truyền None nếu muốn mỗi lần evaluate ra chuỗi random khác nhau
    # - use_ui     : True -> cập nhật UI, False -> chạy headless (không gọi hàm UI)
    global lastReward

    # Seed toàn bộ RNG (random, numpy, rng nội bộ) nếu seed được truyền vào
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if load_path is None:
        # Baseline chưa train: mọi Q(s,a) = default_q
        qtable = QTable_QLearning(default_q)
    else:
        qtable = loadQTable(load_path, default_q=default_q)

    results = []

    first_reset = True

    for ep in range(episodes):
        # Reset map theo chế độ fixed / random
        if fixed_map:
            # Fixed map: lần đầu randomize map 1 lần theo seed nội bộ, sau đó giữ nguyên layout
            if first_reset:
                env_seed = int(rng.integers(1_000_000_000))
                maze.reset(seed=env_seed, randomize_map=True)
                first_reset = False
            else:
                maze.reset(seed=None, randomize_map=False)
        else:
            # Map random: mỗi episode random layout mới theo chuỗi env_seed
            env_seed = int(rng.integers(1_000_000_000))
            maze.reset(seed=env_seed, randomize_map=True)

        # Reset UI status (nếu bật UI)
        lastReward = 0.0
        if use_ui:
            z.z.angularBind('gameStatus', 'Đang chơi')
            _updateUI()

        done = False
        total_reward = 0.0
        steps = 0
        agent = Agent()

        while not done:
            # Lấy state & action hợp lệ
            s = maze.getState(agent)
            actions = maze.getPosActions()
            if not actions:
                # fallback move
                actions = [0, 1, 2, 3, 8]

            # Chọn action greedy từ Q-table (argmax, tie-break random)
            qs = [(a, qtable[s, a]) for a in actions]
            max_q = max(q for _, q in qs)
            best_actions = [a for a, q in qs if q == max_q]
            a = rng.choice(best_actions)

            # Thực hiện step
            _, r, done = maze.step(a)
            lastReward = float(r)
            total_reward += r
            steps += 1

            # Cập nhật UI mỗi bước (nếu bật UI)
            if use_ui:
                _updateUI()
                time.sleep(sleep_time)

        # Cập nhật trạng thái thắng/thua trên UI (nếu bật UI)
        if use_ui:
            _setGameStatus()

        er = maze.end_reason
        if er is None:
            raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")
        # (reward, steps, end_floor, end_reason)
        results.append((float(total_reward), steps, int(maze.floor), er))

    return results


# ===== Cell 32: 8.4.1. Train QLearning với Decayed Epsilon Greedy | Phase 1 =====
# PHASE TRAINING 1
# SOFT WALL = 0
# BOMB = 0
# MAP CỐ ĐỊNH SEED 42
# Phase 1: 50k episode

MapCoDinhHayKhong = True   # True = train trên 1 map cố định; False = map random mỗi episode     #<----------------------------------

# CÓ ĐỊNH LƯU Q TABLE KO
saveQLearningModel = True  # True = lưu checkpoint + best + final; False = không lưu file train

save_dir = "QLearning_Phase1_F"                                                                    #<----------------------------------
if saveQLearningModel:
    os.makedirs(save_dir, exist_ok=True)
    
# CÓ ĐỊNH LOAD LẠI Q TABLE ĐỂ TRAIN TIẾP TỪ Q TABLE ĐÓ KO
loadExistingQTable = False

load_path = "QLearning_Phase1/best_qtable.pkl"                                                  #<----------------------------------


# CHỈNH EPISODE TRƯỚC KHI TRAIN!!!!!!!                                                           #<----------------------------------
episodeAmount = 50000
printAfterEpisode = 2000

# Chọn setting Q-learning theo chế độ map (cố định vs random)
if MapCoDinhHayKhong:
    # Map cố định:
    agent_QLearning = QLearning(
        maze,
        lrate=0.08,
        discount=0.98,  # Đã test, chọn 0.98 hoặc 0.97
        epsilon=1.0,                      # Lưu ý setting epsilon nếu đang định train model mới, cho nó explore: ep lớn hay muốn exploit: ep nhỏ. Nhất là khi load lại QTable
        epsilon_min=0.05,
        decay_rate=0.99992,   # decay = (ep_min / ep) ** (1 / (episodeAmount * 75%)), để chạm min ở 75% của tổng episode
        defaultQ=0,
        maxSteps=250,
        seed=42
    )
else:
    # Map random: 150k ep
    agent_QLearning = QLearning(
        maze,
        lrate=0.06,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.1, # Random map cần 1 khoảng explore nếu gặp trạng thái mới
        decay_rate=0.999979,
        defaultQ=0,
        maxSteps=300,
        seed=42
    )


# Nếu muốn train tiếp từ Q-table cũ thì load
if loadExistingQTable:
    try:
        loaded_table = loadQTable(load_path, default_q=0.0)  # loader compat numpy._core
        # loadQTable trả về QTable_QLearning, update thẳng vào table đang train
        agent_QLearning.table.update(dict(loaded_table))
        print(f"Loaded Q-table from {os.path.abspath(load_path)} "
              f"({len(loaded_table)} entries).")
    except FileNotFoundError:
        print(f"Không tìm thấy file Q-table tại {load_path}, train từ đầu")


currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}

episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []

best_success = 0
    

for done, score, end_floor in agent_QLearning.play(episodeAmount, fixed_map=MapCoDinhHayKhong):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)

    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode == 0:

        scoreTrungBinh = np.mean(episodeScoreResult[-printAfterEpisode:])
        recentEnds = endHistory[-printAfterEpisode:]
        recentFloors = endFloorHistory[-printAfterEpisode:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = np.mean(stepHistory[-printAfterEpisode:])
        goal_recent = recentEnds.count("goal")
        death_recent = recentEnds.count("death")
        timeout_recent = recentEnds.count("timeout")
        stagnation_recent = recentEnds.count("stagnation")
        # Thống kê tầng kết thúc: số episode kết thúc ở từng tầng
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode+1}/{episodeAmount} | "
            f"avg_reward={scoreTrungBinh:.2f} | "
            f"success_rate={success_rate:.2f} | "
            f"avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode} ep gần nhất]: goal={goal_recent} death={death_recent} timeout={timeout_recent} stagnation={stagnation_recent} | "
            f"epsilon={agent_QLearning.epsilon:.3f}"
        )
        print(f"Vị trí tầng kết thúc [{printAfterEpisode} ep]: {floor_stat}")
        print("Q-table size:", len(agent_QLearning.table))

        if saveQLearningModel:
            # SAVE CHECKPOINT
            checkpoint_path = os.path.join(
                save_dir, f"qtable_ep{currentEpisode+1}.pkl"
            )
            with open(checkpoint_path, "wb") as f:
                # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                pickle.dump(dict(agent_QLearning.table), f, protocol=4)
            print("Saved checkpoint:", os.path.abspath(checkpoint_path))

            # SAVE BEST MODEL
            if success_rate > best_success:
                best_success = success_rate
                best_path = os.path.join(save_dir, "best_qtable.pkl")
                with open(best_path, "wb") as f:
                    # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                    pickle.dump(dict(agent_QLearning.table), f, protocol=4)
                print("\nNew BEST model saved:", os.path.abspath(best_path))

    currentEpisode += 1


print(f"\nTotal episodes: {episodeAmount}")
print(f"Điểm TB:        {np.mean(episodeScoreResult):.2f}")
print(f"Goals:          {endType['goal']}")
print(f"Deaths:         {endType['death']}")
print(f"Timeout:        {endType['timeout']}")
print(f"Stagnation:     {endType['stagnation']}")


# SAVE FINAL MODEL
if saveQLearningModel:
    final_path = os.path.join(save_dir, "qtable_final.pkl")
    with open(final_path, "wb") as f:
        # Dùng protocol=4 để tương thích môi trường Python cũ hơn
        pickle.dump(dict(agent_QLearning.table), f, protocol=4)
    print("\nFinal model saved:", os.path.abspath(final_path))


# VẼ ĐỒ THỊ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50

plt.figure(figsize=(14, 16))

# RETURN
plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    smoothed = np.convolve(
        episodeScoreResult,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], smoothed, linewidth=2, label="Moving Avg")
plt.title("Return")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# SUCCESS RATE
plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory])
if len(success) >= window:
    success_rate = np.convolve(
        success,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], success_rate)
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

# END FLOOR RATE
plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
max_floor = int(TOTAL_FLOORS) - 1
rates = []
for f in range(max_floor + 1):
    is_floor_f = (end_floors == f).astype(float)
    if len(is_floor_f) >= window:
        floor_rate = np.convolve(
            is_floor_f,
            np.ones(window) / window,
            mode="valid"
        )
        rates.append(floor_rate)
    else:
        rates.append(is_floor_f)
# Stacked area
if len(end_floors) >= window:
    x = episodes[window - 1:]
else:
    x = episodes
plt.stackplot(
    x,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8
)
plt.title("End Floor Rate")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

# EPISODE LENGTH
plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    step_smooth = np.convolve(
        stepHistory,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], step_smooth, linewidth=2)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 33: 8.4.2. Train QLearning với Decayed Epsilon Greedy | Phase 2 =====
# PHASE TRAINING 2
# SOFT WALL = 5
# BOMB = 2
# MAP CỐ ĐỊNH SEED 42
# Phase 2: 50k episode

MapCoDinhHayKhong = True   # True = train trên 1 map cố định; False = map random mỗi episode     #<----------------------------------

# CÓ ĐỊNH LƯU Q TABLE KO
saveQLearningModel = True  # True = lưu checkpoint + best + final; False = không lưu file train

save_dir = "QLearning_Phase2_F"                                                                    #<----------------------------------
if saveQLearningModel:
    os.makedirs(save_dir, exist_ok=True)
    
# CÓ ĐỊNH LOAD LẠI Q TABLE ĐỂ TRAIN TIẾP TỪ Q TABLE ĐÓ KO
loadExistingQTable = True

load_path = "QLearning_Phase1_F/best_qtable.pkl"                                                  #<----------------------------------


# CHỈNH EPISODE TRƯỚC KHI TRAIN!!!!!!!                                                           #<----------------------------------
episodeAmount = 50000
printAfterEpisode = 2000

# Chọn setting Q-learning theo chế độ map (cố định vs random)
if MapCoDinhHayKhong:
    # Map cố định:
    agent_QLearning = QLearning(
        maze,
        lrate=0.08,
        discount=0.98,  # Đã test, chọn 0.98 hoặc 0.97
        epsilon=1.0,                      # Lưu ý setting epsilon nếu đang định train model mới, cho nó explore: ep lớn hay muốn exploit: ep nhỏ. Nhất là khi load lại QTable
        epsilon_min=0.05,
        decay_rate=0.99992,   # decay = (ep_min / ep) ** (1 / (episodeAmount * 75%)), để chạm min ở 75% của tổng episode
        defaultQ=0,
        maxSteps=250,
        seed=42
    )
else:
    # Map random: 200k ep
    agent_QLearning = QLearning(
        maze,
        lrate=0.06,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.15, # Random map cần 1 khoảng explore nếu gặp trạng thái mới
        decay_rate=0.999987,
        defaultQ=0,
        maxSteps=300,
        seed=42
    )


# Nếu muốn train tiếp từ Q-table cũ thì load
if loadExistingQTable:
    try:
        loaded_table = loadQTable(load_path, default_q=0.0)  # loader compat numpy._core
        # loadQTable trả về QTable_QLearning, update thẳng vào table đang train
        agent_QLearning.table.update(dict(loaded_table))
        print(f"Loaded Q-table from {os.path.abspath(load_path)} "
              f"({len(loaded_table)} entries).")
    except FileNotFoundError:
        print(f"Không tìm thấy file Q-table tại {load_path}, train từ đầu")


currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}

episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []

best_success = 0
    

for done, score, end_floor in agent_QLearning.play(episodeAmount, fixed_map=MapCoDinhHayKhong):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)

    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode == 0:

        scoreTrungBinh = np.mean(episodeScoreResult[-printAfterEpisode:])
        recentEnds = endHistory[-printAfterEpisode:]
        recentFloors = endFloorHistory[-printAfterEpisode:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = np.mean(stepHistory[-printAfterEpisode:])
        goal_recent = recentEnds.count("goal")
        death_recent = recentEnds.count("death")
        timeout_recent = recentEnds.count("timeout")
        stagnation_recent = recentEnds.count("stagnation")
        # Thống kê tầng kết thúc: số episode kết thúc ở từng tầng
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode+1}/{episodeAmount} | "
            f"avg_reward={scoreTrungBinh:.2f} | "
            f"success_rate={success_rate:.2f} | "
            f"avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode} ep gần nhất]: goal={goal_recent} death={death_recent} timeout={timeout_recent} stagnation={stagnation_recent} | "
            f"epsilon={agent_QLearning.epsilon:.3f}"
        )
        print(f"Vị trí tầng kết thúc [{printAfterEpisode} ep]: {floor_stat}")
        print("Q-table size:", len(agent_QLearning.table))

        if saveQLearningModel:
            # SAVE CHECKPOINT
            checkpoint_path = os.path.join(
                save_dir, f"qtable_ep{currentEpisode+1}.pkl"
            )
            with open(checkpoint_path, "wb") as f:
                # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                pickle.dump(dict(agent_QLearning.table), f, protocol=4)
            print("Saved checkpoint:", os.path.abspath(checkpoint_path))

            # SAVE BEST MODEL
            if success_rate > best_success:
                best_success = success_rate
                best_path = os.path.join(save_dir, "best_qtable.pkl")
                with open(best_path, "wb") as f:
                    # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                    pickle.dump(dict(agent_QLearning.table), f, protocol=4)
                print("\nNew BEST model saved:", os.path.abspath(best_path))

    currentEpisode += 1


print(f"\nTotal episodes: {episodeAmount}")
print(f"Điểm TB:        {np.mean(episodeScoreResult):.2f}")
print(f"Goals:          {endType['goal']}")
print(f"Deaths:         {endType['death']}")
print(f"Timeout:        {endType['timeout']}")
print(f"Stagnation:     {endType['stagnation']}")


# SAVE FINAL MODEL
if saveQLearningModel:
    final_path = os.path.join(save_dir, "qtable_final.pkl")
    with open(final_path, "wb") as f:
        # Dùng protocol=4 để tương thích môi trường Python cũ hơn
        pickle.dump(dict(agent_QLearning.table), f, protocol=4)
    print("\nFinal model saved:", os.path.abspath(final_path))


# VẼ ĐỒ THỊ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50

plt.figure(figsize=(14, 16))

# RETURN
plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    smoothed = np.convolve(
        episodeScoreResult,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], smoothed, linewidth=2, label="Moving Avg")
plt.title("Return")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# SUCCESS RATE
plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory])
if len(success) >= window:
    success_rate = np.convolve(
        success,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], success_rate)
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

# END FLOOR RATE
plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
max_floor = int(TOTAL_FLOORS) - 1
rates = []
for f in range(max_floor + 1):
    is_floor_f = (end_floors == f).astype(float)
    if len(is_floor_f) >= window:
        floor_rate = np.convolve(
            is_floor_f,
            np.ones(window) / window,
            mode="valid"
        )
        rates.append(floor_rate)
    else:
        rates.append(is_floor_f)
# Stacked area
if len(end_floors) >= window:
    x = episodes[window - 1:]
else:
    x = episodes
plt.stackplot(
    x,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8
)
plt.title("End Floor Rate")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

# EPISODE LENGTH
plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    step_smooth = np.convolve(
        stepHistory,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], step_smooth, linewidth=2)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 34: 8.4.3. Train QLearning với Decayed Epsilon Greedy | Phase 3 =====
# PHASE TRAINING 3
# SOFT WALL = 5
# BOMB = 2
# MAP RANDOM SEED 42
# Phase 3: 50k episode

MapCoDinhHayKhong = False   # True = train trên 1 map cố định; False = map random mỗi episode     #<----------------------------------

# CÓ ĐỊNH LƯU Q TABLE KO
saveQLearningModel = True  # True = lưu checkpoint + best + final; False = không lưu file train

save_dir = "QLearning_Phase3_F"                                                                    #<----------------------------------
if saveQLearningModel:
    os.makedirs(save_dir, exist_ok=True)
    
# CÓ ĐỊNH LOAD LẠI Q TABLE ĐỂ TRAIN TIẾP TỪ Q TABLE ĐÓ KO
loadExistingQTable = True

load_path = "QLearning_Phase2_F/best_qtable.pkl"                                                  #<----------------------------------


# CHỈNH EPISODE TRƯỚC KHI TRAIN!!!!!!!                                                           #<----------------------------------
episodeAmount = 50000
printAfterEpisode = 2000

# Chọn setting Q-learning theo chế độ map (cố định vs random)
if MapCoDinhHayKhong:
    # Map cố định: 50k ep
    agent_QLearning = QLearning(
        maze,
        lrate=0.08,
        discount=0.98,  # Đã test, chọn 0.98 hoặc 0.97
        epsilon=1.0,                      # Lưu ý setting epsilon nếu đang định train model mới, cho nó explore: ep lớn hay muốn exploit: ep nhỏ. Nhất là khi load lại QTable
        epsilon_min=0.05,
        decay_rate=0.99992,   # decay = (ep_min / ep) ** (1 / (episodeAmount * 75%)), để chạm min ở 75% của tổng episode
        defaultQ=0,
        maxSteps=250,
        seed=42
    )
else:
    # Map random: 50k ep
    agent_QLearning = QLearning(
        maze,
        lrate=0.06,
        discount=0.98,
        epsilon=1.0,
        epsilon_min=0.1, # Random map cần 1 khoảng explore nếu gặp trạng thái mới
        decay_rate=0.9999385,
        defaultQ=0,
        maxSteps=300,
        seed=42
    )


# Nếu muốn train tiếp từ Q-table cũ thì load
if loadExistingQTable:
    try:
        loaded_table = loadQTable(load_path, default_q=0.0)  # loader compat numpy._core
        # loadQTable trả về QTable_QLearning, update thẳng vào table đang train
        agent_QLearning.table.update(dict(loaded_table))
        print(f"Loaded Q-table from {os.path.abspath(load_path)} "
              f"({len(loaded_table)} entries).")
    except FileNotFoundError:
        print(f"Không tìm thấy file Q-table tại {load_path}, train từ đầu")


currentEpisode = 0
endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}

episodeScoreResult = []
stepHistory = []
endHistory = []
endFloorHistory = []

best_success = 0
    

for done, score, end_floor in agent_QLearning.play(episodeAmount, fixed_map=MapCoDinhHayKhong):
    typeEnd = maze.end_reason
    if typeEnd is None:
        raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

    episodeScoreResult.append(score)
    stepHistory.append(maze.stepCount)
    endHistory.append(typeEnd)
    endFloorHistory.append(end_floor)

    endType[typeEnd] += 1

    if (currentEpisode + 1) % printAfterEpisode == 0:

        scoreTrungBinh = np.mean(episodeScoreResult[-printAfterEpisode:])
        recentEnds = endHistory[-printAfterEpisode:]
        recentFloors = endFloorHistory[-printAfterEpisode:]
        success_rate = recentEnds.count("goal") / len(recentEnds)
        avg_steps = np.mean(stepHistory[-printAfterEpisode:])
        goal_recent = recentEnds.count("goal")
        death_recent = recentEnds.count("death")
        timeout_recent = recentEnds.count("timeout")
        stagnation_recent = recentEnds.count("stagnation")
        # Thống kê tầng kết thúc: số episode kết thúc ở từng tầng
        floor_counts = {}
        for f in recentFloors:
            floor_counts[f] = floor_counts.get(f, 0) + 1
        floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

        print(
            f"\nEp {currentEpisode+1}/{episodeAmount} | "
            f"avg_reward={scoreTrungBinh:.2f} | "
            f"success_rate={success_rate:.2f} | "
            f"avg_steps={avg_steps:.1f} | "
            f"TỔNG: goal={endType['goal']} death={endType['death']} timeout={endType['timeout']} stagnation={endType['stagnation']} | "
            f"[{printAfterEpisode} ep gần nhất]: goal={goal_recent} death={death_recent} timeout={timeout_recent} stagnation={stagnation_recent} | "
            f"epsilon={agent_QLearning.epsilon:.3f}"
        )
        print(f"Vị trí tầng kết thúc [{printAfterEpisode} ep]: {floor_stat}")
        print("Q-table size:", len(agent_QLearning.table))

        if saveQLearningModel:
            # SAVE CHECKPOINT
            checkpoint_path = os.path.join(
                save_dir, f"qtable_ep{currentEpisode+1}.pkl"
            )
            with open(checkpoint_path, "wb") as f:
                # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                pickle.dump(dict(agent_QLearning.table), f, protocol=4)
            print("Saved checkpoint:", os.path.abspath(checkpoint_path))

            # SAVE BEST MODEL
            if success_rate > best_success:
                best_success = success_rate
                best_path = os.path.join(save_dir, "best_qtable.pkl")
                with open(best_path, "wb") as f:
                    # Dùng protocol=4 để tương thích môi trường Python cũ hơn
                    pickle.dump(dict(agent_QLearning.table), f, protocol=4)
                print("\nNew BEST model saved:", os.path.abspath(best_path))

    currentEpisode += 1


print(f"\nTotal episodes: {episodeAmount}")
print(f"Điểm TB:        {np.mean(episodeScoreResult):.2f}")
print(f"Goals:          {endType['goal']}")
print(f"Deaths:         {endType['death']}")
print(f"Timeout:        {endType['timeout']}")
print(f"Stagnation:     {endType['stagnation']}")


# SAVE FINAL MODEL
if saveQLearningModel:
    final_path = os.path.join(save_dir, "qtable_final.pkl")
    with open(final_path, "wb") as f:
        # Dùng protocol=4 để tương thích môi trường Python cũ hơn
        pickle.dump(dict(agent_QLearning.table), f, protocol=4)
    print("\nFinal model saved:", os.path.abspath(final_path))


# VẼ ĐỒ THỊ
episodes = np.arange(1, len(episodeScoreResult) + 1)
window = 50

plt.figure(figsize=(14, 16))

# RETURN
plt.subplot(4, 1, 1)
plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
if len(episodeScoreResult) >= window:
    smoothed = np.convolve(
        episodeScoreResult,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], smoothed, linewidth=2, label="Moving Avg")
plt.title("Return")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# SUCCESS RATE
plt.subplot(4, 1, 2)
success = np.array([1 if e == "goal" else 0 for e in endHistory])
if len(success) >= window:
    success_rate = np.convolve(
        success,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], success_rate)
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Goal Rate")
plt.ylim(0, 1)
plt.grid(True)

# END FLOOR RATE
plt.subplot(4, 1, 3)
end_floors = np.array(endFloorHistory, dtype=int)
max_floor = int(TOTAL_FLOORS) - 1
rates = []
for f in range(max_floor + 1):
    is_floor_f = (end_floors == f).astype(float)
    if len(is_floor_f) >= window:
        floor_rate = np.convolve(
            is_floor_f,
            np.ones(window) / window,
            mode="valid"
        )
        rates.append(floor_rate)
    else:
        rates.append(is_floor_f)
# Stacked area
if len(end_floors) >= window:
    x = episodes[window - 1:]
else:
    x = episodes
plt.stackplot(
    x,
    rates,
    labels=[f"End at floor {f}" for f in range(max_floor + 1)],
    alpha=0.8
)
plt.title("End Floor Rate")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

# EPISODE LENGTH
plt.subplot(4, 1, 4)
plt.plot(episodes, stepHistory, alpha=0.4)
if len(stepHistory) >= window:
    step_smooth = np.convolve(
        stepHistory,
        np.ones(window)/window,
        mode="valid"
    )
    plt.plot(episodes[window-1:], step_smooth, linewidth=2)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== Cell 35: 8.5. Evaluate QLearning với Decayed Epsilon Greedy =====
# Chạy evaluate
eval_results = evaluate_qtable_on_ui(
    # load path None = Model lúc chưa train
    # load path != None = Model đã train
    
    load_path="FinalModel/QL2/best_qtable.pkl",  # đường dẫn Q-table
    episodes=1000,              # số episode test
    fixed_map=True,          # True: giữ 1 layout map; False: random map mỗi episode
    sleep_time=0.05,          # delay giữa các step để mắt kịp nhìn UI
    seed=42,                  # seed chung cho evaluate chung cho RNG; truyền None nếu muốn mỗi lần evaluate ra chuỗi random khác nhau
    use_ui = False
)

print("\nChi tiết từng episode:")
for i, (total_r, n_steps, end_floor, end_type) in enumerate(eval_results, start=1):
    print(
        f"  Ep {i:03d} | reward={total_r:8.2f} | steps={n_steps:4d} | "
        f"end_floor={end_floor} | end_type={end_type}"
    )

if len(eval_results) > 0:
    rewards = np.array([r for r, _, _, _ in eval_results], dtype=float)
    steps_a = np.array([s for _, s, _, _ in eval_results], dtype=float)
    ends = [t for _, _, _, t in eval_results]
    print(
        f"\nTổng quan | avg_reward={rewards.mean():.2f} | avg_steps={steps_a.mean():.1f} | "
        f"goal={ends.count('goal')} death={ends.count('death')} "
        f"timeout={ends.count('timeout')} stagnation={ends.count('stagnation')}"
    )
