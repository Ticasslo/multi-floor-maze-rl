"""Double DQN (PyTorch): state processor, replay buffer, multi-branch MLP, the agent, train/evaluate helpers, training setup and the evaluation run.

Copied cell by cell from the Zeppelin note notebooks/MultiFloorMaze_Zeppelin.json so the code can be read on GitHub.
Only the "%pyspark" line at the top of each cell is removed.
Some helpers call Zeppelin's z object for the game UI, so run the code in the notebook (or the Colab notebook for Double DQN).
"""

# ===== Cell 37: 9.1. Thư viện cho việc train Double DQL =====
# DQN
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

import torch
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("cuda device count:", torch.cuda.device_count())
  print("current device:", torch.cuda.current_device())
  print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
  print("cuda runtime version:", torch.version.cuda)
  print("capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
  print("Tensor device:", torch.tensor([1.0], device="cuda" if torch.cuda.is_available() else "cpu").device)
  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ===== Cell 38 =====
# Số action trong môi trường (đúng với MazeEnv.getPosActions: 0 -> 8)
ACTION_COUNT = 9
MOVE_ACTIONS = [0, 1, 2, 3]  # Up, Right, Down, Left
HAMMER_ACTIONS = [4, 5, 6, 7]  # Hammer Up, Right, Down, Left
WAIT_ACTION = 8 # Wait


# ===== Cell 39: 9.2. Class DQNStateProcessor và ReplayBuffer =====
class DQNStateProcessor:
    # Quy ước index cho state vector 19 chiều (khớp MazeEnv.getState(Agent))
    STATE_DIM = 19  # Tổng số feature trong state
    FLOOR_IDX = 0  # Tầng hiện tại
    HP_STATE_IDX = 1  # HP rời rạc: 0 (nguy hiểm), 1 (ổn)
    KEY_IDX = 2  # Có chìa khóa hay chưa (0/1)
    SHIELD_IDX = 3  # Có khiên hay chưa (0/1)
    UP1_IDX = 4  # Loại ô phía trên 1 bước
    DOWN1_IDX = 5  # Loại ô phía dưới 1 bước
    LEFT1_IDX = 6  # Loại ô phía trái 1 bước
    RIGHT1_IDX = 7  # Loại ô phía phải 1 bước
    LOCAL_START_IDX = 4  # Bắt đầu block local 12 ô
    LOCAL_END_EXCLUSIVE_IDX = 16  # Kết thúc block local (exclusive)
    MONSTER_CELL_IDX = 16  # Quái nằm ở ô nào quanh agent (0/1..12/13)
    MONSTER_STUN_FLAG_IDX = 17  # Cờ quái đang stun (0/1)
    HAMMER_READY_IDX = 18  # Búa sẵn sàng dùng (0/1)
    INTERACTION_IDXS = [HP_STATE_IDX, KEY_IDX, SHIELD_IDX, HAMMER_READY_IDX]
    CONTEXT_IDXS = [FLOOR_IDX, MONSTER_CELL_IDX, MONSTER_STUN_FLAG_IDX]
    CELL_TYPE_COUNT = 13  # Cell id trong env

    @staticmethod
    def encode(state):
        # Encode state tuple (từ env.getState(Agent())) thành vector float32 1 chiều
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] != DQNStateProcessor.STATE_DIM:
            raise ValueError(
                f"DQNStateProcessor.encode: state vector phải có {DQNStateProcessor.STATE_DIM} phần tử, nhưng có {arr.shape[0]}"
            )
        return arr

    @staticmethod
    def split_state_vec(state_vec):
        # Split vector 19 chiều -> local(12), interaction(4), context(3)
        vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != DQNStateProcessor.STATE_DIM:
            raise ValueError(f"split_state_vec: expected state dim {DQNStateProcessor.STATE_DIM}, got {vec.shape[0]}")
        local = vec[DQNStateProcessor.LOCAL_START_IDX:DQNStateProcessor.LOCAL_END_EXCLUSIVE_IDX]
        
        # One-hot hóa 12 ô local (mỗi ô 13 loại)
        local_idx = np.clip(np.rint(local).astype(np.int64), 0, DQNStateProcessor.CELL_TYPE_COUNT - 1)
        local_onehot = np.eye(DQNStateProcessor.CELL_TYPE_COUNT, dtype=np.float32)[local_idx].reshape(-1)
        interaction = vec[DQNStateProcessor.INTERACTION_IDXS]
        context = vec[DQNStateProcessor.CONTEXT_IDXS]
        return local_onehot, interaction, context

    @staticmethod
    def split_state_batch_torch(states_batch_t):
        # Split batch (batch,19) -> local(batch,12), interaction(batch,4), context(batch,3)
        # Dùng torch indexing trực tiếp để tránh convert numpy và tránh lỗi khi tensor ở CUDA
        # local: idx 4..15 (12 phần tử), interaction/context dùng index constants của DQNStateProcessor
        local = states_batch_t[:, DQNStateProcessor.LOCAL_START_IDX:DQNStateProcessor.LOCAL_END_EXCLUSIVE_IDX]
        local_idx = torch.clamp(torch.round(local), 0, DQNStateProcessor.CELL_TYPE_COUNT - 1).long()
        local_onehot = F.one_hot(local_idx, num_classes=DQNStateProcessor.CELL_TYPE_COUNT).float()
        local_onehot = local_onehot.view(local_onehot.shape[0], -1)
        interaction = states_batch_t[:, DQNStateProcessor.INTERACTION_IDXS]
        context = states_batch_t[:, DQNStateProcessor.CONTEXT_IDXS]
        
        return local_onehot, interaction, context

    @staticmethod
    def make_action_mask_batch(states_batch):
        # Tạo mask hợp lệ (boolean) cho từng action trong action space 0..8
        # Output: (batch, ACTION_COUNT) với True = action hợp lệ
        states_batch = np.asarray(states_batch, dtype=np.float32)

        key = states_batch[:, DQNStateProcessor.KEY_IDX]  # 0/1
        up1 = states_batch[:, DQNStateProcessor.UP1_IDX]
        down1 = states_batch[:, DQNStateProcessor.DOWN1_IDX]
        left1 = states_batch[:, DQNStateProcessor.LEFT1_IDX]
        right1 = states_batch[:, DQNStateProcessor.RIGHT1_IDX]
        hammer_ready = states_batch[:, DQNStateProcessor.HAMMER_READY_IDX]

        def is_blocked(cell):
            blocked_by_wall = (cell == WALL) | (cell == SOFT_WALL)
            blocked_by_door = (cell == DOOR) & (key == 0)
            return blocked_by_wall | blocked_by_door

        valid_up = ~is_blocked(up1)
        valid_down = ~is_blocked(down1)
        valid_left = ~is_blocked(left1)
        valid_right = ~is_blocked(right1)

        valid_wait = np.ones_like(valid_up, dtype=bool)
        valid_hammer = (hammer_ready == 1)

        mask = np.zeros((states_batch.shape[0], ACTION_COUNT), dtype=bool)
        mask[:, 0] = valid_up
        mask[:, 1] = valid_right
        mask[:, 2] = valid_down
        mask[:, 3] = valid_left
        mask[:, WAIT_ACTION] = valid_wait
        mask[:, 4:8] = valid_hammer[:, np.newaxis]

        # Nếu mask trống, ép WAIT luôn hợp lệ để tránh argmax lỗi
        no_action = ~mask.any(axis=1)
        if np.any(no_action):
            mask[no_action, WAIT_ACTION] = True

        return mask


class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        # ReplayBuffer lưu lại nhiều trải nghiệm (transitions) để train ổn định hơn
        # DQN sẽ sample ngẫu nhiên một batch thay vì train trực tiếp chuỗi liên tiếp
        # Điều này giảm tương quan mẫu và giúp học mượt hơn
        self.capacity = int(capacity)
        # deque(maxlen=capacity) tự loại phần tử cũ nhất khi vượt capacity
        self.buffer = deque(maxlen=self.capacity)
        # RNG độc lập cho lấy mẫu (seed để test/so sánh)
        self.randGen = np.random.default_rng(seed)

    def __len__(self):
        # Trả về số transition đang có trong buffer
        return len(self.buffer)

    def add(self, state_vec, action, reward, next_state_vec, done):
        # Lưu 1 transition vào buffer:
        # (s_t, a_t, r_t, s_{t+1}, done_t)
        #
        # Trong đó:
        # state_vec, next_state_vec: vector float32 (cùng độ dài)
        # action: int
        # reward: float
        # done: 1.0 nếu episode kết thúc, 0.0 nếu chưa kết thúc
        #
        # Việc ép dtype ngay khi add giúp train_step ít lỗi kiểu dữ liệu
        self.buffer.append(
            (
                # Vector hóa state để network nhận input số
                np.asarray(state_vec, dtype=np.float32),
                int(action),
                float(reward),
                # Vector hóa next_state cho target network
                np.asarray(next_state_vec, dtype=np.float32),
                # done để tính target: r + gamma * maxQ(...) * (1 - done)
                float(done),
            )
        )

    def sample(self, batch_size):
        # Lấy ngẫu nhiên batch_size transitions từ buffer (uniform sampling)
        # Trả về dạng numpy để tính toán vector hóa ở train_step
        if batch_size <= 0:
            raise ValueError("batch_size phải > 0")
        if len(self.buffer) < batch_size:
            raise ValueError("ReplayBuffer chưa đủ phần tử để sample")

        # Uniform sampling không lặp trong cùng minibatch (replace=False)
        idxs = self.randGen.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]

        # Tách từng thành phần trong transition thành mảng theo cột
        states = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        actions = np.asarray([b[1] for b in batch], dtype=np.int64)
        rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch], axis=0).astype(np.float32)
        # dones dùng kiểu float để phù hợp phép nhân trong target
        dones = np.asarray([b[4] for b in batch], dtype=np.float32)

        # Shapes kỳ vọng:
        # states: (batch, state_dim)
        # actions: (batch,)
        # rewards: (batch,)
        # next_states: (batch, state_dim)
        # dones: (batch,)
        return states, actions, rewards, next_states, dones


# ===== Cell 40: 9.3. Mạng nơron QNetworkMLP =====
class QNetworkMLP(nn.Module):
    # Mạng nơ-ron xấp xỉ Q(s, a):
    # - input: 3 tensor đã tách từ state vector 19 chiều:
    #   + local (12), interaction (4), context (3)
    # - output: Q-value cho 9 action (0 tới 8)

    # Kiến trúc nhiều nhánh:
    # - local: 12 dims -> fc_local -> hidden
    # - interaction: 4 dims -> fc_interaction -> hidden
    # - context: 3 dims -> fc_context -> hidden
    # concat 3 hidden -> fc_out -> Q(action)
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        if int(state_dim) != DQNStateProcessor.STATE_DIM:
            raise ValueError(f"QNetworkMLP: state_dim phải là {DQNStateProcessor.STATE_DIM}, nhưng nhận {state_dim}")

        self.local_dim = 12 * DQNStateProcessor.CELL_TYPE_COUNT
        self.interaction_dim = 4
        self.context_dim = 3

        self.fc_local = nn.Linear(self.local_dim, hidden_dim)
        self.fc_interaction = nn.Linear(self.interaction_dim, hidden_dim)
        self.fc_context = nn.Linear(self.context_dim, hidden_dim)
        self.fc_out = nn.Linear(3 * hidden_dim, action_dim)

        nn.init.kaiming_normal_(self.fc_local.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_local.bias)
        nn.init.kaiming_normal_(self.fc_interaction.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_interaction.bias)
        nn.init.kaiming_normal_(self.fc_context.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc_context.bias)
        nn.init.kaiming_normal_(self.fc_out.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, local, interaction, context):
        h_local = F.relu(self.fc_local(local))
        h_interaction = F.relu(self.fc_interaction(interaction))
        h_context = F.relu(self.fc_context(context))
        h = torch.cat([h_local, h_interaction, h_context], dim=1)
        return self.fc_out(h)


# ===== Cell 41: 9.4. Định nghĩa class DQNAgent - Double DQL =====
class DQNAgent:
    # Agent DQN gồm:
    # - online network: dự đoán Q(s,a) để chọn action và tính loss
    # - target network: dùng để tạo target ổn định hơn
    # - replay buffer: lưu transition và sample batch ngẫu nhiên
    # - action masking: chỉ xét action hợp lệ khi tính argmax/max_next
    # DQN (có replay + target network + mask hợp lệ)
    def __init__(
        self,
        state_dim,
        action_dim=ACTION_COUNT,
        gamma=0.98,
        lr=1e-4,
        hidden_dim=256,
        buffer_capacity=200000,
        batch_size=128,
        tau=0.005,
        seed=42,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99995,
        huber_delta=1.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.huber_delta = float(huber_delta)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # H2D async khi CUDA
        self._non_blocking = self.device.type == "cuda"
        self.randGen = np.random.default_rng(seed)

        self.replay = ReplayBuffer(buffer_capacity, seed=seed)

        self.online_net = QNetworkMLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = QNetworkMLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optim = optim.Adam(self.online_net.parameters(), lr=lr)
        self.train_steps = 0

    def decay_epsilon_step(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state_vec):
        # Epsilon-greedy + action mask hợp lệ theo state vector
        valid_mask = DQNStateProcessor.make_action_mask_batch(np.asarray(state_vec, dtype=np.float32)[None, :])[0]
        valid_actions = [int(a) for a in range(self.action_dim) if bool(valid_mask[a])]
        if not valid_actions:
            valid_actions = [WAIT_ACTION]

        # Epsilon greedy trong các action hợp lệ
        if self.randGen.random() < self.epsilon:
            return int(self.randGen.choice(valid_actions))

        local_np, inter_np, ctx_np = DQNStateProcessor.split_state_vec(state_vec)
        local_t = torch.from_numpy(np.ascontiguousarray(local_np, dtype=np.float32)).unsqueeze(0).to(
            self.device, non_blocking=self._non_blocking
        )
        inter_t = torch.from_numpy(np.ascontiguousarray(inter_np, dtype=np.float32)).unsqueeze(0).to(
            self.device, non_blocking=self._non_blocking
        )
        ctx_t = torch.from_numpy(np.ascontiguousarray(ctx_np, dtype=np.float32)).unsqueeze(0).to(
            self.device, non_blocking=self._non_blocking
        )
        with torch.inference_mode():
            q_values = self.online_net(local_t, inter_t, ctx_t)[0]  # (action_dim,)

        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
        q_values_masked = q_values.masked_fill(~mask_t, -1e9)
        best_q = torch.max(q_values_masked).item()
        best_actions = [a for a in valid_actions if abs(q_values_masked[a].item() - best_q) < 1e-8]
        return int(self.randGen.choice(best_actions))

    def push_transition(self, s_vec, a, r, s_next_vec, done):
        self.replay.add(s_vec, a, r, s_next_vec, done)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def learn_step(self):
        # 1 step cập nhật mạng online từ một minibatch replay
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        nb = self._non_blocking
        states_t = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(
            self.device, non_blocking=nb
        )
        actions_t = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64)).to(
            self.device, non_blocking=nb
        )
        rewards_t = torch.from_numpy(np.ascontiguousarray(rewards, dtype=np.float32)).to(
            self.device, non_blocking=nb
        )
        next_states_t = torch.from_numpy(np.ascontiguousarray(next_states, dtype=np.float32)).to(
            self.device, non_blocking=nb
        )
        dones_t = torch.from_numpy(np.ascontiguousarray(dones, dtype=np.float32)).to(
            self.device, non_blocking=nb
        )

        next_mask = DQNStateProcessor.make_action_mask_batch(next_states)  # numpy bool: (batch, action_dim)
        next_mask_t = torch.as_tensor(np.ascontiguousarray(next_mask), dtype=torch.bool, device=self.device)

        # Tách state vector -> 3 tensor trước khi đưa vào network
        local_t, inter_t, ctx_t = DQNStateProcessor.split_state_batch_torch(states_t)

        # Q_pred(s,a)
        q_pred = self.online_net(local_t, inter_t, ctx_t)  # (batch, action_dim)
        q_sa = q_pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # Q_target(s',a') - cập nhật theo công thức của Double DQL
        with torch.no_grad():
            next_local_t, next_inter_t, next_ctx_t = DQNStateProcessor.split_state_batch_torch(next_states_t)
            # 1) Online net chọn action tốt nhất ở s'
            q_next_online = self.online_net(next_local_t, next_inter_t, next_ctx_t)  # (batch, action_dim)
            q_next_online_masked = q_next_online.masked_fill(~next_mask_t, -1e9)
            next_actions = q_next_online_masked.argmax(dim=1, keepdim=True)  # (batch,1)
            # 2) Target net đánh giá giá trị của action đã chọn
            q_next_target = self.target_net(next_local_t, next_inter_t, next_ctx_t)  # (batch, action_dim)
            max_next_q = q_next_target.gather(1, next_actions).squeeze(1)  # (batch,)

            target_q = rewards_t + self.gamma * (1.0 - dones_t) * max_next_q

        loss = F.smooth_l1_loss(q_sa, target_q, beta=self.huber_delta)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping để ổn định
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optim.step()

        self.train_steps += 1
        # Soft update: target = tau * online + (1 - tau) * target
        with torch.no_grad():
            for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                t_param.data.copy_(self.tau * o_param.data + (1.0 - self.tau) * t_param.data)

        return float(loss.item())


# ===== Cell 42: 9.5. Các hàm trợ giúp train, evaluate Double DQL =====
def loadDQNCheckpoint(
    dqn_agent,
    load_path,
    load_map_location="cpu",  # 'cpu' (mặc định), hoặc 'cuda'/'cuda:0' nếu muốn nạp lên GPU
    epsilon_override=None,
    verbose=True,
):
    # Load checkpoint từ file .pt và nạp:
    # - model (online + target)
    # - optimizer (nếu checkpoint lưu)
    # - epsilon/train_steps (nếu checkpoint lưu)
    if load_path is None:
        return None

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Không thấy file load_path: {load_path}")

    # map_location quyết định tensor trong checkpoint sẽ được nạp về thiết bị nào.
    # Ví dụ: 'cpu' -> chạy CPU dù file từng save trên GPU; 'cuda'/'cuda:0' -> nạp thẳng lên GPU.
    checkpoint_obj = torch.load(load_path, map_location=load_map_location)

    state_dict = None
    target_state_dict = None
    optimizer_state_dict = None
    ckpt_epsilon = None
    ckpt_train_steps = None

    if isinstance(checkpoint_obj, dict) and (
        "model_state_dict" in checkpoint_obj
        or "state_dict" in checkpoint_obj
        or "optimizer_state_dict" in checkpoint_obj
    ):
        if "model_state_dict" in checkpoint_obj:
            state_dict = checkpoint_obj["model_state_dict"]
        elif "state_dict" in checkpoint_obj:
            state_dict = checkpoint_obj["state_dict"]

        target_state_dict = checkpoint_obj.get("target_state_dict", None)
        optimizer_state_dict = checkpoint_obj.get("optimizer_state_dict", None)
        ckpt_epsilon = checkpoint_obj.get("epsilon", None)
        ckpt_train_steps = checkpoint_obj.get("train_steps", None)
    else:
        # checkpoint cũ: raw state_dict
        state_dict = checkpoint_obj

    dqn_agent.online_net.load_state_dict(state_dict)
    if target_state_dict is not None:
        dqn_agent.target_net.load_state_dict(target_state_dict)
    else:
        dqn_agent.target_net.load_state_dict(state_dict)

    if optimizer_state_dict is not None:
        dqn_agent.optim.load_state_dict(optimizer_state_dict)

    if epsilon_override is not None:
        dqn_agent.epsilon = float(epsilon_override)
    elif ckpt_epsilon is not None:
        dqn_agent.epsilon = float(ckpt_epsilon)

    if ckpt_train_steps is not None:
        dqn_agent.train_steps = int(ckpt_train_steps)

    if verbose:
        print(f"[DQN] Loaded checkpoint từ: {load_path}")

    return state_dict


def train_dqn_on_env(
    dqn_agent,
    maze,
    episodeAmount=50000,
    fixed_map=True,
    seed=42,
    maxSteps=None,
    printAfterEpisode=2000,
    saveDQNModel=False,
    save_dir="DQN_Phase1",
    plot_metrics=True,
    window=50,
    load_path=None,
    load_map_location="cpu",  # 'cpu' / 'cuda' / 'cuda:0' cho torch.load() checkpoint
    epsilon_override=None,  # resume thì có thể set lại epsilon để "reheat" exploration
    learn_every=1,
):
    # Train DQN theo đúng env hiện có (MazeEnv).
    # Mỗi bước:
    # - lấy state từ maze.getState(Agent())
    # - encode state -> vector float
    # - chọn action bằng epsilon-greedy + action mask
    # - chạy maze.step(action) để lấy reward/next_state/done
    # - đẩy transition vào replay buffer
    # - (mỗi learn_every bước) gọi learn_step() cập nhật mạng
    #
    # Cuối mỗi printAfterEpisode sẽ log:
    # - success_rate (goal)
    # - avg_steps
    # - số death/timeout/stagnation
    
    # Train DQN:
    # - state lấy từ maze.getState(player_agent)
    # - action chọn theo epsilon-greedy + action mask
    # - mỗi bước lưu transition vào replay buffer rồi learn_step
    #
    # Metric log:
    # - success_rate: tỉ lệ episode kết thúc tại GOAL
    # - avg_steps, goal/death/timeout/stagnation counts

    if maxSteps is not None:
        maze.maxSteps = maxSteps

    if saveDQNModel:
        os.makedirs(save_dir, exist_ok=True)

    # Load weights để train tiếp từ checkpoint .pt
    # Khi resume/train lại từ checkpoint (`load_path`):
    # - BẮT BUỘC khớp checkpoint: kiến trúc mạng (state_dim, hidden_dim) và action_dim/action space.
    #   Nếu khác shape thì `load_state_dict` sẽ lỗi.
    # - CÓ THỂ thay đổi: episodeAmount/fixed_map/seed/maxSteps/printAfterEpisode, và các hyperparam như gamma/batch_size/buffer_capacity/tau (replay state không được load lại).
    # - Lưu ý lr: do checkpoint có optimizer_state_dict, lr trong optimizer thường được restore từ checkpoint.
    # - epsilon: hoặc giữ epsilon đang có trong `dqn_agent`, hoặc set lại bằng `epsilon_override`
    loadDQNCheckpoint(
        dqn_agent=dqn_agent,
        load_path=load_path,
        load_map_location=load_map_location,
        epsilon_override=epsilon_override,
        verbose=True,
    )
    
    _learn_every = max(1, int(learn_every))

    player_agent = Agent()
    rng = np.random.default_rng(seed)

    endType = {"goal": 0, "death": 0, "timeout": 0, "stagnation": 0}
    episodeScoreResult = []
    stepHistory = []
    endHistory = []
    endFloorHistory = []

    best_success = 0.0

    fixedMapRandom1Time = False
    total_env_steps = 0

    for ep in range(episodeAmount):
        # Reset môi trường theo chế độ fixed_map/random_map
        if fixed_map:
            # Fixed map:
            # - lần đầu: randomize map theo env_seed
            # - lần sau: giữ layout, chỉ reset trạng thái động
            if not fixedMapRandom1Time:
                env_seed = int(rng.integers(1_000_000_000))
                maze.reset(seed=env_seed, randomize_map=True)
                fixedMapRandom1Time = True
            else:
                maze.reset(seed=None, randomize_map=False)
        else:
            # Random map:
            env_seed = int(rng.integers(1_000_000_000))
            maze.reset(seed=env_seed, randomize_map=True)

        s_raw = maze.getState(player_agent)
        s_vec = DQNStateProcessor.encode(s_raw)

        done = False
        score = 0.0

        while not done:
            # Lấy danh sách actions hợp lệ theo env tại thời điểm hiện tại
            # (để đảm bảo luôn khớp logic MazeEnv)
            env_actions = maze.getPosActions()
            if not env_actions:
                env_actions = MOVE_ACTIONS + [WAIT_ACTION]

            a = dqn_agent.select_action(s_vec)
            # Safety: nếu action chọn ra không hợp lệ (hiếm), ép WAIT
            if a not in env_actions:
                a = WAIT_ACTION

            _, r, done = maze.step(a)
            score += float(r)

            s_next_raw = maze.getState(player_agent)
            s_next_vec = DQNStateProcessor.encode(s_next_raw)

            dqn_agent.push_transition(s_vec, a, float(r), s_next_vec, 1.0 if done else 0.0)
            total_env_steps += 1
            if _learn_every <= 1 or (total_env_steps % _learn_every == 0):
                dqn_agent.learn_step()

            s_vec = s_next_vec

        # Xác định loại kết thúc episode (ưu tiên end_reason từ env).
        typeEnd = maze.end_reason
        if typeEnd is None:
            raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

        endType[typeEnd] += 1
        episodeScoreResult.append(score)
        stepHistory.append(maze.stepCount)
        endHistory.append(typeEnd)
        endFloorHistory.append(int(maze.floor))

        # Decay epsilon sau mỗi episode
        dqn_agent.decay_epsilon_step()

        # Log theo mốc
        if (ep + 1) % printAfterEpisode == 0:
            scoreTrungBinh = float(np.mean(episodeScoreResult[-printAfterEpisode:]))

            recentEnds = endHistory[-printAfterEpisode:]
            recentFloors = endFloorHistory[-printAfterEpisode:]
            success_rate = recentEnds.count("goal") / len(recentEnds)
            avg_steps = float(np.mean(stepHistory[-printAfterEpisode:]))

            goal_recent = recentEnds.count("goal")
            death_recent = recentEnds.count("death")
            timeout_recent = recentEnds.count("timeout")
            stagnation_recent = recentEnds.count("stagnation")

            floor_counts = {}
            for f in recentFloors:
                floor_counts[f] = floor_counts.get(f, 0) + 1
            floor_stat = " ".join(f"tầng{k}={v}" for k, v in sorted(floor_counts.items()))

            print(
                f"\nEp {ep+1}/{episodeAmount} | "
                f"avg_reward={scoreTrungBinh:.2f} | "
                f"success_rate={success_rate:.2f} | "
                f"avg_steps={avg_steps:.1f} | "
                f"TỔNG: goal={endType['goal']} death={endType['death']} timeout={endType['timeout']} stagnation={endType['stagnation']} | "
                f"[{printAfterEpisode} ep gần nhất]: goal={goal_recent} death={death_recent} timeout={timeout_recent} stagnation={stagnation_recent} | "
                f"epsilon={dqn_agent.epsilon:.3f}"
            )
            print(f"Vị trí tầng kết thúc [{printAfterEpisode} ep]: {floor_stat}")

            # SAVE BEST / CHECKPOINT nếu bật
            if saveDQNModel:
                checkpoint_path = os.path.join(save_dir, f"dqn_ep{ep+1}.pt")
                checkpoint = {
                    "model_state_dict": dqn_agent.online_net.state_dict(),
                    "target_state_dict": dqn_agent.target_net.state_dict(),
                    "optimizer_state_dict": dqn_agent.optim.state_dict(),
                    "epsilon": dqn_agent.epsilon,
                    "train_steps": dqn_agent.train_steps,
                }
                torch.save(checkpoint, checkpoint_path)

                if success_rate > best_success:
                    best_success = success_rate
                    best_path = os.path.join(save_dir, "best_dqn.pt")
                    torch.save(checkpoint, best_path)

    print(f"\nDQN Train xong. Total episodes: {episodeAmount}")
    print(f"Điểm TB: {float(np.mean(episodeScoreResult)):.2f}")
    print(
        f"Goals: {endType['goal']}  Deaths: {endType['death']}  "
        f"Timeout: {endType['timeout']}  Stagnation: {endType['stagnation']}"
    )

    if plot_metrics:
        episodes = np.arange(1, len(episodeScoreResult) + 1)

        plt.figure(figsize=(14, 16))

        # RETURN
        plt.subplot(4, 1, 1)
        plt.plot(episodes, episodeScoreResult, alpha=0.3, label="Return")
        if len(episodeScoreResult) >= window:
            smoothed = np.convolve(
                episodeScoreResult,
                np.ones(window) / window,
                mode="valid",
            )
            plt.plot(episodes[window - 1 :], smoothed, linewidth=2, label="Moving Avg")
        plt.title("Return (DQN)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

        # SUCCESS RATE
        plt.subplot(4, 1, 2)
        success = np.array([1 if e == "goal" else 0 for e in endHistory], dtype=float)
        if len(success) >= window:
            success_rate = np.convolve(success, np.ones(window) / window, mode="valid")
            plt.plot(episodes[window - 1 :], success_rate)
        else:
            plt.plot(episodes, success)
        plt.title("Success Rate (DQN)")
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
                floor_rate = np.convolve(is_floor_f, np.ones(window) / window, mode="valid")
                rates.append(floor_rate)
            else:
                rates.append(is_floor_f)

        if len(end_floors) >= window:
            x = episodes[window - 1 :]
        else:
            x = episodes

        plt.stackplot(
            x,
            rates,
            labels=[f"End at floor {f}" for f in range(max_floor + 1)],
            alpha=0.8,
        )
        plt.title("End Floor Rate (DQN)")
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
                np.ones(window) / window,
                mode="valid",
            )
            plt.plot(episodes[window - 1 :], step_smooth, linewidth=2)
        plt.title("Episode Length (DQN)")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return dqn_agent, {
        "episodeScoreResult": episodeScoreResult,
        "stepHistory": stepHistory,
        "endHistory": endHistory,
        "endFloorHistory": endFloorHistory,
        "endType": endType,
    }


def evaluate_dqn_on_env(
    dqn_agent,
    maze,
    episodes=10,
    fixed_map=True,
    seed=42,
    sleep_time=0.1,
    use_ui=False,
    load_path=None,
    load_map_location="cpu",  # torch.load(): 'cpu' (mặc định) hoặc 'cuda'/'cuda:0' để nạp lên GPU
):
    # Evaluate greedy (epsilon = 0) trên môi trường.
    # Mục tiêu: chạy policy greedy để kiểm tra DQN sau train
    # Không cập nhật trọng số (không học), chỉ quan sát performance
    player_agent = Agent()
    rng = np.random.default_rng(seed)

    # Nếu truyền load_path thì load weights vào mạng trước khi evaluate
    if load_path is not None:
        loadDQNCheckpoint(
            dqn_agent=dqn_agent,
            load_path=load_path,
            load_map_location=load_map_location,
            epsilon_override=None,
            verbose=True,
        )

    old_epsilon = dqn_agent.epsilon
    dqn_agent.epsilon = 0.0

    results = []
    fixedMapRandom1Time = False

    try:
        for ep in range(episodes):
            if fixed_map:
                if not fixedMapRandom1Time:
                    env_seed = int(rng.integers(1_000_000_000))
                    maze.reset(seed=env_seed, randomize_map=True)
                    fixedMapRandom1Time = True
                else:
                    maze.reset(seed=None, randomize_map=False)
            else:
                env_seed = int(rng.integers(1_000_000_000))
                maze.reset(seed=env_seed, randomize_map=True)

            # Reset UI status (nếu bật UI)
            if use_ui:
                z.z.angularBind('gameStatus', 'Đang chơi')
                _updateUI()

            s_raw = maze.getState(player_agent)
            s_vec = DQNStateProcessor.encode(s_raw)

            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                env_actions = maze.getPosActions()
                if not env_actions:
                    env_actions = MOVE_ACTIONS + [WAIT_ACTION]

                a = dqn_agent.select_action(s_vec)
                if a not in env_actions:
                    a = WAIT_ACTION

                _, r, done = maze.step(a)
                total_reward += float(r)

                steps += 1
                s_next_raw = maze.getState(player_agent)
                s_vec = DQNStateProcessor.encode(s_next_raw)

                # Cập nhật UI mỗi bước (nếu bật UI)
                if use_ui:
                    _updateUI()
                    time.sleep(sleep_time)

            # Phân loại kết thúc
            end_type = maze.end_reason
            if end_type is None:
                raise RuntimeError("Episode ended nhưng maze.end_reason chưa được set")

            results.append((float(total_reward), steps, int(maze.floor), end_type))
    finally:
        # Dù evaluate lỗi giữa chừng vẫn khôi phục trạng thái agent/UI.
        if use_ui:
            _setGameStatus()
        dqn_agent.epsilon = old_epsilon

    return results

def _print_dqn_eval(tag, results):
    print(f"\n{tag} - Chi tiết từng episode:")
    for ep_idx, (total_reward, steps, end_floor, end_type) in enumerate(results, start=1):
        print(
            f"  Ep {ep_idx:03d} | reward={total_reward:8.2f} | "
            f"steps={steps:3d} | end_floor={end_floor} | end_type={end_type}"
        )

    if len(results) > 0:
        rewards = np.array([r for r, _, _, _ in results], dtype=float)
        steps_arr = np.array([s for _, s, _, _ in results], dtype=float)
        end_types = [t for _, _, _, t in results]
        print(
            f"{tag} - Tổng quan | avg_reward={rewards.mean():.2f} | "
            f"avg_steps={steps_arr.mean():.1f} | "
            f"goal={end_types.count('goal')} death={end_types.count('death')} "
            f"timeout={end_types.count('timeout')} stagnation={end_types.count('stagnation')}"
        )


# ===== Cell 43: 9.6.1. Train Double DQL với Decayed Epsilon Greedy =====
# CÁC TRAINING PHASE ĐƯỢC TRAIN TRÊN GG COLLAB VỚI LINK: https://colab.research.google.com/drive/1juScovKKG9k0vtVOJIrnX10Bp0okmBMw?usp=sharing 

# PHASE 1: 20k ep / MAP cố định seed 42, SOFT_WALL=5, MAX_BOMB=2, TOTAL_FLOORS=1, maxSteps = 100
# PHASE 2: 20k ep / MAP cố định seed 42, SOFT_WALL=5, MAX_BOMB=2, TOTAL_FLOORS=2, maxSteps = 200
# PHASE 3: 20k ep / MAP cố định seed 42, SOFT_WALL=5, MAX_BOMB=2, TOTAL_FLOORS=3, maxSteps = 250
# PHASE 4: 20k ep / MAP random seed 42, SOFT_WALL=5, MAX_BOMB=2, TOTAL_FLOORS=3, maxSteps = 250

DQN_EP_AMOUNT = 20000
DQN_EP_MIN = 0.1

dqn_agent = DQNAgent(
    state_dim=DQNStateProcessor.STATE_DIM,  # BẮT BUỘC khớp shape state hiện tại của env
    gamma=0.98,            # Discount factor (ưu tiên reward tương lai)
    lr=3e-4,               # Learning rate của optimizer
    hidden_dim=128,        # Số neuron mỗi nhánh MLP
    buffer_capacity=200000,# Replay buffer lớn -> mẫu đa dạng hơn nhưng tốn RAM hơn
    batch_size=64,        # Batch lớn ổn định hơn nhưng chậm/đòi GPU-RAM hơn
    tau=0.005,  # Soft update: nhỏ -> target ổn định hơn, lớn -> bám online nhanh hơn

    # epsilon_start chỉ có hiệu lực khi train mới (không load checkpoint)
    # Nếu có load_path thì epsilon có thể bị ghi đè bởi checkpoint và/hoặc epsilon_override bên dưới
    epsilon_start=1.0,
    epsilon_min=DQN_EP_MIN,
    epsilon_decay=float(np.power(DQN_EP_MIN, 1.0 / int(DQN_EP_AMOUNT * 0.75))),
)

dqn_agent, logs = train_dqn_on_env(
    dqn_agent=dqn_agent,
    maze=maze,
    episodeAmount=DQN_EP_AMOUNT,
    fixed_map=True,        # True: 1 layout map cố định (ổn định benchmark); False: random map mỗi ep
    seed=42,               # Seed cho chuỗi env_seed (để tái lập kết quả)
    maxSteps=250,          # Giới hạn bước/episode; timeout khi chạm ngưỡng
    printAfterEpisode=int(DQN_EP_AMOUNT * 0.04),
    saveDQNModel=True,      # True để lưu checkpoint/best/final
    save_dir="/content/drive/MyDrive/DQN_PhaseF3",   # Thư mục lưu model khi saveDQNModel=True
    plot_metrics=True,     # Vẽ biểu đồ sau train
    window=50,             # Cửa sổ moving average cho metric/plot
    # load_path != None: resume từ checkpoint, model/optimizer/epsilon có thể được nạp từ file.
    load_path=None,          # Ví dụ: "DQN_Phase1/best_dqn.pt" để train tiếp
    load_map_location="cpu", # 'cpu' hoặc 'cuda'/'cuda:0'
    # epsilon_override ưu tiên cao nhất: set lại epsilon sau khi load (reheat exploration), nếu ko load thì ko set
    epsilon_override=None,
    learn_every=4,
)


# ===== Cell 44: 9.7. Evaluate Double DQL với Decayed Epsilon Greedy =====
dqn_agent = DQNAgent(
    state_dim=DQNStateProcessor.STATE_DIM,  # BẮT BUỘC khớp shape state hiện tại của env
    gamma=0.98,            # Discount factor (ưu tiên reward tương lai)
    lr=3e-4,               # Learning rate của optimizer
    hidden_dim=128,        # Số neuron mỗi nhánh MLP
    buffer_capacity=200000,# Replay buffer lớn -> mẫu đa dạng hơn nhưng tốn RAM hơn
    batch_size=64,        # Batch lớn ổn định hơn nhưng chậm/đòi GPU-RAM hơn
    tau=0.005,  # Soft update: nhỏ -> target ổn định hơn, lớn -> bám online nhanh hơn
    # epsilon_start chỉ có hiệu lực khi train mới (không load checkpoint).
    # Nếu có load_path thì epsilon có thể bị ghi đè bởi checkpoint và/hoặc epsilon_override bên dưới.
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=float(np.power(0.05, 1.0 / int(50000 * 0.75))),
)

# Đánh giá policy DQN sau khi train (greedy, epsilon=0 trong lúc evaluate).
dqn_eval_results = evaluate_dqn_on_env(
    # load_path=None: dùng trực tiếp weights đang có trong dqn_agent (có thể đã train hoặc chưa).
    # load_path!=None: nạp weights từ checkpoint file trước khi evaluate.
    
    dqn_agent=dqn_agent,
    maze=maze,
    episodes=1000,            # Số episode đánh giá
    fixed_map=True,         # True: giữ 1 layout map để so sánh ổn định; False: random map mỗi ep
    seed=42,                # Seed cho chuỗi env_seed ở evaluate
    sleep_time=0.05,         # >0 nếu muốn xem UI chậm lại từng bước
    use_ui=False,           # True để render UI
    load_path="FinalModel/DQN3/best_dqn.pt",         # Ví dụ: "DQN_Phase1/best_dqn.pt" nếu muốn evaluate từ checkpoint file
    load_map_location="cpu",# Dùng khi load_path != None: 'cpu' hoặc 'cuda'/'cuda:0'
)
_print_dqn_eval("Kết quả", dqn_eval_results)
