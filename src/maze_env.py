"""The multi-floor maze environment: constants, monsters, floor generation, MazeEnv and the reward settings.

Copied cell by cell from the Zeppelin note notebooks/MultiFloorMaze_Zeppelin.json so the code can be read on GitHub.
Only the "%pyspark" line at the top of each cell is removed.
Some helpers call Zeppelin's z object for the game UI, so run the code in the notebook (or the Colab notebook for Double DQN).
"""

# ===== Cell 0: 1. Khai báo thư viện =====
import numpy as np
import copy
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle
import os
import time

# Link train Double DQN trên GG Collab: https://colab.research.google.com/drive/1juScovKKG9k0vtVOJIrnX10Bp0okmBMw?usp=sharing


# ===== Cell 2: 2. Các constant =====
TOTAL_FLOORS = 3

EMPTY = 0
WALL = 1
KEY = 2
BOMB = 3
HOLE = 4
BLOOD = 5
DOOR = 6
STAIRS_UP = 7
ENTRY = 8
GOAL = 9
MONSTER = 10
SHIELD = 11
SOFT_WALL = 12

# Các hướng di chuyển theo thứ tự: up, right, down, left
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Các tỷ lệ của quái
MONSTER_MISS_WHEN_CHASE = 0.35
MONSTER_STAY = 0.20

# Khoảng cách tối thiểu đặt bomb/ blood
MIN_DIST_PLAYER_BB = 3
MIN_DIST_BB_BB = 3

# Khoảng cách tối thiểu đặt Key-Spawn và Door-Goal
MIN_DIST_SPAWN_KEY = 4
MIN_DIST_DOOR_GOAL = 4

# Số lượng item mỗi floor
MAX_SOFT_WALL_PER_FLOOR = 5
MAX_BOMB_PER_FLOOR = 2
MAX_BLOOD_PER_FLOOR = 3


# ===== Cell 3: 3.1. Định nghĩa tác tử =====
class Agent:
    pass


# ===== Cell 4: 3.2. Định nghĩa quái trong mê cung =====
class Monster:
    def __init__(self, floor, y, x, stun=0, chasing=False):
        self.floor = floor
        self.y = y
        self.x = x
        # Biểu thị số bước còn lại để hết choáng
        self.stun = stun
        # Biểu thị có đang đuổi theo người chơi ko
        self.chasing = chasing


# ===== Cell 5: 4.1. Các pattern floor mê cung =====
PATTERNS = [
    # Vertical center
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    # Vertical left
    [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ],
    # Vertical right
    [
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ],
    # Horizontal center
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    # Horizontal top
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    # Horizontal bottom
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
]


# ===== Cell 6: 4.2. Cơ chế random floor =====
def manhattan(a, b):
    # Khoảng cách manhattan giữa 2 ô có tọa độ (y, x)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def _empty_cells_in_region(floor, region):
    # Lấy danh sách tọa độ ô EMPTY trong vùng
    return [(y, x) for y, x in region if floor[y][x] == EMPTY]
    
def _bfs_reachable(floor, start, target, allow_door=False):
    # BFS kiểm tra từ start có tới được target hay không.
    if start is None or target is None:
        return False
        
    rows, cols = len(floor), len(floor[0])
    
    # Các ô bị chặn: WALL, HOLE và DOOR (nếu allow_door=False). SOFT_WALL bị coi là đi được (để agent có thể dùng búa phá)
    blocked = {WALL, HOLE}
    # Chặn door nếu ko phải là điểm xét ở start hay target
    if not allow_door:
        blocked.add(DOOR)
        
    # Xem ô mà BFS lan kế tiếp cho phải nằm trong bị chặn hay ngoài biên ko
    def is_passable(y, x):
        if y < 0 or y >= rows or x < 0 or x >= cols:
            return False
        return floor[y][x] not in blocked

    sy, sx = start
    ty, tx = target
    if not is_passable(sy, sx):
        return False

    q = deque([(sy, sx)])
    visited = {(sy, sx)}
    while q:
        y, x = q.popleft()
        if (y, x) == (ty, tx):
            return True
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if (ny, nx) not in visited and is_passable(ny, nx):
                visited.add((ny, nx))
                q.append((ny, nx))
    return False


# ====================================================
def generating_floor(floor_index, total_floors):
    # floor_index: 0 = tầng đầu, total_floors-1 = tầng cuối, giữa = tầng giữa
    # total_floors: tổng số tầng (ít nhất 1)
    for attempt in range(200):
        
        # BƯỚC 1: CHỌN RANDOM 1 PATTERN CỦA FLOOR VỀ TƯỜNG ĐỂ CHIA LÀM 2 NỬA CHO FLOOR ĐÓ
        p = random.choice(PATTERNS)
        # Gán pattern tường vào floor trống
        floor = [list(row) for row in p]
        
        # Biến để gán vị trí các bức tường trong floor có pattern mới tạo
        rows, cols = len(floor), len(floor[0])
        wall_cells = [(y, x) for y in range(rows) for x in range(cols) if floor[y][x] == WALL]
        if not wall_cells:
            continue
        
        # Chọn random 1 trong các vị trí wall để làm cửa, từ đó có ranh giới ngăn 2 vùng
        y_door, x_door = random.choice(wall_cells)
        floor[y_door][x_door] = DOOR

        # BƯỚC 2: Lấy 2 vùng của một floor đã chia ranh giới, regions gồm 2 list region: mỗi list chứa tọa độ các vị trí empty của mỗi region đó
        regions = get_two_regions(floor)
        if regions is None:
            continue
        
        # BƯỚC 3: Đặt các thành phần quan trọng trước
        # Mỗi region bao gồm key_region và goal_region, 2 phần region này random lẫn nhau trái phải hay trên xuống
        # key_region: đặt ENTRY và KEY
        # goal_region: tầng cuối đặt GOAL, tầng khác đặt STAIRS_UP, và 1 MONSTER
        # Shield đặt mỗi tầng 1 cái ở key_region
        # Hole mỗi tầng đặt 1 cái mỗi region (key_region + goal_region), phải xét sao để đường đi luôn luôn tới được đích
        
        # entry_pos: Trả về vị trí ENTRY của mỗi tầng
        entry_pos = place_objects(floor, regions, floor_index, total_floors)
        if entry_pos is None:
            # Pattern này không bố trí được đủ item đúng luật -> thử lại pattern khác
            continue
        
        # BƯỚC 4: Thêm SOFT_WALL (ko thêm trên ENTRY)
        add_soft_walls(floor, max_soft_per_floor=MAX_SOFT_WALL_PER_FLOOR, entry_pos=entry_pos)
         
        # BƯỚC 5: Kiểm tra lại toàn cục một lần nữa
        if has_required_paths(floor, floor_index, total_floors, entry_pos):
            return floor, entry_pos
            
        # Nếu vẫn fail thì thử pattern khác
        
    raise RuntimeError(f"Không generate được floor {floor_index} sau 200 lần thử")


# BƯỚC 2: Lấy 2 vùng của một floor đã chia ranh giới, regions gồm 2 list region: mỗi list chứa tọa độ các vị trí empty của mỗi region đó
def get_two_regions(floor):
    # Coi WALL và DOOR là chặn, không đi xuyên qua
    # Ô chỉ đi được khi là EMPTY hoặc các loại khác không phải tường/cửa
    # Dùng BFS: từ một ô trống, thăm tất cả ô trống nối với nó (4 hướng lên/xuống/trái/phải) mà không đi qua tường/cửa -> được 1 vùng
    # Sau khi có vùng 1, tìm một ô trống chưa thuộc vùng nào, BFS tiếp -> được vùng 2
    # Return [region_0, region_1], mỗi phần là list các (y, x) thuộc vùng đó
    
    rows, cols = len(floor), len(floor[0])
    blocked = {WALL, DOOR}

    # Xem ô mà BFS lan kế tiếp cho phải nằm trong bị chặn hay ngoài biên ko
    def is_passable(y, x):
        if y < 0 or y >= rows or x < 0 or x >= cols:
            return False
        return floor[y][x] not in blocked

    visited = set()
    regions = []
    for sy in range(rows):
        for sx in range(cols):
            if (sy, sx) in visited or not is_passable(sy, sx):
                continue
            # BFS từ (sy, sx) để lấy hết vùng này
            region = []
            q = deque([(sy, sx)])
            visited.add((sy, sx))
            while q:
                y, x = q.popleft()
                region.append((y, x))
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if (ny, nx) not in visited and is_passable(ny, nx):
                        visited.add((ny, nx))
                        q.append((ny, nx))
            regions.append(region)

    if len(regions) != 2:
        return None
    return regions

# BƯỚC 3: Đặt các thành phần quan trọng trước
def place_objects(floor, regions, floor_index, total_floors):
    # Đặt object theo 2 vùng:
    # - Tầng không phải tầng cuối: ENTRY + KEY  |  STAIRS_UP + MONSTER
    # - Tầng cuối:                 ENTRY + KEY  |  GOAL + MONSTER

    # Sau đó đặt shield và hole
    # Trả về entry_pos (y, x) là vị trí ENTRY mỗi floor
        
    rows, cols = len(floor), len(floor[0])
    regions = list(regions)
    random.shuffle(regions)
    key_region, goal_region = regions[0], regions[1]

    key_cells_all = _empty_cells_in_region(floor, key_region)
    goal_cells_all = _empty_cells_in_region(floor, goal_region)
    if not key_cells_all or not goal_cells_all:
        return None
        
    spawn_y, spawn_x = None, None

    # P1: key_region: ENTRY + KEY, đảm bảo ENTRY và KEY cách nhau tối thiểu X ô manhattan
    random.shuffle(key_cells_all)
    spawn_y, spawn_x = key_cells_all[0]
    floor[spawn_y][spawn_x] = ENTRY
    entry_pos = (spawn_y, spawn_x)

    key_candidates = [cell for cell in key_cells_all[1:] if manhattan(entry_pos, cell) >= MIN_DIST_SPAWN_KEY]
    if not key_candidates:
        return None
    yk, xk = random.choice(key_candidates)
    floor[yk][xk] = KEY

    # P2: goal_region: (STAIRS_UP hoặc GOAL) + MONSTER, cách DOOR tối thiểu X ô manhattan
    door_pos = None
    for y in range(rows):
        for x in range(cols):
            if floor[y][x] == DOOR:
                door_pos = (y, x)
                break
        if door_pos:
            break
    if door_pos is None:
        return None

    goal_cells = [cell for cell in goal_cells_all if manhattan(door_pos, cell) >= MIN_DIST_DOOR_GOAL]
    if len(goal_cells) < 2:
        return None

    random.shuffle(goal_cells)
    primary_y, primary_x = goal_cells[0]
    if floor_index == total_floors - 1:
        floor[primary_y][primary_x] = GOAL
    else:
        floor[primary_y][primary_x] = STAIRS_UP

    ym, xm = goal_cells[1]
    floor[ym][xm] = MONSTER


    # BƯỚC PHỤ: Thêm SHIELD và HOLE sau khi đã đặt xong đường chính
    # Dùng luôn key_region / goal_region đã chọn ở trên (spawn chắc chắn thuộc key_region)
    empties_key = _empty_cells_in_region(floor, key_region)
    empties_goal = _empty_cells_in_region(floor, goal_region)

    # Đặt 1 SHIELD ở key_region (không đè spawn/ entry)
    shield_candidates = [(y, x) for (y, x) in empties_key
                         if not (spawn_y is not None and spawn_x is not None and (y, x) == (spawn_y, spawn_x))]
    if shield_candidates:
        sy, sx = random.choice(shield_candidates)
        floor[sy][sx] = SHIELD
        empties_key = [(y, x) for (y, x) in empties_key if (y, x) != (sy, sx)]

    # Đặt 1 HOLE ở key_region (không đè spawn/ entry)
    hole_key_candidates = [(y, x) for (y, x) in empties_key
                           if not (spawn_y is not None and spawn_x is not None and (y, x) == (spawn_y, spawn_x))]
    if hole_key_candidates:
        hy, hx = random.choice(hole_key_candidates)
        floor[hy][hx] = HOLE

    # Đặt 1 HOLE ở goal_region
    hole_goal_candidates = list(empties_goal)
    if hole_goal_candidates:
        gy, gx = random.choice(hole_goal_candidates)
        floor[gy][gx] = HOLE

    return (spawn_y, spawn_x)
    

# BƯỚC 4: Thêm SOFT_WALL (ko thêm trên ENTRY)
def add_soft_walls(floor, max_soft_per_floor, entry_pos=None):
    rows, cols = len(floor), len(floor[0])
    candidates = []
    for y in range(rows):
        for x in range(cols):
            if floor[y][x] != EMPTY:
                continue
            if entry_pos is not None and (y, x) == entry_pos:
                continue
            candidates.append((y, x))
    
    random.shuffle(candidates)
    for i, (y, x) in enumerate(candidates):
        if i >= max_soft_per_floor:
            break
        floor[y][x] = SOFT_WALL
        
        
# BƯỚC 5: Kiểm tra lại toàn cục một lần nữa
def has_required_paths(floor, floor_index, total_floors, entry_pos):
    # Kiểm tra lại các đường bắt buộc sau:
    # - ENTRY -> KEY
    # - KEY -> DOOR
    # - DOOR -> STAIRS_UP (tầng không phải tầng cuối) hoặc GOAL (tầng cuối)
    
    rows, cols = len(floor), len(floor[0])
    key_pos = None
    door_pos = None
    up_pos = None
    goal_pos = None
    for y in range(rows):
        for x in range(cols):
            v = floor[y][x]
            if v == KEY:
                key_pos = (y, x)
            elif v == DOOR:
                door_pos = (y, x)
            elif v == STAIRS_UP:
                up_pos = (y, x)
            elif v == GOAL:
                goal_pos = (y, x)

    # Check đường từ entry tới key
    if not _bfs_reachable(floor, entry_pos, key_pos, allow_door=False):
        return False
        
    # Check đường từ key tới door
    if not _bfs_reachable(floor, key_pos, door_pos, allow_door=True):
        return False
    
    # Nếu là tầng cuối thì check door tới goal còn ko thì check tới stair_ups
    if floor_index == total_floors - 1:
        if not _bfs_reachable(floor, door_pos, goal_pos, allow_door=True):
            return False
    else:
        if not _bfs_reachable(floor, door_pos, up_pos, allow_door=True):
            return False
            
    return True
    
    
# BƯỚC 6: Mỗi board sẽ generate ra các floor khác nhau, từ đó truyền vào initPos nếu ở tầng 0
def generate_board_and_initPos(total_floors, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    board = []
    entry0 = None
    for fi in range(total_floors):
        floor, entry_pos = generating_floor(floor_index=fi, total_floors=total_floors)
        board.append(floor)
        if fi == 0:
            entry0 = entry_pos

    initPos = (0, int(entry0[0]), int(entry0[1]))
    return board, initPos


# ===== Cell 7: 5.1. Định nghĩa môi trường MazeEnv =====
class MazeEnv:
    def __init__(self, board, rewards, initPos):
        # Sơ đồ mê cung
        self.initBoard = board # shape: [floors][rows][cols] -> 3D
        # Các điểm cộng/ trừ
        self.rewards = rewards
        # Bao gồm (floor, y, x) cho biết vị trí hiện tại
        self.initPos = initPos
        # Máu tối đa
        self.maxHP = 3
        
        # Biến đại diện cho số step tối đa thực hiện mỗi lượt chơi/ theo dõi số step
        self.maxSteps = 250
        self.stepCount = 0
    
        # ENTRY vị trí xuất hiện của từng tầng
        self.entryPosByFloor = {}
        self.getFloorInformation() # Gán thông tin vào biến trên, mỗi khi khởi tạo hoặc reset lại map
        
        # Biến chứa vị trí monster trên map, để biểu thị sau, vì monster là vật thể động còn map là tĩnh
        # Monster vision đại diện cho khoảng cách manhanttan từ monster đến player để monster thực hiện đuổi theo nếu trong phạm vi
        # MonsterStun: Số step monster phải trải qua nếu bị stun
        self.monsters = []
        self.monsterVision = 3
        self.monsterStun = 2
        
        # Búa
        # Số bước cần để hồi búa trước khi dùng - lớn hơn monsterStun để ko có chuyện reward hacking 
        self.hammerCooldown = 3
        # Số bước còn lại để hồi búa/ biến theo dõi
        self.hammerTimer = 0
        
        # Spawn động Bomb / Máu
        # Mỗi tầng tối đa m bomb, n máu tồn tại cùng lúc
        self.maxBombPerFloor = MAX_BOMB_PER_FLOOR
        self.maxBloodPerFloor = MAX_BLOOD_PER_FLOOR
        # Sau một số bước ngẫu nhiên trong [min,max] sẽ spawn 1 bomb hoặc 1 máu
        self.itemSpawnMinSteps = 5
        self.itemSpawnMaxSteps = 7
        self.stepsSinceItemSpawn = 0
        self.nextItemSpawnSteps = random.randint(self.itemSpawnMinSteps, self.itemSpawnMaxSteps)
        
        # Cơ chế stagnation
        # Early-stop nếu agent không tạo tiến triển nhiệm vụ trong quá nhiều bước liên tiếp
        # Ngưỡng này sẽ tự co giãn theo maxSteps để tránh hardcode.
        self.stagnationRatio = 0.38
        self.minNoProgressSteps = 20
        self.maxNoProgressSteps = max(self.minNoProgressSteps, int(self.maxSteps * self.stagnationRatio))
        self.noProgressSteps = 0
        
        # Intrinsic motivation (novelty): thưởng nhỏ khi agent đi tới ô mới trong cùng episode
        self.novelty_bonus = float(self.rewards.get("novelty_bonus", 0.0))
        
        # Biến lưu loại kết thúc episode
        self.end_reason = None
        
        self.reset()
    
    
    # ====================================================
    def getState(self, context=None):
        if isinstance(context, Agent):
            # 12 ô hình kim cương xung quanh
            cell_up = self.getCell(self.y - 1, self.x)
            cell_down = self.getCell(self.y + 1, self.x)
            cell_left = self.getCell(self.y, self.x - 1)
            cell_right = self.getCell(self.y, self.x + 1)
        
            cell_up2 = self.getCell(self.y - 2, self.x)
            cell_down2 = self.getCell(self.y + 2, self.x)
            cell_left2 = self.getCell(self.y, self.x - 2)
            cell_right2 = self.getCell(self.y, self.x + 2)
        
            cell_ul = self.getCell(self.y - 1, self.x - 1)
            cell_ur = self.getCell(self.y - 1, self.x + 1)
            cell_dl = self.getCell(self.y + 1, self.x - 1)
            cell_dr = self.getCell(self.y + 1, self.x + 1)
            
            # Búa sẵn sàng nếu số bước còn lại để hồi búa là 0
            hammer_ready = 1 if self.hammerTimer == 0 else 0
            
            # Đại diện cho vị trí quái so với agent: 0 = không thấy trong 12 ô, 1..12 = ô tương ứng, 13 = trùng vị trí agent
            monster_cell_index = self.getMonsterCellIndex()
            # Quái có đang stun hay không (0/1)
            monster_stun_flag = self.getMonsterStunFlag()
            
            # HP rời rạc: 0 = nguy hiểm (hp <= 1), 1 = ổn (hp >= 2) -> giảm state space
            hp_state = 0 if self.hp <= 1 else 1
            
            return (
                self.floor, hp_state, self.key, self.shield,
            
                cell_up, cell_down, cell_left, cell_right,
                cell_up2, cell_down2, cell_left2, cell_right2,
                cell_ul, cell_ur, cell_dl, cell_dr,
            
                monster_cell_index, monster_stun_flag, hammer_ready
            )
            
        else:
            return (copy.deepcopy(self.board), self.floor, self.y, self.x, self.hp, self.key, self.score, self.hammerTimer)


    # ====================================================
    # Dùng khi chơi trên map để respawn lại nhân vật khi trúng hole hoặc quái nếu ko có khiên
    def getFloorStartPos(self, floor):
        if floor in self.entryPosByFloor:
            return self.entryPosByFloor[floor]

        fb = self.initBoard[floor]
        for y, row in enumerate(fb):
            for x, cell in enumerate(row):
                if cell == ENTRY:
                    return (y, x)

        return (self.initPos[1], self.initPos[2])
    
    # Gán vị trí entry vào trong cache entry
    def getFloorInformation(self):
        # Cache vị trí ENTRY cho từng tầng từ initBoard (map tĩnh)
        self.entryPosByFloor = {}
        for f, floor in enumerate(self.initBoard):
            floor_np = np.array(floor)
            down = np.where(floor_np == ENTRY)
            if len(down[0]) > 0:
                self.entryPosByFloor[f] = (int(down[0][0]), int(down[1][0]))
       
      
    # ====================================================
    # Ô tại vị trí y,x là loại gì
    def getCell(self, y, x):
        floor_board = self.board[self.floor]
        rows = len(floor_board)
        cols = len(floor_board[0])
        if y < 0 or y >= rows or x < 0 or x >= cols:
            # Coi ngoài map là tường
            return WALL
        
        return int(floor_board[y][x])
        
    def isTerm(self):
        return self.isFinished
        
    def isWall(self, cell):
        if cell == WALL or cell == SOFT_WALL:
            return True
        if cell == DOOR and self.key == 0:
            return True
        return False

    def getCells12(self):
        # 12 ô xung quanh agent theo đúng thứ tự dùng cho state và monster
        return [
            (self.y - 1, self.x), (self.y + 1, self.x), (self.y, self.x - 1), (self.y, self.x + 1),
            (self.y - 2, self.x), (self.y + 2, self.x), (self.y, self.x - 2), (self.y, self.x + 2),
            (self.y - 1, self.x - 1), (self.y - 1, self.x + 1), (self.y + 1, self.x - 1), (self.y + 1, self.x + 1),
        ]
      
      
    # ====================================================
    def getMonsterCellIndex(self):
        # Vị trí quái trong 12 ô quanh agent:
        # - 0 = không có trong 12 ô
        # - 1..12 = quái nằm ở ô nào (cùng thứ tự như getState)
        # - 13 = quái trùng vị trí agent
        # Thứ tự: up, down, left, right, up2, down2, left2, right2, ul, ur, dl, dr
        for m in self.monsters:
            if m.floor == self.floor and m.y == self.y and m.x == self.x:
                return 13

        cells_12 = self.getCells12()
        
        for idx, (my, mx) in enumerate(cells_12):
            for m in self.monsters:
                if m.floor == self.floor and m.y == my and m.x == mx:
                    return idx + 1  # 1..12
        return 0
    
    def getMonsterStunFlag(self):
        # Cờ stun quái trên tầng hiện tại cho state (giả định mỗi tầng tối đa 1 quái như generator hiện tại)
        # 0 = quái không stun (hoặc không có quái trên tầng hiện tại)
        # 1 = quái đang stun
        for m in self.monsters:
            if m.floor == self.floor:
                return 1 if m.stun > 0 else 0
        return 0
        
    # Ô này có chặn quái, ko cho nó đi vào ko
    def isMonsterBlocked(self, cell):
        if cell in [WALL, SOFT_WALL, DOOR, HOLE, BOMB]:
            return True
        return False
    
    # Mỗi tầng có duy nhất 1 monster và vị trí spawn của nó sẽ được đặt kĩ lưỡng
    def moveMonsters(self):
        for monster in self.monsters:
            # Nếu vị trí của monster trong self.monster ko nằm ở tầng cùng agent thì ko di chuyển
            if monster.floor != self.floor:
                continue
            
            my = monster.y
            mx = monster.x

            # Xử lý stun
            if monster.stun > 0:
                monster.stun -= 1
                monster.chasing = False
                continue
            
            # Tính khoảng cách manhattan để xét nằm trong phạm vi đuổi theo ko
            dist = abs(my - self.y) + abs(mx - self.x)
            if dist <= self.monsterVision:
                monster.chasing = True
                
                # % chance hụt bước khi đang đuổi
                if random.random() < MONSTER_MISS_WHEN_CHASE:
                    continue
                
                # Chase player
                moves = DIRS.copy()   # [(-1,0),(0,1),(1,0),(0,-1)]
                random.shuffle(moves)
                
                # Ưu tiên hướng giảm Manhattan distance
                def manhattan_after_move(m):
                    return abs((my + m[0]) - self.y) + abs((mx + m[1]) - self.x)
                
                moves.sort(key=manhattan_after_move)
                
            # Không thì di chuyển random
            else:
                monster.chasing = False
                
                # % tỷ lệ đứng yên
                if random.random() < MONSTER_STAY:
                    continue
                
                moves = DIRS.copy()
                random.shuffle(moves)
            
            # Thực hiện di chuyển quái
            for dy,dx in moves:
                ny = my + dy
                nx = mx + dx
                if not self.isMonsterBlocked(self.getCell(ny, nx)):
                    monster.y = ny
                    monster.x = nx
                    break
    
    # Event xảy ra khi quái và người chơi va chạm nhau
    def checkMonsterCollision(self, reward, prev_player_pos=None, prev_monster_positions=None):
        # Khi quái và người chơi va chạm nhau - nghĩa là cả 2 cùng cell, thì respawn cả 2 về lại vị trí ban đầu của chúng mỗi tầng
        for i, monster in enumerate(self.monsters):
            same_cell_collision = (
                monster.floor == self.floor and monster.y == self.y and monster.x == self.x
            )

            # Va chạm kiểu "đi xuyên": player và monster đổi vị trí cho nhau trong cùng 1 step.
            crossing_collision = False
            if (
                not same_cell_collision
                and prev_player_pos is not None
                and prev_monster_positions is not None
                and i in prev_monster_positions
            ):
                prev_floor, prev_my, prev_mx = prev_monster_positions[i]
                curr_player = (self.y, self.x)
                player_moved = curr_player != prev_player_pos

                if (
                    player_moved
                    and prev_floor == self.floor
                    and monster.floor == self.floor
                    and (prev_my, prev_mx) == curr_player
                    and (monster.y, monster.x) == prev_player_pos
                ):
                    crossing_collision = True

            if same_cell_collision or crossing_collision:
                # Nếu monster đang stun thì không gây damage cho player
                if monster.stun > 0:
                    continue
            
                if self.shield == 1: # Có khiên chặn quái
                    self.shield = 0
                    reward += self.rewards["shield_block"]
                
                    # Stun monster x step
                    monster.stun = self.monsterStun
                
                    return reward, True
                else:
                    # Trừ điểm cho việc bị trúng quái
                    self.hp -= 1
                    reward += self.rewards["monster"]
    
                # Respawn player
                self.y, self.x = self.getFloorStartPos(self.floor)
                
                # Respawn monster
                monster.floor, monster.y, monster.x = self.monsterSpawn[i]
                monster.stun = 0
                    
                if self.hp <= 0:
                    reward += self.rewards["death"]
                    self.isFinished = True
                    self.end_reason = "death"

                return reward, True
        return reward, False


    # ====================================================
    # Spawn động Bomb / Máu theo số step
    def spawnBombBlood(self):
        # Không spawn nếu game đã kết thúc
        if self.isFinished:
            return
        
        # Tăng bộ đếm step cho spawn
        self.stepsSinceItemSpawn += 1
        if self.stepsSinceItemSpawn < self.nextItemSpawnSteps:
            return
        
        # Đến ngưỡng spawn: chọn 1 ô EMPTY hợp lệ trên tầng hiện tại
        f = self.floor
        fb = self.board[f]
        rows = len(fb)
        cols = len(fb[0])
        
        # Tập vị trí monster trên tầng hiện tại
        monster_cells = {(m.y, m.x) for m in self.monsters if m.floor == f}
        # Tập vị trí bomb / máu hiện có trên tầng hiện tại
        bomb_cells = []
        blood_cells = []
        for y in range(rows):
            for x in range(cols):
                if fb[y][x] == BOMB:
                    bomb_cells.append((y, x))
                elif fb[y][x] == BLOOD:
                    blood_cells.append((y, x))
        
        candidates = []
        for y in range(rows):
            for x in range(cols):
                if fb[y][x] != EMPTY:
                    continue
                # Không spawn dưới player
                if (y, x) == (self.y, self.x):
                    continue
                # Không spawn dưới monster
                if (y, x) in monster_cells:
                    continue
                # Không spawn quá gần player (Manhattan < MIN_DIST_PLAYER_BB)
                if abs(y - self.y) + abs(x - self.x) < MIN_DIST_PLAYER_BB:
                    continue
                # Không spawn quá gần bomb/máu hiện có (Manhattan < MIN_DIST_BB_BB)
                too_close_item = False
                for (iy, ix) in bomb_cells:
                    if abs(y - iy) + abs(x - ix) < MIN_DIST_BB_BB:
                        too_close_item = True
                        break
                if too_close_item:
                    continue
                for (iy, ix) in blood_cells:
                    if abs(y - iy) + abs(x - ix) < MIN_DIST_BB_BB:
                        too_close_item = True
                        break
                if too_close_item:
                    continue
                candidates.append((y, x))
        
        if not candidates:
            # Không có ô phù hợp, reset timer
            self.stepsSinceItemSpawn = 0
            self.nextItemSpawnSteps = random.randint(self.itemSpawnMinSteps, self.itemSpawnMaxSteps)
            return
        
        # Đếm bomb / máu hiện có trên tầng
        bomb_count = len(bomb_cells)
        blood_count = len(blood_cells)
        
        if bomb_count >= self.maxBombPerFloor and blood_count >= self.maxBloodPerFloor:
            # Đủ số lượng cả hai loại
            self.stepsSinceItemSpawn = 0
            self.nextItemSpawnSteps = random.randint(self.itemSpawnMinSteps, self.itemSpawnMaxSteps)
            return
        
        # Quyết định spawn Bomb hay Máu
        if bomb_count >= self.maxBombPerFloor and blood_count < self.maxBloodPerFloor:
            spawn_type = BLOOD
        elif blood_count >= self.maxBloodPerFloor and bomb_count < self.maxBombPerFloor:
            spawn_type = BOMB
        else:
            # Cả hai đều dưới ngưỡng chọn ngẫu nhiên 50/50
            spawn_type = BOMB if random.random() < 0.5 else BLOOD
        
        # Chọn ô ngẫu nhiên
        y, x = random.choice(candidates)
        fb[y][x] = spawn_type
        
        # Đặt lại timer cho lần spawn tiếp theo
        self.stepsSinceItemSpawn = 0
        self.nextItemSpawnSteps = random.randint(self.itemSpawnMinSteps, self.itemSpawnMaxSteps)
    
    
    # ====================================================
    def reset(self, seed=None, randomize_map=True):
        # Nếu truyền seed thì cố định lại random theo seed, dùng cho train để so sánh các thuật toán train cùng seed với nhau
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if randomize_map:
            # Sinh lại map mới dựa trên số tầng hiện tại của initBoard
            # Map sẽ phụ thuộc vào seed truyền vào reset (nếu có),
            # giúp chuỗi map random lặp lại được khi dùng cùng seed.
            floorAmount = len(self.initBoard)
            try:
                newBoard, newInitPos = generate_board_and_initPos(floorAmount, seed=seed)
            except RuntimeError as e:
                # Nếu gen map mới thất bại (sau 200 lần thử), giữ nguyên map cũ để không làm dừng quá trình train
                print(f"Reset() giữ nguyên map cũ vì generate_board_and_initPos fail: {e}")
            else:
                self.initBoard = newBoard
                self.initPos = newInitPos
                
                # Xóa cache respawn cũ, lấy vị trí ENTRY của map mới
                self.getFloorInformation()

        # Đồng bộ ngưỡng stagnation theo maxSteps hiện tại.
        self.maxNoProgressSteps = max(self.minNoProgressSteps, int(self.maxSteps * self.stagnationRatio))

        # Reset timer spawn động Bomb/Máu
        self.stepsSinceItemSpawn = 0
        self.nextItemSpawnSteps = random.randint(self.itemSpawnMinSteps, self.itemSpawnMaxSteps)
     
        self.board = [np.array(floor) for floor in self.initBoard]
        self.floor, self.y, self.x = self.initPos

        self.score = 0.0
        self.hp = self.maxHP
        self.key = 0
        self.shield = 0
        self.isFinished = False
        self.stepCount = 0
        self.end_reason = None

        # Reset lại board, có vị trí monster ban đầu
        # Biến chứa vị trí monster trên map, lưu vị trí của chúng để quản lý riêng, vì monster là vật thể động còn map là tĩnh
        self.monsters = []
        # Biến lưu vị trí spawn của monster để monster respawn nếu trúng player
        self.monsterSpawn = []
        for f, floor in enumerate(self.board):
            for y, row in enumerate(floor):
                for x, cell in enumerate(row):
                    if cell == MONSTER:
                        # Đưa tọa độ vào biến để quản lý động monster
                        self.monsters.append(Monster(f, y, x))
                        # Đưa tọa độ vào biến để biết vị trí cần đặt respawn monster
                        self.monsterSpawn.append((f,y,x))
                        # Xóa thành phần monster tĩnh ra khỏi map
                        self.board[f][y][x] = EMPTY
        
        self.hammerTimer = 0 # Búa sẵn sàng khi bắt đầu

        self.noProgressSteps = 0 # reset bộ đếm tiến độ

        # Reset tập các ô đã thăm trong episode này.
        # Để tránh thưởng ngay tại vị trí khởi đầu, ta đánh dấu ô start là đã thăm
        # Gán vào dict đếm số lần thăm (pos_key -> count)
        self.visited_this_episode = {(self.floor, self.y, self.x): 1}
        
        return self.getState()
    
    # ====================================================
    def getPosActions(self):
        if self.isFinished:
            return []

        actions = []

        # UP: 0
        if not self.isWall(self.getCell(self.y - 1, self.x)):
            actions.append(0)

        # RIGHT: 1
        if not self.isWall(self.getCell(self.y, self.x + 1)):
            actions.append(1)

        # DOWN: 2
        if not self.isWall(self.getCell(self.y + 1, self.x)):
            actions.append(2)

        # LEFT: 3
        if not self.isWall(self.getCell(self.y, self.x - 1)):
            actions.append(3)
        
        # WAIT: 8 (đứng yên, hợp lệ bất kể ô xung quanh là gì)
        actions.append(8)
        
        # HAMMER DIRECTIONS
        # Các hành động dùng búa chỉ tính là valid khi búa ko cooldown, ko thì là invalid
        if self.hammerTimer == 0:
            actions.extend([4, 5, 6, 7])
            
        return actions

    # Tổng kết lại điểm sau khi kết thúc 1 step
    def finalizeStep(self, reward, progress_event=False):
        # Cập nhật counter stagnation, có thể kết thúc sớm nếu không tiến triển quá lâu
        # progress_event=True cho các mốc như key/door/stairs/goal
        effective_progress_event = bool(progress_event)

        # Tính số lần thăm vị trí hiện tại trong episode (pos_key -> count)
        pos_key = (self.floor, self.y, self.x)
        prev_count = self.visited_this_episode.get(pos_key, 0)
        new_count = prev_count + 1
        self.visited_this_episode[pos_key] = new_count

        # Intrinsic novelty: chỉ cộng bonus lần đầu vào ô này
        if self.novelty_bonus != 0.0 and prev_count == 0:
            reward += self.novelty_bonus
            effective_progress_event = True

        # Phạt lặp vị trí: tránh local optima kiểu đi qua lại 2-4 ô rồi stagnation
        repeat_threshold = int(self.rewards.get("repeat_visit_threshold", 3))
        repeat_penalty_scale = float(self.rewards.get("repeat_visit_penalty_scale", 0.5))
        if new_count >= repeat_threshold:
            # threshold=3 => new_count=3 bị phạt: -scale*(3-2) = -scale
            reward += -repeat_penalty_scale * (new_count - (repeat_threshold - 1))

        if effective_progress_event:
            self.noProgressSteps = 0
        else:
            self.noProgressSteps += 1

        if (not self.isFinished) and self.noProgressSteps >= self.maxNoProgressSteps:
            self.isFinished = True
            reward += self.rewards["stagnation"]
            self.end_reason = "stagnation"

        self.score += reward
        return self.getState(), reward, self.isFinished

    # ====================================================
    def step(self, a):
        # Kiểm tra trò chơi kết thúc chưa trước khi thực hiện bước đi
        if self.isFinished:
            return self.getState(), 0, True
            
            
        # Tăng 1 step - Kết thúc khi vượt quá số bước cho phép
        self.stepCount += 1
        if self.stepCount >= self.maxSteps:
            self.isFinished = True
            reward = self.rewards["timeout"]
            self.end_reason = "timeout"
            self.score += reward
            return self.getState(), reward, True
        
        # Nếu chưa kết thúc thì khi thực hiện step sẽ -x reward
        reward = self.rewards["step"]
        changedFloor = False
        progress_event = False

        
        # KIỂM TRA ACTION HỢP LỆ
        actions = self.getPosActions()
        
        # Giảm thời gian hồi búa sau mỗi hành động (khi đang cooldown)
        if self.hammerTimer > 0:
            self.hammerTimer -= 1
            
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NẾU LÀ CÁC HÀNH ĐỘNG INVALID
        if a not in actions:
            reward += self.rewards["invalid"]
            
            # Monster vẫn thực hiện move 1 step dù cho hành động này invalid/ player đứng yên một chỗ do làm hành động sai
            self.moveMonsters()
            reward, hit = self.checkMonsterCollision(reward)
            
            # Sau khi player + monster xử lý xong step, có thể spawn Bomb/Máu
            self.spawnBombBlood()
            
            # RETURN, hết step
            return self.finalizeStep(reward, progress_event=progress_event)  # Đụng tường / action sai
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NẾU LÀ CÁC HÀNH ĐỘNG VALID
        # Lưu vị trí trước khi di chuyển
        prev_y = self.y
        prev_x = self.x

        # Thực hiện hành động
        if a == 0:        # Up
            self.y -= 1
        elif a == 1:      # Right
            self.x += 1
        elif a == 2:      # Down
            self.y += 1
        elif a == 3:      # Left
            self.x -= 1
        elif a == 8:      # WAIT
            # Đứng yên: giữ nguyên (self.y, self.x)
            reward += self.rewards["wait"]
        
        # XỬ LÝ ACTION: DÙNG BÚA
        elif a in [4, 5, 6, 7]:
            if a == 4:
                dy, dx = -1, 0
            elif a == 5:
                dy, dx = 0, 1
            elif a == 6:
                dy, dx = 1, 0
            elif a == 7:
                dy, dx = 0, -1
            
            # Vị trí cell xem xét mà búa được dùng
            ny = self.y + dy
            nx = self.x + dx
            
            rows = len(self.board[self.floor])
            cols = len(self.board[self.floor][0])
            
            # Xét trường hợp dùng búa nhưng ngoài map
            out_of_map = not (0 <= ny < rows and 0 <= nx < cols)
            
            monster_hit = False
            target = None
            
            # Xét monster trước
            if not out_of_map:
                target = self.getCell(ny, nx)
        
                # Kiểm tra có monster ở ô đó không
                for i, monster in enumerate(self.monsters):
                    if monster.floor == self.floor and monster.y == ny and monster.x == nx:
                        # Nếu có thì stun monster và trả điểm reward
                            
                        old_stun = monster.stun
                        # Cộng thêm 1 để trừ cho việc chạy monste_move
                        self.monsters[i].stun = self.monsterStun + 1
                        if old_stun == 0:
                            # Chỉ thưởng nếu monster chưa bị stun
                            reward += self.rewards["hammer_monster"]
    
                        monster_hit = True
                        break
    
            # Nếu không phải monster thì xử lý tường hoặc bomb
            if not monster_hit:
                if out_of_map:
                    reward += self.rewards["hammer_break_nothing"]
                elif target == SOFT_WALL:
                    self.board[self.floor][ny][nx] = EMPTY
                    reward += self.rewards["hammer_break_soft_wall"]
                elif target == BOMB:
                    self.board[self.floor][ny][nx] = EMPTY
                    reward += self.rewards["hammer_break_bomb"]
                else:
                    reward += self.rewards["hammer_break_nothing"]
            
            # Búa rơi vào trạng thái cooldown 
            self.hammerTimer = self.hammerCooldown
            
            # Hành động búa ko cần xét tiếp các ô di chuyển tới nên sẽ return sớm
            # Monster move
            self.moveMonsters()
            reward, hit = self.checkMonsterCollision(reward)
            
            # Sau khi player + monster xử lý xong step, có thể spawn Bomb/Máu
            self.spawnBombBlood()
            
            return self.finalizeStep(reward, progress_event=progress_event)


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Kiểm tra ô hiện tại sau khi thực hiện một hành động
        currentCell = int(self.board[self.floor][self.y][self.x])
    
        # GOAL
        # Hoàn thành game
        if currentCell == GOAL:
            # Đạt đích, kết thúc trò chơi
            reward += self.rewards["goal"]
            self.isFinished = True
            self.end_reason = "goal"
            progress_event = True
            
            return self.finalizeStep(reward, progress_event=progress_event)

        # STAIRS_UP: chạm là tự động sang tầng kế tiếp
        elif currentCell == STAIRS_UP:
            floors = len(self.board)
            if self.floor < floors - 1:
                self.floor += 1
                changedFloor = True
                
                # Xóa khiên khi sang tầng mới
                self.shield = 0
                
                # Thưởng cho việc lên tầng mới
                reward += self.rewards["stairs_up_first_time"]
                progress_event = True

                # Đặt nhân vật vào ô ENTRY của tầng mới
                if self.floor in self.entryPosByFloor:
                    self.y, self.x = self.entryPosByFloor[self.floor]
                else:
                    self.y, self.x = self.getFloorStartPos(self.floor)
    
        # BOMB: mất máu và bomb biến mất, người chơi vẫn ở vị trí bị trúng bomb
        # Có shield: Bom mất, agent ở vị trí vừa bước tới chứa bomb bị mất đi
        elif currentCell == BOMB:
            if self.shield == 1:
                self.shield = 0
                reward += self.rewards["shield_block"]
            else:
                # Không có khiên
                self.hp -= 1
                reward += self.rewards["bomb"]
                
            # BOMB biến mất
            self.board[self.floor][self.y][self.x] = EMPTY
                
            # Nếu hết máu -> kết thúc
            if self.hp <= 0:
                reward += self.rewards["death"]
                self.isFinished = True

                self.end_reason = "death"
                return self.finalizeStep(reward, progress_event=progress_event)
        
        # HOLE
        # Rơi xuống hố: mất máu, HỐ KO MẤT ĐI KHI BỊ ĐỤNG VÀO, người chơi teleport lại vị trí bắt đầu của tầng
        elif currentCell == HOLE:
            # Có shield -> Lùi lại ô trước
            if self.shield == 1:
                self.shield = 0
                reward += self.rewards["shield_block"]
        
                # Lùi lại ô trước
                self.y = prev_y
                self.x = prev_x
                
                # Monster move và return sớm để đảm bảo đúng 1 step, ngưng việc kiểm tra cell bị đẩy về
                self.moveMonsters()
                reward, hit = self.checkMonsterCollision(reward)
                
                # Sau khi player + monster xử lý xong step, có thể spawn Bomb/Máu
                self.spawnBombBlood()  
        
                return self.finalizeStep(reward, progress_event=progress_event)
            else:
                # Không có shield -> mất mát, respawn
                self.hp -= 1
                reward += self.rewards["hole"]
            
                # Kiểm tra chết trước
                if self.hp <= 0:
                    # KHI MÁU VỀ 0 LÀ KẾT THÚC TRÒ CHƠI
                    reward += self.rewards["death"]
                    self.isFinished = True

                    self.end_reason = "death"
                    return self.finalizeStep(reward, progress_event=progress_event)
            
                # Teleport về vị trí bắt đầu
                self.y, self.x = self.getFloorStartPos(self.floor)
        
        # BLOOD
        # Nhặt máu (biến mất sau khi nhặt)
        elif currentCell == BLOOD:
            if self.hp < self.maxHP:
                self.hp += 1
                reward += self.rewards["blood_not_full"]
    
            self.board[self.floor][self.y][self.x] = EMPTY # Máu biến mất khi nhặt
    
        # KEY
        # Nhặt chìa khóa
        elif currentCell == KEY:
            self.key = 1
            reward += self.rewards["key"]
            progress_event = True
            self.board[self.floor][self.y][self.x] = EMPTY #  Chìa khóa biến mất
        
        # SHIELD
        # Nhặt shield và được cộng điểm chỉ khi đang không có shield, shield được nhặt sẽ biến mất
        # Nếu còn shield mà nhặt thì coi như ko có chuyện gì, cell shield đó vẫn tồn tại trên bản đồ
        elif currentCell == SHIELD:
            if self.shield == 0:
                self.shield = 1
                reward += self.rewards["shield_pick"]
                self.board[self.floor][self.y][self.x] = EMPTY
        
        # DOOR
        # Cửa tầng (nếu chưa có key thì -> cửa như một bức tường chặn lại)
        elif currentCell == DOOR:
            if self.key == 1:
                reward += self.rewards["door"]
                progress_event = True
                self.board[self.floor][self.y][self.x] = EMPTY #  Cửa biến mất
                self.key = 0 # Key mất khi mở cửa
    
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Monster thực hiện 1 lần hành động mỗi lần player thực hiện 1 step
        # Cần 2 luồng check collision trong cùng step:
        # (1) Trước khi monster move: bắt case player vừa bước vào ô monster
        # (2) Sau khi monster move: bắt case monster bước vào ô player và case "đổi chỗ" (crossing collision)
        if not changedFloor:
            # Check va chạm ngay sau khi player hành động (trước khi monster di chuyển)
            # Tránh bug: player bước vào ô quái, nhưng quái kịp di chuyển ra ô khác nên không bị tính va chạm
            reward, hit = self.checkMonsterCollision(reward)
            if hit:
                self.spawnBombBlood()
                return self.finalizeStep(reward, progress_event=progress_event)

            prev_monster_positions = {i: (m.floor, m.y, m.x) for i, m in enumerate(self.monsters)}
            self.moveMonsters()
            reward, hit = self.checkMonsterCollision(
                reward,
                prev_player_pos=(prev_y, prev_x),
                prev_monster_positions=prev_monster_positions,
            )
            if hit:
                # Sau khi player + monster xử lý xong step, có thể spawn Bomb/Máu
                self.spawnBombBlood()
                return self.finalizeStep(reward, progress_event=progress_event)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sau khi player + monster xử lý xong step, có thể spawn Bomb/Máu
        self.spawnBombBlood()  
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cập nhật score
        # Trả về (state, reward, done)
        return self.finalizeStep(reward, progress_event=progress_event)


# ===== Cell 8: 5.2. Khởi tạo môi trường MazeEnv =====
board, initPos = generate_board_and_initPos(TOTAL_FLOORS)

# HỆ THỐNG PHẦN THƯỞNG/ PHẠT
reward_setting = {
    "step": -0.3,
    "wait": -0.5,
    "invalid": -3,

    "goal": 700,

    "bomb": -12,
    "hole": -28,
    "monster": -22,
    "death": -520,

    "key": 35,
    "door": 45,

    "stairs_up_first_time": 60,

    "blood_not_full": 1,
    "timeout": -220,
    "stagnation": -180,
    
    "novelty_bonus": 0.3,
    "repeat_visit_penalty_scale": 0.3,
    "repeat_visit_threshold": 3,
    
    "shield_pick": 10,
    "shield_block": 5,

    "hammer_break_soft_wall": 1,
    "hammer_break_bomb": 0.5,
    "hammer_break_nothing": -3,
    "hammer_monster": 0.5
}

maze = MazeEnv(board, reward_setting, initPos)
