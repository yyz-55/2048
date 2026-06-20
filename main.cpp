/**
 * 2048 AI — C++ Bitboard + 深度 Expectimax
 *
 * 核心优化 (vs Python):
 *   1. uint64_t 位棋盘 — 16 格 × 4 bit, 移动用查表 (65536 条预计算)
 *   2. 置换表 — unordered_map 缓存 (board, depth) → score
 *   3. 精确残局 — 空格 ≤4 时穷举所有生成 (零采样噪声)
 *   4. 多角蛇形 — 4 角自适应, 评估时取最优
 *   5. 自适应深度 — 空格多浅搜, 空格少深搜 (depth 3~9)
 *
 * 编译: g++ -std=c++17 -O3 -march=native -o 2048 main.cpp
 * 运行: ./2048 --play 10    (自动玩 10 局)
 *       ./2048 --bench 20   (基准测试 20 局)
 */

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// 1. 位棋盘 + 预计算移动表
// ============================================================

using Board = uint64_t;  // 16 tiles × 4 bits = 64 bits
// tile 0 (top-left) = bits 0..3, tile 15 (bottom-right) = bits 60..63
// 每格存 log2(值): 0=空, 1=2, 2=4, ..., 16=65536

// 预计算行移动表: left_table[row_16bit] → {new_row_16bit, score}
// 一行 4 格 × 4 bit = 16 bit → 65536 条目
static std::array<std::pair<uint16_t, int>, 65536> left_table;

// ---- 棋盘操作 ----
inline uint16_t get_row(Board b, int r) {
    return (b >> (r * 16)) & 0xFFFF;
}
inline Board set_row(Board b, int r, uint16_t row) {
    Board mask = ~(0xFFFFULL << (r * 16));
    return (b & mask) | (static_cast<uint64_t>(row) << (r * 16));
}

// 提取列 (转为行格式)
inline uint16_t get_col(Board b, int c) {
    uint16_t v = 0;
    for (int r = 0; r < 4; r++)
        v |= ((b >> (r * 16 + c * 4)) & 0xF) << (r * 4);
    return v;
}
// 写回列
inline Board set_col(Board b, int c, uint16_t val) {
    for (int r = 0; r < 4; r++) {
        int shift = r * 16 + c * 4;
        b = (b & ~(0xFULL << shift)) |
            (static_cast<uint64_t>((val >> (r * 4)) & 0xF) << shift);
    }
    return b;
}

// 翻转行 (用于右滑 = 翻转 → 左滑 → 翻转)
inline uint16_t reverse_row(uint16_t r) {
    return ((r >> 12) & 0xF) | ((r >> 4) & 0xF0) |
           ((r << 4) & 0xF00) | ((r << 12) & 0xF000);
}

// 翻转棋盘行 (用于上下翻转, 下/上滑 = 翻转 → 上/下滑 → 翻转)
inline Board flip_board(Board b) {
    return (b >> 48) | ((b >> 32) & 0xFFFF0000ULL) |
           ((b << 32) & 0xFFFF00000000ULL) | (b << 48);
}

inline int tile_value(int log2_val) { return log2_val == 0 ? 0 : 1 << log2_val; }

// ============================================================
// 2. 初始化移动表 (程序启动时调用一次)
// ============================================================
void init_move_table() {
    for (uint32_t row_bits = 0; row_bits < 65536; row_bits++) {
        int tiles[4];
        for (int i = 0; i < 4; i++)
            tiles[i] = (row_bits >> (i * 4)) & 0xF;

        int merged[4] = {}, score = 0, pos = 0;
        for (int i = 0; i < 4; i++) {
            if (tiles[i] == 0) continue;
            if (pos > 0 && merged[pos - 1] == tiles[i]) {
                merged[pos - 1]++;          // 合并: log2 值 +1
                score += 1 << merged[pos - 1]; // 得分: 合并后的实际值
            } else {
                merged[pos++] = tiles[i];
            }
        }
        uint16_t new_row = 0;
        for (int i = 0; i < 4; i++)
            new_row |= static_cast<uint16_t>(merged[i]) << (i * 4);

        left_table[row_bits] = {new_row, score};
    }
}

// ============================================================
// 3. 模拟移动
// ============================================================
struct MoveResult {
    Board board;
    int score;
    bool moved;
};

MoveResult do_move(Board b, char dir) {
    int total = 0;
    Board nb = 0;
    bool moved = false;

    switch (dir) {
        case 'a': // 左
            for (int r = 0; r < 4; r++) {
                auto [nr, sc] = left_table[get_row(b, r)];
                nb = set_row(nb, r, nr);
                total += sc;
                if (nr != get_row(b, r)) moved = true;
            }
            break;
        case 'd': // 右: 翻转→左→翻转
            for (int r = 0; r < 4; r++) {
                auto [nr, sc] = left_table[reverse_row(get_row(b, r))];
                nb = set_row(nb, r, reverse_row(nr));
                total += sc;
                if (nr != reverse_row(get_row(b, r))) moved = true;
            }
            break;
        case 'w': // 上
            for (int c = 0; c < 4; c++) {
                auto [nr, sc] = left_table[get_col(b, c)];
                nb = set_col(nb, c, nr);
                total += sc;
                if (nr != get_col(b, c)) moved = true;
            }
            break;
        case 's': // 下: 翻转列→左→翻转列
            for (int c = 0; c < 4; c++) {
                auto [nr, sc] = left_table[reverse_row(get_col(b, c))];
                nb = set_col(nb, c, reverse_row(nr));
                total += sc;
                if (nr != reverse_row(get_col(b, c))) moved = true;
            }
            break;
        default:
            return {b, 0, false};
    }
    return {nb, total, moved};
}

// ============================================================
// 4. 随机生成新方块
// ============================================================
Board spawn(Board b, std::mt19937 &rng) {
    int empties[16], cnt = 0;
    for (int i = 0; i < 16; i++)
        if (((b >> (i * 4)) & 0xF) == 0) empties[cnt++] = i;
    if (cnt == 0) return b;
    int pos = empties[std::uniform_int_distribution<>(0, cnt - 1)(rng)];
    int val = std::uniform_real_distribution<>(0, 1)(rng) < 0.9 ? 1 : 2; // log2(2)=1, log2(4)=2
    return b | (static_cast<uint64_t>(val) << (pos * 4));
}

// ============================================================
// 5. 辅助函数
// ============================================================
int count_empty(Board b) {
    int cnt = 0;
    for (int i = 0; i < 16; i++)
        if (((b >> (i * 4)) & 0xF) == 0) cnt++;
    return cnt;
}

int max_tile(Board b) {
    int mx = 0;
    for (int i = 0; i < 16; i++) {
        int v = (b >> (i * 4)) & 0xF;
        if (v > mx) mx = v;
    }
    return mx;
}

bool is_game_over(Board b) {
    for (int i = 0; i < 16; i++)
        if (((b >> (i * 4)) & 0xF) == 0) return false;
    // 水平相邻
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 3; c++)
            if (((b >> (r * 16 + c * 4)) & 0xF) ==
                ((b >> (r * 16 + c * 4 + 4)) & 0xF))
                return false;
    // 垂直相邻
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            if (((b >> (r * 16 + c * 4)) & 0xF) ==
                ((b >> ((r + 1) * 16 + c * 4)) & 0xF))
                return false;
    return true;
}

void print_board(Board b) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int v = (b >> (r * 16 + c * 4)) & 0xF;
            if (v == 0) std::cout << "   .";
            else std::cout << std::setw(4) << (1 << v);
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// ============================================================
// 6. 蛇形权重 (4 角变体)
// ============================================================
// 每个蛇形: 16 个权重, 对应 16 个棋盘位置
// 权重越高 → 大方块应该在该位置
static const int snake_variants[4][16] = {
    // 0: 左上角 (index 0)
    {15,14,13,12,  8, 9,10,11,  7, 6, 5, 4,  0, 1, 2, 3},
    // 1: 右上角 (index 3)
    {12,13,14,15, 11,10, 9, 8,  4, 5, 6, 7,  3, 2, 1, 0},
    // 2: 左下角 (index 12)
    { 0, 1, 2, 3,  7, 6, 5, 4,  8, 9,10,11, 15,14,13,12},
    // 3: 右下角 (index 15)
    { 3, 2, 1, 0,  4, 5, 6, 7, 11,10, 9, 8, 12,13,14,15},
};

// 相邻对预计算 (每个位置 → {右邻, 下邻})
static const int neighbors[16][2] = {
    {1,4},{2,5},{3,6},{-1,7},
    {5,8},{6,9},{7,10},{-1,11},
    {9,12},{10,13},{11,14},{-1,15},
    {13,-1},{14,-1},{15,-1},{-1,-1},
};

// ============================================================
// 7. 启发式评估
// ============================================================
inline int fast_log2(int v) {
    if (v == 0) return 0;
    return 31 - __builtin_clz(v); // GCC/Clang builtin
}

// RL 前向声明 (定义在 AI 段之后)
using WeightTable = std::unordered_map<std::string, double>;
extern WeightTable   rl_weights;
extern bool          rl_enabled;
extern double        rl_lr, rl_epsilon;
extern int           rl_step;
extern double        rl_td_ema;
extern int           rl_games, rl_best_score, rl_best_tile;
double rl_value(Board b);
void   rl_td_update(Board before, Board after, int reward);
bool   rl_should_explore(std::mt19937 &rng);
void   rl_save();
void   rl_load();

double evaluate(Board b) {
    int empty = 0, max_t = 0, max_idx = -1;
    for (int i = 0; i < 16; i++) {
        int v = (b >> (i * 4)) & 0xF;
        if (v == 0) empty++;
        else if (v > max_t) { max_t = v; max_idx = i; }
    }

    // 1) 空格 — 生存底线
    double score = empty * 35000.0;

    // 2) 多角蛇形 — 取 4 角最优
    double best_snake = 0;
    for (int si = 0; si < 4; si++) {
        double ss = 0;
        for (int i = 0; i < 16; i++) {
            int v = (b >> (i * 4)) & 0xF;
            if (v) ss += v * snake_variants[si][i];
        }
        if (ss > best_snake) best_snake = ss;
    }
    score += best_snake * 600.0;

    // 3) 角位置最大方块
    if (max_t > 0) {
        if (max_idx == 0 || max_idx == 3 || max_idx == 12 || max_idx == 15) {
            score += max_t * 6500.0;
        } else {
            int row = max_idx / 4, col = max_idx % 4;
            int edge_dist = std::min(row, 3 - row) + std::min(col, 3 - col);
            score -= (1.0 + edge_dist * 0.6) * max_t * 1200.0;
        }
    }

    // 4) 行单调性
    double mono = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            int a = (b >> (r * 16 + c * 4)) & 0xF;
            int bv = (b >> (r * 16 + (c + 1) * 4)) & 0xF;
            if (a && bv) {
                if (a >= bv) mono += a * 300.0;
                else         mono -= bv * 480.0;
            }
        }
    }
    score += mono;

    // 5) 列单调性
    double col_mono = 0;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            int a = (b >> (r * 16 + c * 4)) & 0xF;
            int bv = (b >> ((r + 1) * 16 + c * 4)) & 0xF;
            if (a && bv) {
                if (a >= bv) col_mono += a * 260.0;
                else         col_mono -= bv * 420.0;
            }
        }
    }
    score += col_mono;

    // 6) 平滑度
    double smooth = 0;
    for (int i = 0; i < 16; i++) {
        int v = (b >> (i * 4)) & 0xF;
        if (!v) continue;
        int nr = neighbors[i][0], nd = neighbors[i][1];
        if (nr >= 0) {
            int nv = (b >> (nr * 4)) & 0xF;
            if (nv) smooth += std::abs(v - nv);
        }
        if (nd >= 0) {
            int nv = (b >> (nd * 4)) & 0xF;
            if (nv) smooth += std::abs(v - nv);
        }
    }
    score -= smooth * 280.0;

    // 7) 角梯度 — 距角越近值越大 (取 4 角最优)
    double best_grad = 0;
    for (int ci = 0; ci < 4; ci++) {
        int corner = (ci == 0) ? 0 : (ci == 1) ? 3 : (ci == 2) ? 12 : 15;
        int cr = corner / 4, cc = corner % 4;
        double grad = 0;
        for (int i = 0; i < 16; i++) {
            int v = (b >> (i * 4)) & 0xF;
            if (!v) continue;
            int r = i / 4, c = i % 4;
            int dist = std::abs(r - cr) + std::abs(c - cc);
            // 离角越近-值越大-得分越高; 违反递减则扣分
            grad += v * (8.0 - dist) * 500.0;
        }
        if (grad > best_grad) best_grad = grad;
    }
    score += best_grad;

    // 8) 边缘偏好
    double merge_bonus = 0;
    for (int i = 0; i < 16; i++) {
        int v = (b >> (i * 4)) & 0xF;
        if (!v) continue;
        int nr = neighbors[i][0], nd = neighbors[i][1];
        if (nr >= 0 && static_cast<int>((b >> (nr * 4)) & 0xF) == v)
            merge_bonus += 1 << v;
        if (nd >= 0 && static_cast<int>((b >> (nd * 4)) & 0xF) == v)
            merge_bonus += 1 << v;
    }
    score += merge_bonus * 60.0;

    // 9) 边缘偏好
    double edge_bonus = 0;
    for (int i = 0; i < 16; i++) {
        int v = (b >> (i * 4)) & 0xF;
        if (v >= 7) { // 128+, log2(128)=7
            int row = i / 4, col = i % 4;
            if (row == 0 || row == 3 || col == 0 || col == 3)
                edge_bonus += v * 150.0;
        }
    }
    score += edge_bonus;

    // RL 做主评估: 启发式随训练从 100% 逐步退出到 ~5%
    if (rl_enabled) {
        double h_weight = std::exp(-rl_step / 200000.0);  // 20万步后退到 37%
        double rl_v = rl_value(b);
        score = score * h_weight + rl_v * 50000.0;
    }

    return score;
}

// ============================================================
// 8. 置换表 + Expectimax
// ============================================================
static std::unordered_map<uint64_t, double> transpos; // key = board | (depth << 60) | (is_max << 63)
static int transpos_hit = 0, transpos_miss = 0;

inline uint64_t make_key(Board b, int depth, bool is_max) {
    return b | (static_cast<uint64_t>(depth) << 60) |
           (static_cast<uint64_t>(is_max ? 1 : 0) << 63);
}

// 前向声明
double expectimax(Board b, int depth, bool is_max, std::mt19937 &rng);

// 获取空格列表
int get_empties(Board b, int *buf) {
    int cnt = 0;
    for (int i = 0; i < 16; i++)
        if (((b >> (i * 4)) & 0xF) == 0) buf[cnt++] = i;
    return cnt;
}

double expectimax(Board b, int depth, bool is_max, std::mt19937 &rng) {
    if (depth == 0) return evaluate(b);
    if (is_game_over(b)) return -1e12;

    // 置换表
    auto key = make_key(b, depth, is_max);
    auto it = transpos.find(key);
    if (it != transpos.end()) { transpos_hit++; return it->second; }
    transpos_miss++;

    if (is_max) {
        double best = -1e18;
        for (char d : {'a', 'd', 'w', 's'}) {
            auto [nb, gained, moved] = do_move(b, d);
            if (!moved) continue;
            double v = gained * 300.0 + expectimax(nb, depth - 1, false, rng);
            if (v > best) best = v;
        }
        if (best == -1e18) best = evaluate(b);
        transpos[key] = best;
        return best;
    } else {
        int empties[16];
        int n = get_empties(b, empties);
        if (n == 0) {
            double v = evaluate(b);
            transpos[key] = v;
            return v;
        }

        // 精确评估 (≤4 空格) vs 加权采样 (深度大, 必须限制分支)
        int sample_n;
        bool exact = (n <= 4);
        if (exact) {
            sample_n = n;
        } else {
            sample_n = (n > 12) ? 5 : (n > 8) ? 6 : 7;
            // 加权采样: 空格近高值方块优先
            double weights[16], total_w = 0;
            for (int i = 0; i < n; i++) {
                int cell = empties[i];
                double w = 1.0;
                int nr = neighbors[cell][0], nd = neighbors[cell][1];
                if (nr >= 0) { int nv = (b >> (nr * 4)) & 0xF; if (nv) w += nv; }
                if (nd >= 0) { int nv = (b >> (nd * 4)) & 0xF; if (nv) w += nv; }
                weights[i] = w;
                total_w += w;
            }
            // 加权随机选择
            int sampled[8];
            for (int k = 0; k < sample_n; k++) {
                double r = std::uniform_real_distribution<>(0, total_w)(rng);
                double cum = 0;
                int pick = 0;
                for (int i = 0; i < n; i++) {
                    cum += weights[i];
                    if (r <= cum) { pick = i; break; }
                    if (i == n - 1) pick = i;
                }
                sampled[k] = empties[pick];
                total_w -= weights[pick];
                weights[pick] = 0;
            }
            // 复制到 empties
            for (int k = 0; k < sample_n; k++) empties[k] = sampled[k];
        }

        double total = 0;
        for (int k = 0; k < sample_n; k++) {
            int cell = empties[k];
            // 2-生成 (90%)
            Board b2 = b | (1ULL << (cell * 4));
            total += 0.9 * expectimax(b2, depth - 1, true, rng);
            // 4-生成 (10%)
            Board b4 = b | (2ULL << (cell * 4));
            total += 0.1 * expectimax(b4, depth - 1, true, rng);
        }

        double result = total / sample_n;
        transpos[key] = result;
        return result;
    }
}

// ============================================================
// 9. AI 最佳移动选择
// ============================================================
struct ScoredMove {
    char dir;
    Board board;
    int gained;
    double score;
};

char ai_best_move(Board b, std::mt19937 &rng, bool allow_explore = false) {
    // RL 探索: 随机选合法方向
    if (allow_explore && rl_enabled && rl_should_explore(rng)) {
        char valid[4]; int cnt = 0;
        for (char d : {'a','d','w','s'})
            if (do_move(b, d).moved) valid[cnt++] = d;
        if (cnt) return valid[std::uniform_int_distribution<>(0,cnt-1)(rng)];
    }

    transpos.clear();
    transpos_hit = transpos_miss = 0;

    int empty = count_empty(b);

    int depth;
    if (empty >= 10)      depth = 5;
    else if (empty >= 6)  depth = 6;
    else if (empty >= 3)  depth = 7;
    else if (empty >= 1)  depth = 8;
    else                  depth = 1;

    // 收集合法移动
    ScoredMove moves[4];
    int move_cnt = 0;
    for (char d : {'a', 'd', 'w', 's'}) {
        auto [nb, gained, moved] = do_move(b, d);
        if (moved) {
            moves[move_cnt++] = {d, nb, gained, 0};
        }
    }
    if (move_cnt == 0) return '?';

    // depth-1 预排序
    for (int i = 0; i < move_cnt; i++) {
        moves[i].score = moves[i].gained * 300.0 +
                         expectimax(moves[i].board, 1, true, rng);
    }
    std::sort(moves, moves + move_cnt,
              [](auto &a, auto &b) { return a.score > b.score; });

    // 深度搜索
    char best_d = moves[0].dir;
    double best_v = -1e18;
    for (int i = 0; i < move_cnt; i++) {
        double val = moves[i].gained * 300.0 +
                     expectimax(moves[i].board, depth, true, rng);
        if (val > best_v) {
            best_v = val;
            best_d = moves[i].dir;
        }
    }
    return best_d;
}

// ============================================================
// 10. N-Tuple 强化学习 (TD(0) + ε-greedy)
// ============================================================
WeightTable   rl_weights;
bool          rl_enabled   = false;
double        rl_lr        = 0.1;
double        rl_epsilon   = 0.1;
int           rl_step      = 0;
double        rl_td_ema    = 0.0;
int           rl_games     = 0;
int           rl_best_score = 0;
int           rl_best_tile  = 0;

// ---- N-Tuple 特征提取 ----
static std::string canonical(const std::array<int,4> &t) {
    std::array<int,4> rev = {t[3],t[2],t[1],t[0]};
    auto key = t <= rev ? t : rev;
    char buf[64];
    snprintf(buf, sizeof(buf), "%d,%d,%d,%d", key[0],key[1],key[2],key[3]);
    return buf;
}

static std::string canonical6(const std::array<int,6> &t) {
    std::array<int,6> rev = {t[5],t[4],t[3],t[2],t[1],t[0]};
    auto key = t <= rev ? t : rev;
    char buf[80];
    snprintf(buf, sizeof(buf), "%d,%d,%d,%d,%d,%d",
             key[0],key[1],key[2],key[3],key[4],key[5]);
    return buf;
}

static void extract_ntuples(Board b, std::vector<std::pair<int,std::string>> &out) {
    int log[16];
    for (int i = 0; i < 16; i++) log[i] = (b >> (i*4)) & 0xF;

    // 4 行 (pid 0-3)
    for (int i = 0; i < 4; i++) {
        std::array<int,4> t = {log[i*4],log[i*4+1],log[i*4+2],log[i*4+3]};
        bool any = false; for (int v : t) if (v) any = true;
        if (any) out.emplace_back(i, canonical(t));
    }
    // 4 列 (pid 4-7)
    for (int j = 0; j < 4; j++) {
        std::array<int,4> t = {log[j],log[4+j],log[8+j],log[12+j]};
        bool any = false; for (int v : t) if (v) any = true;
        if (any) out.emplace_back(4+j, canonical(t));
    }
    // 9 个 2x2 (pid 8-16)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i*4+j;
            std::array<int,4> t = {log[idx],log[idx+1],log[idx+4],log[idx+5]};
            bool any = false; for (int v : t) if (v) any = true;
            if (any) out.emplace_back(8+i*3+j, canonical(t));
        }
    }
    // 6 个 2x3 矩形 (pid 17-22) — 捕捉更大范围的角构建模式
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i*4+j;
            std::array<int,6> t = {log[idx],log[idx+1],log[idx+4],log[idx+5],log[idx+8],log[idx+9]};
            bool any = false; for (int v : t) if (v) any = true;
            if (any) out.emplace_back(17+i*3+j, canonical6(t));
        }
    }
    // 6 个 3x2 矩形 (pid 23-28)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            int idx = i*4+j;
            std::array<int,6> t = {log[idx],log[idx+1],log[idx+2],log[idx+4],log[idx+5],log[idx+6]};
            bool any = false; for (int v : t) if (v) any = true;
            if (any) out.emplace_back(23+i*2+j, canonical6(t));
        }
    }
}

// ---- RL 评估 ----
double rl_value(Board b) {
    thread_local std::vector<std::pair<int,std::string>> feats; feats.clear();
    extract_ntuples(b, feats);
    double v = 0;
    for (auto &[pid, key] : feats) {
        auto it = rl_weights.find(std::to_string(pid) + ":" + key);
        if (it != rl_weights.end()) v += it->second;
    }
    return v;
}

// ---- TD(0) 更新 ----
void rl_td_update(Board before, Board after, int reward) {
    double v_cur = rl_value(before);
    double v_nxt = after ? rl_value(after) : 0;
    double td = reward + v_nxt - v_cur;

    thread_local std::vector<std::pair<int,std::string>> feats; feats.clear();
    extract_ntuples(before, feats);
    if (!feats.empty()) {
        double delta = rl_lr * td / feats.size();
        for (auto &[pid, key] : feats)
            rl_weights[std::to_string(pid) + ":" + key] += delta;
    }

    rl_step++;
    rl_lr     = std::max(0.001, rl_lr * 0.9995);
    rl_epsilon = std::max(0.0, rl_epsilon * 0.9995);
    rl_td_ema = 0.95 * rl_td_ema + 0.05 * fabs(td);
}

bool rl_should_explore(std::mt19937 &rng) {
    return std::uniform_real_distribution<>(0,1)(rng) < rl_epsilon;
}

// ---- 持久化 ----
const char *RL_FILE = "rl_weights.json";

void rl_save() {
    std::ofstream f(RL_FILE);
    if (!f) return;
    f << "{\"step\":" << rl_step
      << ",\"lr\":" << rl_lr
      << ",\"epsilon\":" << rl_epsilon
      << ",\"games\":" << rl_games
      << ",\"best_tile\":" << rl_best_tile
      << ",\"best_score\":" << rl_best_score
      << ",\"weights\":{";
    bool first = true;
    for (auto &[k, v] : rl_weights) {
        if (!first) f << ",";
        first = false;
        f << "\"" << k << "\":" << v;
    }
    f << "}}";
}

void rl_load() {
    std::ifstream f(RL_FILE);
    if (!f) return;
    try {
        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        auto find_int = [&](const char *key) -> int {
            auto p = json.find(std::string("\"") + key + "\":");
            if (p == std::string::npos) return 0;
            try { return std::stoi(json.substr(p + strlen(key) + 4)); }
            catch (...) { return 0; }
        };
        auto find_dbl = [&](const char *key) -> double {
            auto p = json.find(std::string("\"") + key + "\":");
            if (p == std::string::npos) return 0;
            try { return std::stod(json.substr(p + strlen(key) + 4)); }
            catch (...) { return 0; }
        };
        rl_step    = find_int("step");
        rl_games   = find_int("games");
        rl_best_tile  = find_int("best_tile");
        rl_best_score = find_int("best_score");
        rl_lr      = find_dbl("lr");
        rl_epsilon = find_dbl("epsilon");

        auto wp = json.find("\"weights\":{");
        if (wp == std::string::npos) return;
        wp += 11;
        rl_weights.clear();
        while (wp < json.size() && json[wp] != '}') {
            if (json[wp] == ',' || json[wp] == '\n') { wp++; continue; }
            wp++;
            auto eq = json.find("\":", wp);
            if (eq == std::string::npos) break;
            std::string key = json.substr(wp, eq - wp);
            auto comma = json.find_first_of(",}", eq + 2);
            if (comma == std::string::npos) break;
            try {
                double val = std::stod(json.substr(eq + 2, comma - eq - 2));
                rl_weights[key] = val;
            } catch (...) {}
            wp = comma;
        }
    } catch (...) {
        // 文件损坏，删除后重来
        f.close();
        remove(RL_FILE);
    }
}

void rl_reset() {
    rl_weights.clear();
    rl_lr = 0.1; rl_epsilon = 0.1;
    rl_step = 0; rl_td_ema = 0;
    rl_games = 0; rl_best_score = 0; rl_best_tile = 0;
}

// ============================================================
// 11. 单局游戏
// ============================================================
struct GameResult {
    int score;
    int max_tile;
    int moves;
    double seconds;
};

GameResult play_game(std::mt19937 &rng, bool verbose = false) {
    Board b = 0;
    b = spawn(b, rng); // 初始生成 1 个方块
    int score = 0, moves = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (true) {
        if (is_game_over(b)) break;

        char best = ai_best_move(b, rng);
        if (best == '?') break;

        auto [nb, gained, moved] = do_move(b, best);
        if (!moved) continue; // 不应该发生

        b = spawn(nb, rng);
        score += gained;
        moves++;

        if (verbose) {
            std::cout << "Move " << moves << ": " << best
                      << " gain=" << gained << " score=" << score
                      << " max=" << (1 << max_tile(b)) << "\n";
            print_board(b);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    return {score, max_tile(b), moves, secs};
}

// ============================================================
// 11. 主程序
// ============================================================
void print_usage() {
    std::cout << "Usage: 2048 [--play N] [--bench N] [--verbose]\n"
              << "  --play N    自动玩 N 局 (默认 1)\n"
              << "  --bench N   基准测试 N 局, 输出统计\n"
              << "  --gui       启动可视化窗口 (Windows)\n"
              << "  --verbose   显示每步棋盘\n";
}

// ============================================================
// 12. Win32 可视化 GUI
// ============================================================
#ifdef _WIN32
#include <windows.h>

// ---- 全局状态 ----
static Board   g_board     = 0;
static int     g_score     = 0;
static int     g_high      = 0;
static const char *HS_FILE = "highscore.txt";

static void load_high() {
    std::ifstream f(HS_FILE);
    if (f) f >> g_high;
}
static void save_high() {
    std::ofstream f(HS_FILE);
    if (f) f << g_high;
}
static bool    g_ai        = false;
static bool    g_over      = false;
static Board   g_history[5];
static int     g_hist_score[5];
static int     g_hist_len  = 0;
static HWND    g_hwnd      = nullptr;
static UINT_PTR g_timer    = 0;
static std::mt19937 g_rng(static_cast<unsigned>(
    std::chrono::system_clock::now().time_since_epoch().count()));

// ---- 颜色 ----
static const COLORREF C_BG       = RGB(0xFA,0xF8,0xEF);
static const COLORREF C_BOARD    = RGB(0xBB,0xAD,0xA0);
static const COLORREF C_EMPTY    = RGB(0xCD,0xC1,0xB4);
static const COLORREF C_DARK     = RGB(0x77,0x6E,0x65);
static const COLORREF C_LIGHT    = RGB(0xF9,0xF6,0xF2);
static const COLORREF C_BTN      = RGB(0x8F,0x7A,0x66);
static const COLORREF C_GOLD     = RGB(0xED,0xC2,0x2E);
static const COLORREF C_OVERLAY  = RGB(0xFF,0xFF,0xFF);

static COLORREF tile_color(int log2v) {
    switch (log2v) {
        case 0:  return C_EMPTY;
        case 1:  return RGB(0xEE,0xE4,0xDA); // 2
        case 2:  return RGB(0xED,0xE0,0xC8); // 4
        case 3:  return RGB(0xF2,0xB1,0x79); // 8
        case 4:  return RGB(0xF5,0x95,0x63); // 16
        case 5:  return RGB(0xF6,0x7C,0x5F); // 32
        case 6:  return RGB(0xF6,0x5E,0x3B); // 64
        case 7:  return RGB(0xED,0xCF,0x72); // 128
        case 8:  return RGB(0xED,0xCC,0x61); // 256
        case 9:  return RGB(0xED,0xC8,0x50); // 512
        case 10: return RGB(0xED,0xC5,0x3F); // 1024
        case 11: return RGB(0xED,0xC2,0x2E); // 2048
        default: return RGB(0x3C,0x3A,0x32); // 4096+
    }
}

static COLORREF text_color(int log2v) {
    return (log2v <= 2) ? C_DARK : C_LIGHT;
}

static int font_size(int log2v) {
    if (log2v <= 6) return 34;
    if (log2v <= 9) return 30;
    if (log2v <= 12) return 24;
    return 20;
}

// ---- 布局常量 ----
const int WIN_W = 500, WIN_H = 640;
const int TILE_SZ = 90, GAP = 12;
const int BOARD_PAD = GAP;
const int BOARD_W = 4 * TILE_SZ + 5 * GAP;
const int BOARD_X = (WIN_W - BOARD_W) / 2, BOARD_Y = 175;
const int CARD_W = 180, CARD_H = 55, CARD_Y = 100;
const int BTN_Y = 600;

// ---- 辅助 ----
static void push_hist() {
    if (g_hist_len < 5) g_hist_len++;
    for (int i = g_hist_len - 1; i > 0; i--) {
        g_history[i] = g_history[i-1];
        g_hist_score[i] = g_hist_score[i-1];
    }
    g_history[0] = g_board;
    g_hist_score[0] = g_score;
}

static void pop_hist() {
    if (g_hist_len > 0) {
        g_board = g_history[0];
        g_score = g_hist_score[0];
        for (int i = 0; i < g_hist_len - 1; i++) {
            g_history[i] = g_history[i+1];
            g_hist_score[i] = g_hist_score[i+1];
        }
        g_hist_len--;
        g_over = false;
    }
}

static void new_game() {
    if (rl_enabled) rl_save();
    g_board = 0;
    g_board = spawn(g_board, g_rng);
    g_score = 0;
    g_hist_len = 0;
    g_over = false;
}

static void do_ai_move() {
    if (g_over || is_game_over(g_board)) {
        if (rl_enabled) {
            // 自动重启训练
            rl_games++;
            int tile = 1 << max_tile(g_board);
            if (g_score > rl_best_score) rl_best_score = g_score;
            if (tile > rl_best_tile) rl_best_tile = tile;
            rl_save();
            new_game();
            InvalidateRect(g_hwnd, nullptr, TRUE);
            return;
        }
        g_over = true;
        KillTimer(g_hwnd, g_timer);
        g_timer = 0;
        InvalidateRect(g_hwnd, nullptr, TRUE);
        return;
    }
    char best = ai_best_move(g_board, g_rng, true);
    if (best == '?') return;
    Board before = g_board;
    auto [nb, gained, moved] = do_move(g_board, best);
    if (moved) {
        push_hist();
        g_board = spawn(nb, g_rng);
        g_score += gained;
        if (g_score > g_high) { g_high = g_score; save_high(); }

        // RL TD 更新
        if (rl_enabled) rl_td_update(before, g_board, gained);
    }
    if (is_game_over(g_board)) {
        if (rl_enabled) {
            rl_games++;
            int tile = 1 << max_tile(g_board);
            if (g_score > rl_best_score) rl_best_score = g_score;
            if (tile > rl_best_tile) rl_best_tile = tile;
            rl_save();
            new_game();
        } else {
            g_over = true;
        }
    }
    InvalidateRect(g_hwnd, nullptr, TRUE);
}

// ---- 绘制 ----
static void draw_all(HDC hdc) {
    // 双缓冲
    HDC mem = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, WIN_W, WIN_H);
    SelectObject(mem, bmp);

    // 背景
    HBRUSH bg = CreateSolidBrush(C_BG);
    RECT rc = {0, 0, WIN_W, WIN_H};
    FillRect(mem, &rc, bg);
    DeleteObject(bg);

    // 辅助: 创建字体 (宽字符版)
    auto make_font = [](int h, int wt, const wchar_t *face) {
        HFONT f = CreateFontW(h, 0, 0, 0, wt, 0, 0, 0,
                              DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                              CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                              DEFAULT_PITCH | FF_DONTCARE, face);
        if (!f)
            f = CreateFontW(h, 0, 0, 0, wt, 0, 0, 0,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                            CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                            DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
        return f;
    };

    // 标题
    HFONT title_font = make_font(42, FW_BOLD, L"Arial");
    SelectObject(mem, title_font);
    SetTextColor(mem, C_DARK);
    SetBkMode(mem, TRANSPARENT);
    RECT tr = {0, 5, WIN_W, 55};
    DrawTextW(mem, L"2048", -1, &tr, DT_CENTER | DT_VCENTER);
    DeleteObject(title_font);

    HFONT small = make_font(14, FW_NORMAL, L"Microsoft YaHei");
    SelectObject(mem, small);
    SetTextColor(mem, RGB(0xA0,0x94,0x8A));
    RECT sr = {0, 55, WIN_W, 80};
    DrawTextW(mem, L"加入数字, 拼出 2048 方块!", -1, &sr, DT_CENTER);
    DeleteObject(small);

    // 分数卡片
    auto draw_card = [&](int x, int y, const wchar_t *label, int val) {
        HBRUSH cb = CreateSolidBrush(RGB(0xBB,0xAD,0xA0));
        RECT cr = {x, y, x + CARD_W, y + CARD_H};
        FillRect(mem, &cr, cb);
        DeleteObject(cb);
        HFONT lf = make_font(15, FW_BOLD, L"Microsoft YaHei");
        SelectObject(mem, lf);
        SetTextColor(mem, RGB(0xEE,0xE4,0xDA));
        RECT lr = {x, y + 2, x + CARD_W, y + 22};
        DrawTextW(mem, label, -1, &lr, DT_CENTER);
        DeleteObject(lf);
        HFONT vf = make_font(28, FW_BOLD, L"Arial");
        SelectObject(mem, vf);
        SetTextColor(mem, RGB(255,255,255));
        wchar_t buf[32];
        swprintf(buf, 32, L"%d", val);
        RECT vr = {x, y + 18, x + CARD_W, y + CARD_H};
        DrawTextW(mem, buf, -1, &vr, DT_CENTER);
        DeleteObject(vf);
    };
    draw_card(50,  CARD_Y, L"分数", g_score);
    draw_card(270, CARD_Y, L"最高分", g_high);

    // AI 状态
    if (g_ai) {
        HFONT af = make_font(13, FW_BOLD, L"Microsoft YaHei");
        SelectObject(mem, af);
        SetTextColor(mem, rl_enabled ? RGB(0x3C,0xB3,0x71) : RGB(0x8F,0x7A,0x66));
        RECT ar = {0, CARD_Y + CARD_H + 5, WIN_W, CARD_Y + CARD_H + 22};
        DrawTextW(mem, rl_enabled ? L"🧠 RL 学习中..." : L"🤖 AI 自动对弈中...", -1, &ar, DT_CENTER);
        DeleteObject(af);
    }

    // 棋盘背景
    HBRUSH board_bg = CreateSolidBrush(C_BOARD);
    RECT br = {BOARD_X, BOARD_Y, BOARD_X + BOARD_W, BOARD_Y + BOARD_W};
    FillRect(mem, &br, board_bg);
    DeleteObject(board_bg);

    // 方块
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int x = BOARD_X + BOARD_PAD + c * (TILE_SZ + GAP);
            int y = BOARD_Y + BOARD_PAD + r * (TILE_SZ + GAP);
            int v = (g_board >> (r * 16 + c * 4)) & 0xF;

            HBRUSH tb = CreateSolidBrush(tile_color(v));
            RECT trc = {x, y, x + TILE_SZ, y + TILE_SZ};
            FillRect(mem, &trc, tb);
            DeleteObject(tb);

            if (v != 0) {
                int fs = font_size(v);
                HFONT tf = make_font(fs, FW_BOLD, L"Arial");
                SelectObject(mem, tf);
                SetTextColor(mem, text_color(v));
                wchar_t num[16];
                swprintf(num, 16, L"%d", 1 << v);
                RECT nr = {x, y, x + TILE_SZ, y + TILE_SZ};
                DrawTextW(mem, num, -1, &nr, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
                DeleteObject(tf);
            }
        }
    }

    // Game Over 遮罩
    if (g_over) {
        HBRUSH ov = CreateSolidBrush(C_OVERLAY);
        RECT orc = {BOARD_X, BOARD_Y, BOARD_X + BOARD_W, BOARD_Y + BOARD_W};
        SetBkMode(mem, OPAQUE);
        SetBkColor(mem, C_OVERLAY);
        FillRect(mem, &orc, ov);
        DeleteObject(ov);

        HFONT gof = make_font(38, FW_BOLD, L"Arial");
        SelectObject(mem, gof);
        SetTextColor(mem, C_DARK);
        SetBkMode(mem, TRANSPARENT);
        RECT gor = {BOARD_X, BOARD_Y + BOARD_W/2 - 30, BOARD_X + BOARD_W, BOARD_Y + BOARD_W/2 + 10};
        DrawTextW(mem, L"游戏结束", -1, &gor, DT_CENTER);
        DeleteObject(gof);

        HFONT scf = make_font(18, FW_BOLD, L"Microsoft YaHei");
        SelectObject(mem, scf);
        SetTextColor(mem, RGB(0x8F,0x7A,0x66));
        wchar_t buf[64];
        swprintf(buf, 64, L"最终得分: %d", g_score);
        RECT scr = {BOARD_X, BOARD_Y + BOARD_W/2 + 15, BOARD_X + BOARD_W, BOARD_Y + BOARD_W/2 + 40};
        DrawTextW(mem, buf, -1, &scr, DT_CENTER);
        DeleteObject(scf);
    }

    // 按钮
    auto draw_btn = [&](int x, int w, const wchar_t *text, COLORREF color) {
        HBRUSH bb = CreateSolidBrush(color);
        RECT btr = {x, BTN_Y, x + w, BTN_Y + 35};
        FillRect(mem, &btr, bb);
        DeleteObject(bb);
        HFONT bf = make_font(15, FW_BOLD, L"Microsoft YaHei");
        SelectObject(mem, bf);
        SetTextColor(mem, RGB(255,255,255));
        SetBkMode(mem, TRANSPARENT);
        DrawTextW(mem, text, -1, &btr, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        DeleteObject(bf);
    };
    draw_btn(35,  90, L"新游戏", C_BTN);
    draw_btn(140, 90, g_hist_len ? L"撤回" : L"---", g_hist_len ? C_BTN : RGB(0xCD,0xC1,0xB4));
    draw_btn(245, 110, g_ai ? L"停止AI" : L"AI对弈", g_ai ? C_GOLD : C_BTN);
    draw_btn(370, 95, rl_enabled ? L"RL训练中" : L"RL训练",
             rl_enabled ? RGB(0x3C,0xB3,0x71) : C_BTN);

    // RL 训练状态 (大字，单独一行)
    if (rl_enabled) {
        wchar_t rbuf[128];
        swprintf(rbuf, 128, L"局%d | 最高分 %d | 最大方块 %d | 步 %d | eps=%.4f lr=%.4f",
                 rl_games + 1, rl_best_score, rl_best_tile, rl_step, rl_epsilon, rl_lr);
        // 背景卡片
        HBRUSH rb = CreateSolidBrush(RGB(0xF5,0xF2,0xEB));
        RECT rbr = {20, BTN_Y - 36, WIN_W - 20, BTN_Y - 4};
        FillRect(mem, &rbr, rb);
        DeleteObject(rb);
        // 文字
        HFONT rf = make_font(14, FW_BOLD, L"Microsoft YaHei");
        SelectObject(mem, rf);
        SetTextColor(mem, C_DARK);
        SetBkMode(mem, TRANSPARENT);
        RECT rr = {25, BTN_Y - 36, WIN_W - 25, BTN_Y - 4};
        DrawTextW(mem, rbuf, -1, &rr, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        DeleteObject(rf);
    }

    // 复制到屏幕
    BitBlt(hdc, 0, 0, WIN_W, WIN_H, mem, 0, 0, SRCCOPY);
    DeleteDC(mem);
    DeleteObject(bmp);
}

// ---- 玩家移动 ----
static void player_move(char dir) {
    if (g_over || g_ai) return;
    Board before = g_board;
    auto [nb, gained, moved] = do_move(g_board, dir);
    if (moved) {
        push_hist();
        g_board = spawn(nb, g_rng);
        g_score += gained;
        if (g_score > g_high) { g_high = g_score; save_high(); }
        // 手动玩也贡献 RL 学习
        if (rl_enabled) rl_td_update(before, g_board, gained);
        if (is_game_over(g_board)) {
            if (rl_enabled) {
                rl_games++;
                int tile = 1 << max_tile(g_board);
                if (g_score > rl_best_score) rl_best_score = g_score;
                if (tile > rl_best_tile) rl_best_tile = tile;
                rl_save();
                new_game();
            } else {
                g_over = true;
            }
        }
        InvalidateRect(g_hwnd, nullptr, TRUE);
    }
}

// ---- 窗口过程 ----
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE:
            g_hwnd = hwnd;
            load_high();
            rl_load();
            new_game();
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            draw_all(hdc);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_KEYDOWN:
            switch (wp) {
                case VK_LEFT:  case 'A': player_move('a'); break;
                case VK_RIGHT: case 'D': player_move('d'); break;
                case VK_UP:    case 'W': player_move('w'); break;
                case VK_DOWN:  case 'S': player_move('s'); break;
                case VK_F2: new_game(); InvalidateRect(hwnd, nullptr, TRUE); break;
                case 'Z': if (GetKeyState(VK_CONTROL) & 0x8000) {
                    pop_hist(); InvalidateRect(hwnd, nullptr, TRUE);
                } break;
            }
            return 0;
        case WM_LBUTTONDOWN: {
            int x = LOWORD(lp), y = HIWORD(lp);
            if (y >= BTN_Y && y <= BTN_Y + 35) {
                if (x >= 35 && x <= 125) {            // New Game
                    if (g_ai) { KillTimer(hwnd, g_timer); g_timer = 0; g_ai = false; }
                    new_game();
                } else if (x >= 140 && x <= 230) {     // Undo
                    if (!g_ai) pop_hist();
                } else if (x >= 245 && x <= 355) {     // AI Toggle
                    g_ai = !g_ai;
                    if (g_timer) { KillTimer(hwnd, g_timer); g_timer = 0; }
                    if (g_ai) {
                        g_timer = SetTimer(hwnd, 1, 60, nullptr);
                    }
                } else if (x >= 370 && x <= 465) {     // RL Toggle
                    rl_enabled = !rl_enabled;
                    if (rl_enabled) {
                        rl_load();
                        rl_td_ema = 0;
                    } else {
                        rl_save();
                    }
                }
            }
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }
        case WM_TIMER:
            if (wp == 1 && g_ai) do_ai_move();
            return 0;
        case WM_DESTROY:
            if (g_timer) KillTimer(hwnd, g_timer);
            if (rl_enabled) rl_save();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

static int gui_main(HINSTANCE hInst) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"Game2048";
    RegisterClassW(&wc);

    RECT wr = {0, 0, WIN_W, WIN_H};
    AdjustWindowRect(&wr, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);

    HWND hwnd = CreateWindowW(L"Game2048", L"2048",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        nullptr, nullptr, hInst, nullptr);

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}
#endif // _WIN32

int main(int argc, char **argv) {
    init_move_table();

#ifdef _WIN32
    // 无参数时默认启动 GUI, --gui 也可以
    bool has_cli_arg = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--play") == 0 || strcmp(argv[i], "--bench") == 0 ||
            strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "--verbose") == 0) {
            has_cli_arg = true;
            break;
        }
    }
    if (!has_cli_arg) {
        return gui_main(GetModuleHandle(nullptr));
    }
#endif

    int num_games = 1;
    bool bench_mode = false;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--play") == 0 && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench_mode = true;
            num_games = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    if (bench_mode) {
        std::cout << "=== 2048 C++ AI Benchmark ===\n"
                  << "Games: " << num_games << "\n\n";

        std::vector<GameResult> results;
        double best_score = 0;
        int best_tile = 0;
        int worst_tile = 999999;
        double total_score = 0, total_time = 0;
        int total_moves = 0;

        for (int g = 0; g < num_games; g++) {
            std::mt19937 game_rng(rd());
            auto result = play_game(game_rng, verbose);
            results.push_back(result);

            total_score += result.score;
            total_time += result.seconds;
            total_moves += result.moves;
            if (result.score > best_score) best_score = result.score;
            if ((1 << result.max_tile) > best_tile) best_tile = 1 << result.max_tile;
            if ((1 << result.max_tile) < worst_tile) worst_tile = 1 << result.max_tile;

            std::cout << "  Game " << std::setw(3) << (g + 1)
                      << ": score=" << std::setw(8) << result.score
                      << "  max=" << std::setw(6) << (1 << result.max_tile)
                      << "  moves=" << std::setw(5) << result.moves
                      << "  " << std::fixed << std::setprecision(1)
                      << result.seconds << "s\n";
        }

        double avg_score = total_score / num_games;
        double avg_time = total_time / num_games;
        double avg_moves = static_cast<double>(total_moves) / num_games;
        double avg_ms_per_move = total_time / total_moves * 1000;

        std::cout << "\n--- Summary ---\n"
                  << "Average score:    " << std::fixed << std::setprecision(0) << avg_score << "\n"
                  << "Best score:       " << best_score << "\n"
                  << "Best tile:        " << best_tile << "\n"
                  << "Average moves:    " << std::fixed << std::setprecision(0) << avg_moves << "\n"
                  << "Average time:     " << std::fixed << std::setprecision(1) << avg_time << "s\n"
                  << "Avg ms/move:      " << std::fixed << std::setprecision(1) << avg_ms_per_move << "ms\n";

        if (best_tile >= 2048)
            std::cout << "\n*** Reached 2048! ***\n";
        else if (best_tile >= 1024)
            std::cout << "\nReached " << best_tile << ", need deeper search for 2048\n";
    } else {
        std::cout << "=== 2048 C++ AI — Auto Play (" << num_games
                  << " game" << (num_games > 1 ? "s" : "") << ") ===\n\n";

        for (int g = 0; g < num_games; g++) {
            std::mt19937 game_rng(rd());
            auto result = play_game(game_rng, verbose);
            std::cout << "Game " << (g + 1) << ": score=" << result.score
                      << "  max=" << (1 << result.max_tile)
                      << "  moves=" << result.moves
                      << "  " << std::fixed << std::setprecision(1)
                      << result.seconds << "s\n";
        }
    }

    return 0;
}
