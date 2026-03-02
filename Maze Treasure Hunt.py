import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pygame_menu
import time


# 游戏环境
class MazeGame:


    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

        # 定义游戏元素
        self.EMPTY = 0
        self.WALL = 1
        self.TREASURE = 2
        self.TRAP = 3
        self.PLAYER = 4

        # 创建迷宫
        self.maze = np.zeros((height, width), dtype=int)
        self._create_maze()

        # 初始位置
        self.start_pos = (0, 0)
        self.player_pos = self.start_pos
        self.treasure_pos = self._find_treasure()

        # 动作空间: 0:上, 1:下, 2:左, 3:右
        self.actions = [0, 1, 2, 3]
        self.action_names = ['Up', 'down', 'left', 'right']

        # 游戏状态
        self.game_over = False
        self.total_reward = 0
        self.steps = 0


    def _create_maze(self):

        # 设置墙壁
        walls = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2),
                 (5, 5), (5, 6), (6, 5), (6, 6),
                 (7, 7), (7, 8), (8, 7), (8, 8)]
        for w in walls:
            if w[0] < self.height and w[1] < self.width:
                self.maze[w] = self.WALL

        # 设置宝藏
        treasure_pos = (8, 8)
        if treasure_pos[0] < self.height and treasure_pos[1] < self.width:
            self.maze[treasure_pos] = self.TREASURE

        # 设置陷阱
        traps = [(3, 7), (4, 7), (7, 3), (7, 4)]
        for t in traps:
            if t[0] < self.height and t[1] < self.width:
                self.maze[t] = self.TRAP

    def _find_treasure(self):

        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == self.TREASURE:
                    return (i, j)
        return (self.height - 1, self.width - 1)

    def reset(self):

        self.player_pos = self.start_pos
        self.game_over = False
        self.total_reward = 0
        self.steps = 0
        return self._get_state()

    def _get_state(self):

        return self.player_pos

    def step(self, action):


        if self.game_over:
            print("Game Over，Please press R to reset")
            return self.player_pos, 0, True  # 直接返回，不执行动作
        self.steps += 1

        # 计算新位置
        new_pos = list(self.player_pos)
        if action == 0:  # 上
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # 下
            new_pos[0] = min(self.height - 1, new_pos[0] + 1)
        elif action == 2:  # 左
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # 右
            new_pos[1] = min(self.width - 1, new_pos[1] + 1)

        new_pos = tuple(new_pos)
        old_pos = self.player_pos

        # 检查新位置
        cell_type = self.maze[new_pos]
        reward = -0.1
        done = False

        if cell_type == self.WALL:
            # 撞墙，不移动
            reward = -1.0
        elif cell_type == self.TREASURE:
            # 找到宝藏
            reward = 10.0
            done = True
            self.game_over = True
            self.player_pos = new_pos
        elif cell_type == self.TRAP:
            # 掉进陷阱
            reward = -5.0
            done = True
            self.game_over = True
            self.player_pos = new_pos
        else:
            # 正常移动
            self.player_pos = new_pos

        # 步数限制，防止无限循环
        if self.steps >= 100:
            done = True

        self.total_reward += reward
        return self._get_state(), reward, done

    def render_ascii(self):

        for i in range(self.height):
            row = ''
            for j in range(self.width):
                if (i, j) == self.player_pos:
                    row += 'P '
                elif self.maze[i, j] == self.WALL:
                    row += '█ '
                elif self.maze[i, j] == self.TREASURE:
                    row += 'T '
                elif self.maze[i, j] == self.TRAP:
                    row += 'X '
                else:
                    row += '. '
            print(row)
        print(f"Reward: {self.total_reward:.1f}, Steps: {self.steps}")


#  Q-learning 代理
class QLearningAgent:


    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.01):
        self.actions = actions
        self.lr = learning_rate  # 学习率 α
        self.gamma = discount_factor  # 折扣因子 γ
        self.epsilon = epsilon  # 探索率 ε
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q表：字典
        self.q_table = defaultdict(lambda: defaultdict(float))

        # 训练统计
        self.training_rewards = []
        self.training_steps = []

    def get_action(self, state):
        #ε-greedy 策略选择动作
        if random.random() < self.epsilon:
            # 强制随机选择
            return random.choice(self.actions)
        else:
            # 选择Q值最大的动作
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(self.actions)

            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):

        # 当前Q值
        current_q = self.q_table[state][action]

        # 计算目标Q值
        if done:
            target_q = reward
        else:
            # 下一个状态的最大Q值
            next_q_values = self.q_table[next_state]
            if next_q_values:
                max_next_q = max(next_q_values.values())
            else:
                max_next_q = 0
            target_q = reward + self.gamma * max_next_q

        # Q-learning 更新
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        #衰减探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_table.npy"):
        # 保存Q表转换为可保存的格式
        q_dict = {}
        for state, actions in self.q_table.items():
            q_dict[str(state)] = dict(actions)
        np.save(filename, q_dict)

    def load_q_table(self, filename="q_table.npy"):
        #加载Q表
        try:
            q_dict = np.load(filename, allow_pickle=True).item()
            for state_str, actions in q_dict.items():
                # 从字符串恢复状态元组
                state = eval(state_str)
                for action, value in actions.items():
                    self.q_table[state][action] = value
            print(f"Load Q table successfully，include {len(self.q_table)} sates")
            return True
        except:
            print("Q table file not found, training will start from scratch")
            return False


#  Pygame 可视化界面
class MazeGUI:
    #Pygame图形界面

    def __init__(self, game, agent, cell_size=60):
        self.game = game
        self.agent = agent
        self.cell_size = cell_size

        # 初始化Pygame
        pygame.init()
        self.width = game.width * cell_size
        self.height = game.height * cell_size + 100  # 额外空间用于显示信息
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-learning Maze Treasure Hunt")

        # 颜色定义
        self.COLORS = {
            game.EMPTY: (255, 255, 255),  # 白色
            game.WALL: (100, 100, 100),  # 灰色
            game.TREASURE: (255, 215, 0),  # 金色
            game.TRAP: (255, 0, 0),  # 红色
            game.PLAYER: (0, 100, 255)  # 蓝色
        }

        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # 控制变量
        self.clock = pygame.time.Clock()
        self.running = True
        self.auto_mode = False
        self.episode = 0
        self.max_episodes = 500
        self.message = ""  # 提示消息
        self.message_timer = 0  # 消息计时器
        self.message_duration = 180  # 显示帧数

    def draw_grid(self):

        # 绘制游戏区域
        for i in range(self.game.height):
            for j in range(self.game.width):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size)

                # 绘制格子
                cell_type = self.game.maze[i, j]
                pygame.draw.rect(self.screen, self.COLORS[cell_type], rect)

                # 绘制边框
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

                # 在宝藏和陷阱上绘制标记
                if cell_type == self.game.TREASURE:
                    text = self.small_font.render("$", True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                elif cell_type == self.game.TRAP:
                    text = self.small_font.render("X", True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        # 绘制玩家
        player_rect = pygame.Rect(self.game.player_pos[1] * self.cell_size,
                                  self.game.player_pos[0] * self.cell_size,
                                  self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLORS[self.game.PLAYER], player_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), player_rect, 2)

        # 绘制玩家表情
        text = self.small_font.render("P", True, (255, 255, 255))
        text_rect = text.get_rect(center=player_rect.center)
        self.screen.blit(text, text_rect)

    def draw_info(self):
        #绘制信息背景
        info_y = self.game.height * self.cell_size

        # 背景
        pygame.draw.rect(self.screen, (240, 240, 240),
                         (0, info_y, self.width, 100))

        # 显示游戏信息
        info_texts = [
            f"Round: {self.episode}/{self.max_episodes}",
            f"Reward: {self.game.total_reward:.1f}",
            f"Step count: {self.game.steps}",
            f"Exploration rate ε: {self.agent.epsilon:.3f}",
            f"state: {'training' if self.auto_mode else 'manual'}",
            f"Q Table Size: {len(self.agent.q_table)}"
        ]

        x = 10
        y = info_y + 5
        for i, text in enumerate(info_texts):
            color = (0, 100, 0) if i == 4 and self.auto_mode else (0, 0, 0)
            rendered = self.small_font.render(text, True, color)
            self.screen.blit(rendered, (x, y + i * 25))

        # 显示操作说明
        help_texts = [
            "Space: Start/Stop Training",
            "R: Reset game",
            "S: Single step execution",
            "Q: Save Q-table",
            "L: Load Q table",
            "ESC: exit"
        ]

        x = 400
        for i, text in enumerate(help_texts):
            rendered = self.small_font.render(text, True, (0, 0, 0))
            self.screen.blit(rendered, (x, y + i * 25))

    def run_episode(self, render=True):

        state = self.game.reset()
        done = False

        while not done:
            if render:
                self.handle_events()
                if not self.running:
                    return False

            # 选择动作
            action = self.agent.get_action(state)

            # 执行动作
            next_state, reward, done = self.game.step(action)

            # 更新Q表
            self.agent.update(state, action, reward, next_state, done)

            # 更新状态
            state = next_state

            if render:
                self.render()
                self.clock.tick(10)  # 控制速度

        # 记录训练数据
        self.agent.training_rewards.append(self.game.total_reward)
        self.agent.training_steps.append(self.game.steps)
        self.agent.decay_epsilon()

        return True

    def render(self):
        #渲染一帧
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_info()
        pygame.display.flip()

    def show_message(self, text):
        #显示临时提示消息
        self.message = text
        self.message_timer = self.message_duration

    def show_game_over_popup(self, title, message):
        #显示简单的游戏结束弹窗
        # 创建半透明遮罩
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))

        # 创建弹窗
        popup_width, popup_height = 350, 200

        # 根据标题选择颜色
        if "VICTORY" in title:
            bg_color = (200, 255, 200)  # 浅绿色
            border_color = (0, 150, 0)  # 绿色
            emoji = ""
        else:
            bg_color = (255, 200, 200)  # 浅红色
            border_color = (150, 0, 0)  # 红色
            emoji = ""

        popup = pygame.Surface((popup_width, popup_height))
        popup.fill(bg_color)
        pygame.draw.rect(popup, border_color, popup.get_rect(), 5)

        # 标题
        title_text = self.font.render(f"{emoji} {title} {emoji}", True, border_color)
        title_rect = title_text.get_rect(center=(popup_width // 2, 40))
        popup.blit(title_text, title_rect)

        # 消息内容
        lines = message.split('\n')
        y_offset = 90
        for line in lines:
            msg_text = self.small_font.render(line, True, (0, 0, 0))
            msg_rect = msg_text.get_rect(center=(popup_width // 2, y_offset))
            popup.blit(msg_text, msg_rect)
            y_offset += 25

        # 提示
        hint_text = self.small_font.render("Press any key to continue", True, (100, 100, 100))
        hint_rect = hint_text.get_rect(center=(popup_width // 2, popup_height - 30))
        popup.blit(hint_text, hint_rect)

        # 计算弹窗位置（屏幕中央）
        popup_rect = popup.get_rect(center=(self.width // 2, self.height // 2))

        # 暂停自动模式
        was_auto = self.auto_mode
        self.auto_mode = False

        # 等待用户按键
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    waiting = False
                    self.running = False

            # 绘制原画面
            self.screen.fill((255, 255, 255))
            self.draw_grid()
            self.draw_info()

            # 绘制遮罩和弹窗
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(popup, popup_rect)

            pygame.display.flip()
            self.clock.tick(30)

        # 恢复之前的模式
        self.auto_mode = was_auto

    def draw_info(self):
        #绘制信息面板
        info_y = self.game.height * self.cell_size

        # 背景
        pygame.draw.rect(self.screen, (240, 240, 240),
                         (0, info_y, self.width, 100))

        # 显示游戏信息
        info_texts = [
            f"Episode: {self.episode}/{self.max_episodes}",
            f"Reward: {self.game.total_reward:.1f}",
            f"Steps: {self.game.steps}",
            f"Epsilon ε: {self.agent.epsilon:.3f}",
            f"Mode: {'TRAINING' if self.auto_mode else 'MANUAL'}",
            f"Q-table size: {len(self.agent.q_table)}"
        ]

        x = 10
        y = info_y + 5
        for i, text in enumerate(info_texts):
            color = (0, 150, 0) if i == 4 and self.auto_mode else (0, 0, 0)
            rendered = self.small_font.render(text, True, color)
            self.screen.blit(rendered, (x, y + i * 25))

        # 显示操作说明
        help_texts = [
            "SPACE: Start/Stop training",
            "R: Reset game",
            "S: Single step",
            "Q: Save Q-table",
            "L: Load Q-table",
            "ESC: Exit"
        ]

        x = 400
        for i, text in enumerate(help_texts):
            rendered = self.small_font.render(text, True, (0, 0, 0))
            self.screen.blit(rendered, (x, y + i * 25))

        # 显示临时提示消息
        if self.message_timer > 0:
            # 创建半透明背景
            msg_surface = pygame.Surface((300, 40))
            msg_surface.set_alpha(200)
            msg_surface.fill((50, 50, 50))

            # 显示消息
            msg_x = self.width // 2 - 150
            msg_y = info_y + 30
            self.screen.blit(msg_surface, (msg_x, msg_y))

            msg_text = self.small_font.render(self.message, True, (255, 255, 0))
            msg_rect = msg_text.get_rect(center=(self.width // 2, msg_y + 20))
            self.screen.blit(msg_text, msg_rect)

            self.message_timer -= 1

    def handle_events(self):
        #处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.auto_mode = not self.auto_mode
                    self.show_message(f"Auto mode: {'ON' if self.auto_mode else 'OFF'}")

                elif event.key == pygame.K_r:
                    self.game.reset()
                    self.show_message("Game reset!")

                elif event.key == pygame.K_s and not self.auto_mode:
                    if self.game.game_over:
                        self.show_message("Game over! Press R to reset")
                    else:
                        state = self.game.player_pos
                        action = self.agent.get_action(state)
                        next_state, reward, done = self.game.step(action)
                        self.agent.update(state, action, reward, next_state, done)

                        action_name = self.game.action_names[action]
                        self.show_message(f"Action: {action_name} Reward: {reward:.1f}")

                elif event.key == pygame.K_q:
                    self.agent.save_q_table()
                    self.show_message("Q-table saved!")

                elif event.key == pygame.K_l:
                    if self.agent.load_q_table():
                        self.show_message(f"Q-table loaded! {len(self.agent.q_table)} states")
                    else:
                        self.show_message("No Q-table file found")

    def run_episode(self, render=True):
        #运行一个回合
        state = self.game.reset()
        done = False

        while not done:
            if render:
                self.handle_events()
                if not self.running:
                    return False

            action = self.agent.get_action(state)
            next_state, reward, done = self.game.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

            if render:
                self.render()
                self.clock.tick(10)

        # 游戏结束时显示弹窗（只在手动模式）
        if render and not self.auto_mode:
            if self.game.maze[self.game.player_pos] == self.game.TREASURE:
                self.show_game_over_popup(
                    "VICTORY!",
                    f"Found treasure! +10\nTotal reward: {self.game.total_reward:.1f}\nSteps: {self.game.steps}"
                )
            elif self.game.maze[self.game.player_pos] == self.game.TRAP:
                self.show_game_over_popup(
                    "GAME OVER",
                    f"Hit trap! -5\nTotal reward: {self.game.total_reward:.1f}\nSteps: {self.game.steps}"
                )

        # 记录训练数据
        self.agent.training_rewards.append(self.game.total_reward)
        self.agent.training_steps.append(self.game.steps)
        self.agent.decay_epsilon()

        return True

    def run(self):
        #主循环
        while self.running and self.episode < self.max_episodes:
            self.handle_events()

            if self.auto_mode:
                # 自动训练模式
                success = self.run_episode(render=True)
                if not success:
                    break
                self.episode += 1

                # 每10回合打印一次统计
                if self.episode % 10 == 0:
                    avg_reward = np.mean(self.agent.training_rewards[-10:])
                    avg_steps = np.mean(self.agent.training_steps[-10:])
                    print(f"Round {self.episode}: average reward={avg_reward:.2f}, "
                          f"Average Steps={avg_steps:.1f}, ε={self.agent.epsilon:.3f}")
            else:
                # 手动模式，只渲染
                self.render()
                self.clock.tick(30)

        pygame.quit()

        # 添加延时，确保Pygame完全关闭

        time.sleep(0.5)  # 等待0.5秒

        # 训练结束后绘制学习曲线
        if len(self.agent.training_rewards) > 0:
            self.plot_learning_curve()

    def plot_learning_curve(self):
        """绘制学习曲线"""
        try:
            # 检测是否在Pygame环境中
            try:

                # 检查Pygame是否还在运行
                pygame.display.get_init()
                in_pygame = True
                print("[Info] Running in Pygame environment")
            except:
                in_pygame = False
                print("[Info] Running in console mode")


            # 根据环境选择后端
            if in_pygame:

                # 在Pygame中：先保存，退出Pygame后再显示
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
                plt.rcParams['axes.unicode_minus'] = False

                plt.figure(figsize=(12, 4))


                # 绘制奖励曲线
                plt.subplot(1, 2, 1)
                plt.plot(self.agent.training_rewards, color='blue', linewidth=1.5)
                plt.title('Training Reward Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Total Reward', fontsize=10)
                plt.grid(True, alpha=0.3)


                # 绘制步数曲线
                plt.subplot(1, 2, 2)
                plt.plot(self.agent.training_steps, color='green', linewidth=1.5)
                plt.title('Training Steps Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Steps', fontsize=10)
                plt.grid(True, alpha=0.3)

                plt.tight_layout()


                # 保存文件
                plt.savefig('learning_curve.png', dpi=100, bbox_inches='tight')
                print("\n" + "=" * 60)
                print("✓ Learning curve saved as 'learning_curve.png'")



                # 关闭当前图形
                plt.close()


                # 退出Pygame后再显示
                print("\n[Info] Preparing to display learning curve window...")
                print("[Info] Please wait, Pygame is closing...")


                # 确保Pygame完全退出
                pygame.quit()
                time.sleep(1)


                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt

                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
                plt.rcParams['axes.unicode_minus'] = False


                # 重新绘制并显示
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 2, 1)
                plt.plot(self.agent.training_rewards, color='blue', linewidth=1.5)
                plt.title('Training Reward Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Total Reward', fontsize=10)
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.plot(self.agent.training_steps, color='green', linewidth=1.5)
                plt.title('Training Steps Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Steps', fontsize=10)
                plt.grid(True, alpha=0.3)

                plt.tight_layout()


                # 显示图表窗口
                print("\n[Info] Displaying learning curve...")
                print("[Info] Close the window to continue")
                plt.show()
                print(" Learning curve displayed successfully")


            else:
                # Not in Pygame

                import matplotlib.pyplot as plt



                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
                plt.rcParams['axes.unicode_minus'] = False

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 2, 1)
                plt.plot(self.agent.training_rewards, color='blue', linewidth=1.5)
                plt.title('Training Reward Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Total Reward', fontsize=10)
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.plot(self.agent.training_steps, color='green', linewidth=1.5)
                plt.title('Training Steps Curve', fontsize=12)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Steps', fontsize=10)
                plt.grid(True, alpha=0.3)

                plt.tight_layout()


                # 保存并显示
                plt.savefig('learning_curve.png', dpi=100, bbox_inches='tight')
                print("\n" + "=" * 60)
                print(" Learning curve saved as 'learning_curve.png'")


                print("\n[Info] Displaying learning curve...")
                print("[Info] Close the window to continue")
                plt.show()
                print("Learning curve displayed successfully")



            # 打印最终总结
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY / 训练总结")
            print("=" * 60)
            print(f"Total episodes / 总回合数: {len(self.agent.training_rewards)}")
            print(f"Final average reward / 最终平均奖励: {np.mean(self.agent.training_rewards[-50:]):.2f}")
            print(f"Final average steps / 最终平均步数: {np.mean(self.agent.training_steps[-50:]):.1f}")
            print(f"Q-table size / Q表大小: {len(self.agent.q_table)}")
            print("=" * 60)

        except Exception as e:
            print(f"\n[Error] Failed to plot learning curve: {e}")
            print(f"[错误] 绘制学习曲线失败: {e}")




# 控制台训练模式
def train_console_mode():
    #控制台训练模式

    print("Q-learning Treasure Hunt in the Maze - Console Training Mode")


    # 创建游戏和代理
    game = MazeGame(width=10, height=10)
    agent = QLearningAgent(
        actions=game.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    # 训练参数
    episodes = 500
    print_every = 50

    print(f"\nStart training Round:{episodes} ...")
    start_time = time.time()

    for episode in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.training_rewards.append(total_reward)
        agent.training_steps.append(game.steps)
        agent.decay_epsilon()

        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(agent.training_rewards[-print_every:])
            avg_steps = np.mean(agent.training_steps[-print_every:])
            print(f"Round {episode + 1}/{episodes}: "
                  f"average reward={avg_reward:.2f}, "
                  f"Average Steps={avg_steps:.1f}, "
                  f"ε={agent.epsilon:.3f}")

    train_time = time.time() - start_time
    print(f"\nTraining completed! time-consuming: {train_time:.2f} Second")

    # 测试训练好的策略
    print("\nTest the trained strategy...")
    game.reset()
    agent.epsilon = 0

    for step in range(20):
        state = game.player_pos
        action = agent.get_action(state)
        next_state, reward, done = game.step(action)
        game.render_ascii()
        print(f"Action: {game.action_names[action]}, Reward: {reward}")

        if done:
            if game.maze[game.player_pos] == game.TREASURE:
                print("Great You have successfully found the treasure！")
            elif game.maze[game.player_pos] == game.TRAP:
                print(" You fell into it...")
            break
        time.sleep(0.5)

    #  修复绘图代码
    print("\nDrawing learning curve...")
    try:

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(agent.training_rewards)

        plt.title('Training Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(agent.training_steps)
        plt.title('Training Steps Curve')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)

        plt.tight_layout()

        # 保存文件
        plt.savefig('learning_curve.png', dpi=100, bbox_inches='tight')
        print("The learning curve has been saved as learning_curve.png")

        # 关闭图形释放内存
        plt.close()

    except Exception as e:
        print(f"Drawing with Agg backend failed: {e}")



    # 保存Q表
    agent.save_q_table()
    print("\nThe Q table has been saved as q_table.npy")

    return agent



def main():
    #主函数
    print("Please select the operating mode:")
    print("1. Graphic interface mode")
    print("2. Console training mode")

    choice = input("Pleace select(1/2): ").strip()

    if choice == "2":
        # 控制台训练模式
        agent = train_console_mode()
    else:
        # 图形界面模式
        try:
            game = MazeGame(width=10, height=10)
            agent = QLearningAgent(
                actions=game.actions,
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=0.3,
                epsilon_decay=0.995,
                min_epsilon=0.01
            )

            # 尝试加载已保存的Q表
            agent.load_q_table()

            gui = MazeGUI(game, agent)
            gui.run()

            print("\nProgram exits normally")

        except KeyboardInterrupt:
            print("\nUser Interrupt Program")
        except Exception as e:
            print(f"Failed to launch graphical interface: {e}")
            print("Switch to console mode ...")
            train_console_mode()

    # 最终退出
    print("Program ends")


if __name__ == "__main__":
    main()