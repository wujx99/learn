# $\pi^*_{0.6}$ 相关 RL 入门地图

这份文档面向“小白读者”：目标不是系统学完整强化学习，而是让你能读懂
[$\pi^*_{0.6}$ 论文笔记](README.md) 里为什么要用 value function、advantage、
offline RL、人类纠正、优势条件化策略这些东西。

先澄清名字：这个目录里的论文是 $\pi^*_{0.6}$，论文标题是
`a VLA That Learns From Experience`。它不是简单的 $\pi_{0.6}$，而是在
$\pi_{0.6}$ 架构上加入从真实机器人经验中学习的 RL 流程。这里的星号 `*`
表示“经过经验学习增强后的版本”。

## 你最终要理解什么

读这篇论文时，最重要的一句话是：

> RECAP 用一个价值函数判断数据里的动作是“相对更好”还是“相对更差”，再把这个判断变成 VLA 的条件输入，让机器人在推理时偏向生成更可能成功、更快完成任务的动作。

为了理解这句话，你需要补齐四层知识：

1. 机器人策略是什么：从图像、语言和机器人状态生成动作。
2. 强化学习在解决什么问题：不是只模仿人，而是根据结果反馈改进策略。
3. 离线 RL 为什么重要：真实机器人试错贵，所以要高效利用旧数据、失败数据和人类纠正。
4. RECAP 的特殊设计：不用传统 PPO 直接更新大 VLA，而是把 RL 变成“优势条件化的行为建模”。

## 最小背景：VLA 和机器人控制

VLA 是 Vision-Language-Action 的缩写：

- Vision：机器人看到的相机图像。
- Language：人给的任务指令，比如“fold the shirt”。
- Action：机器人要执行的控制量，比如关节目标位置、夹爪开合。

普通语言模型输出 token；VLA 还要输出动作。$\pi_{0.6}$ 这类模型通常不是只输出一个动作，而是输出一段 action chunk，让机器人以较高频率连续执行。

可以把策略理解成一个函数：

$$
\pi(a \mid o, \ell)
$$

其中 $o$ 是 observation，$\ell$ 是 language instruction，$a$ 是 action。它的意思是：给定当前看到的东西和语言指令，模型应该产生什么动作。

对机器人来说，难点在于：

- 动作是连续的，不像文字 token 那样离散。
- 任务很长，早期动作的影响可能几分钟后才体现。
- 成功标签很稀疏，通常只有“整次任务成功/失败”。
- 真实采样昂贵，机器人摔坏、卡住、失败都要人工处理。

## 从模仿学习开始

最容易理解的机器人训练方式是模仿学习，也叫 behavior cloning 或 SFT：

1. 人类专家演示任务。
2. 记录每一步 observation 和 expert action。
3. 训练模型在相同 observation 下输出专家动作。

目标大致是：

$$
\max_\theta \log \pi_\theta(a_{\mathrm{expert}} \mid o, \ell)
$$

它的优点是稳定、简单、适合大模型预训练。缺点也很明显：

- 如果示范不够好，模型上限受示范限制。
- 部署时模型一旦偏离示范分布，错误会累积。
- 失败轨迹通常很难直接利用，因为模仿失败动作没有意义。
- 它只回答“人当时怎么做”，不回答“哪个动作能让结果变好”。

$\pi^*_{0.6}$ 要解决的核心问题，就是让 VLA 不只模仿，还能从成功、失败和纠正中改进。

## 强化学习最小概念

强化学习关心一个循环：

$$
o_t \rightarrow a_t \rightarrow r_t \rightarrow o_{t+1}
$$

机器人看到状态 $o_t$，执行动作 $a_t$，环境给奖励 $r_t$，然后进入下一个状态。

### Reward

reward 是每一步得到的反馈。真实机器人任务里经常没有细粒度 reward，只有 episode 结束时的成功/失败。

在 $\pi^*_{0.6}$ 笔记里，论文用类似下面的 reward：

$$
r_t =
\begin{cases}
0, & t = T \ \text{and episode succeeds} \\
-C_{\mathrm{fail}}, & t = T \ \text{and episode fails} \\
-1, & \text{otherwise}
\end{cases}
$$

直觉是：

- 每多花一步就扣 $1$，所以模型会偏向更快完成。
- 成功结束不给额外惩罚。
- 失败结束给很大惩罚。

这让 value function 不只学“会不会成功”，也学“离成功还有多远”。

### Return

return 是从当前时刻开始，未来所有 reward 的总和：

$$
G_t = r_t + r_{t+1} + \cdots + r_T
$$

如果每一步都扣 $1$，那么越快成功，return 越高。失败会因为最后的大惩罚而 return 很低。

### Value Function

value function 估计当前状态有多好：

$$
V(o_t, \ell) \approx \mathbb{E}[G_t \mid o_t, \ell]
$$

在机器人任务里，它回答的问题是：

> 从现在这个画面和任务指令出发，最终成功并且快速完成的希望有多大？

$\pi^*_{0.6}$ 的 RECAP 流程会训练一个多任务 value function。它看 observation 和 language instruction，预测一个离散 value bin 分布。这样做比直接回归一个数字更稳定，也能表达不确定性。

### Q Function

Q function 估计“在当前状态先做某个动作，然后继续执行策略”的好坏：

$$
Q(o_t, a_t, \ell) \approx \mathbb{E}[G_t \mid o_t, a_t, \ell]
$$

直觉上，$V$ 评价“这个状态好不好”，$Q$ 评价“这个状态下这个动作好不好”。

### Advantage

advantage 比较的是“这个动作是否比当前策略的平均动作更好”：

$$
A(o_t, a_t, \ell) = Q(o_t, a_t, \ell) - V(o_t, \ell)
$$

如果 $A > 0$，说明这个动作比当前状态下的平均选择更好。如果 $A < 0$，说明这个动作可能让事情变差或变慢。

这就是 RECAP 的关键。它不需要把每个动作都打成绝对分数，只需要判断这个动作相对当前策略是不是正优势。

## 为什么不是直接用 PPO

很多人一听 RL，会想到 PPO。PPO 属于在线策略梯度方法，基本思路是：

1. 用当前策略采样新数据。
2. 根据 reward 估计 advantage。
3. 直接调整策略，让正 advantage 动作概率更高，负 advantage 动作概率更低。

这在游戏和仿真里常见，但放到 $\pi_{0.6}$ 这类真实机器人 VLA 上会很麻烦：

- 真实机器人采样慢，不能像仿真那样采几百万条。
- 大 VLA 参数多，在线更新成本高、稳定性差。
- 动作由 flow matching action expert 生成，传统策略梯度需要的 likelihood 和 ratio 不好直接处理。
- 数据不是纯 on-policy，里面混有示范、历史轨迹、失败、自主 rollout、人类 intervention。

所以论文没有简单把 PPO 套上去，而是走了更适合大 VLA 和真实机器人数据的路线。

## Offline RL 是什么

offline RL 指从已经收集好的数据里学习策略，而不是每更新一步都必须和环境交互。

在 $\pi^*_{0.6}$ 里，数据来源包括：

- 旧的示范数据。
- 当前策略自主执行的成功轨迹。
- 当前策略自主执行的失败轨迹。
- 人类中途接管或纠正的片段。

offline RL 的难点是：数据里不一定包含所有可能动作。如果模型学到了数据外动作，value function 可能会过度乐观，策略也可能崩。RECAP 的做法比较保守：它主要在已有数据动作上判断 advantage，然后继续做行为建模，不鼓励模型离数据分布太远。

## 人类纠正为什么有用

人类 intervention 的意思是：机器人执行任务时，如果快失败了，人类专家接管或给出纠正动作。

这类数据很有价值，因为它通常出现在模型不擅长的状态附近：

- 纯示范数据多是“专家从正常状态顺利完成”。
- 自主失败数据告诉你哪里会坏。
- 人类纠正数据告诉你“坏到一半时怎么救回来”。

RECAP 把人类纠正片段强制标为 positive advantage，因为作者假设专家纠正动作比机器人原本动作更有益。

这和 DAgger 的思想有关：不要只在完美示范状态上训练，还要让模型看到自己实际会遇到的错误状态，并学习专家在这些状态下怎么做。

## AWR、CRR 和 RECAP 的关系

你不需要先完整掌握 AWR/CRR，但要知道它们解决的问题和 RECAP 相近。

AWR 是 Advantage-Weighted Regression。它会给高 advantage 动作更高权重，让策略更像这些好动作：

$$
\max_\theta w(A) \log \pi_\theta(a \mid o, \ell)
$$

CRR 是 Critic-Regularized Regression。它也会用 critic/value 判断哪些动作值得模仿。

这些方法的共同点是：用价值估计指导行为克隆。但它们通常通过“加权”或“过滤”改变训练样本的重要性。

RECAP 的区别是：它不简单丢掉低 advantage 数据，也不只是加权高 advantage 数据。它把 advantage 变成条件输入：

$$
\pi(a \mid I, o, \ell)
$$

其中 $I$ 可以是 `Advantage: positive` 或 `Advantage: negative`。训练时模型同时学习普通行为分布和优势条件行为分布；推理时直接给 `Advantage: positive`，让模型偏向生成更好的动作。

## Flow Matching 和动作生成

$\pi_{0.6}$ 的动作专家不是普通分类器，而是连续动作生成模型。论文笔记里提到它有 flow matching action expert。

你可以先用非常粗略的方式理解：

- 语言模型生成离散 token。
- 机器人动作是连续向量，不能直接当普通词表分类。
- flow matching 学的是如何把噪声逐步变成合理动作。

这类动作生成模型很适合表达多峰动作分布。比如拿杯子时，左手先动和右手先动可能都合理；动作不是唯一答案。

但这也带来一个问题：传统 PPO 常依赖动作概率、log probability、policy ratio。对复杂的 flow matching 动作专家来说，直接做这套东西工程上不自然，也可能不稳定。

RECAP 的优势条件化策略提取，本质上是在绕开这个困难：不直接对策略做复杂的在线 policy gradient，而是把“生成更好动作”变成条件生成问题。

## RECAP 一步步看

RECAP 全称是 RL with Experience and Corrections via Advantage-conditioned Policies。

可以按五步理解：

1. 先有一个会做任务但不够好的 VLA，也就是 $\pi_{0.6}$。
2. 让它上真实机器人执行任务，收集成功、失败和人类纠正。
3. 用所有数据训练 value function，让模型知道哪些状态离成功更近。
4. 用 value function 估计每个动作的 advantage，并离散成 positive/negative。
5. 训练 VLA 学会根据 `Advantage: positive` 条件生成更优动作。

论文里的策略训练目标可以理解成两部分：

$$
\min_{\theta}\ \mathbb{E}_{\mathcal{D}}
\left[
  -\log \pi_{\theta}(a_t \mid o_t, \ell)
  -\alpha \log \pi_{\theta}(a_t \mid I_t, o_t, \ell)
\right]
$$

第一项是普通模仿学习：不管 advantage，先学会数据里的动作分布。第二项是条件模仿学习：告诉模型这个动作是 positive 还是 negative，让它学会区分两类动作。

推理时给定：

```text
Advantage: positive
```

模型就倾向于选择 value function 认为更好的动作类型。

## 为什么这套方法适合 $\pi^*_{0.6}$

它解决了几个现实约束：

- 数据贵：offline RL 可以重复利用已有轨迹。
- 奖励稀疏：只用 episode 成功/失败也能构造 return。
- 动作复杂：避免直接对 flow matching policy 做传统 PPO 更新。
- 数据混杂：示范、失败、自主 rollout、人类纠正都能放进同一套流程。
- 大模型稳定性：保留行为建模目标，降低策略跑出数据分布的风险。

这也是为什么论文强调它不是“把 PPO 套到 VLA 上”，而是为 VLA 和真实机器人重新组织 RL。

## 读论文时的概念对照表

| 论文概念 | 小白解释 | 你要记住的作用 |
| --- | --- | --- |
| VLA | 看图、读指令、出动作的机器人模型 | 策略主体 |
| Rollout | 机器人从开始做到结束的一次执行 | 收集经验 |
| Episode | 一次完整任务 | 最后会被标成功或失败 |
| Sparse reward | 只有结尾成功/失败反馈 | 真实任务常见 |
| Return | 从某一步往后的总奖励 | 衡量未来结果 |
| Value function | 当前状态有多有希望 | 判断进度和成功可能 |
| Q function | 当前状态做某动作有多好 | 判断动作质量 |
| Advantage | 某动作比平均选择好多少 | RECAP 的核心训练信号 |
| Positive advantage | 相对更好的动作 | 推理时希望模型偏向它 |
| Intervention | 人类中途纠正 | 给模型学习错误状态下的救场动作 |
| Offline RL | 用已有数据做 RL | 适合真实机器人 |
| PPO | 常见在线策略梯度算法 | 论文对比但不是主方案 |
| AWR/CRR | 用 advantage/value 指导模仿 | RECAP 的相关基线 |
| Flow matching | 连续动作生成方法 | 解释为什么传统 PPO 不顺手 |

## 推荐学习顺序

如果你完全是小白，建议按这个顺序学：

1. 先读本文件到“RECAP 一步步看”，不要纠结公式推导。
2. 回到 [论文笔记](README.md)，重点读“一句话结论”“方法拆解”“实验结论”。
3. 补 RL 基础：MDP、reward、return、value、Q、advantage。
4. 补 imitation learning：behavior cloning、distribution shift、DAgger。
5. 补 offline RL：为什么不能随便相信数据外动作，为什么要保守。
6. 补 policy gradient/PPO：理解论文为什么不用它做主方案。
7. 补 flow matching/diffusion policy：理解连续动作生成和 likelihood 难点。
8. 再读论文的公式和 controlled comparison。

## 自测问题

读完后，你应该能回答这些问题：

1. 为什么只做 behavior cloning 不够？
2. 为什么真实机器人任务里的 reward 通常很稀疏？
3. value function 和 advantage 分别回答什么问题？
4. 人类 intervention 数据为什么比普通成功示范更特殊？
5. offline RL 为什么适合真实机器人？
6. PPO 为什么不一定适合 $\pi_{0.6}$ 这种 VLA？
7. RECAP 为什么把 advantage 做成条件输入，而不是只给样本加权？
8. 推理时输入 `Advantage: positive` 实际上想控制什么？

## 读懂这篇论文的最低数学要求

你不需要先学完整 RL 教材，但最好能看懂三类式子。

第一类是条件概率：

$$
\pi(a \mid o, \ell)
$$

意思是给定观察和语言，动作的概率分布。

第二类是未来奖励：

$$
G_t = \sum_{k=t}^{T} r_k
$$

意思是从当前时刻到结束的总结果。

第三类是相对好坏：

$$
A(o_t, a_t, \ell) = Q(o_t, a_t, \ell) - V(o_t, \ell)
$$

意思是这个动作比当前状态下的平均选择更好还是更差。

如果你能把这三类式子翻译成自然语言，就能先读懂论文主线。

## 最容易误解的地方

- $\pi^*_{0.6}$ 不是从零开始 RL 训练出来的机器人，而是在已有 VLA 基础上继续用经验改进。
- RECAP 不是纯在线 RL。它是迭代收集真实数据，再用离线方式重新训练 value 和策略。
- 失败数据不是垃圾。失败轨迹对 value function 很重要，因为它告诉模型哪些状态和动作会走向坏结果。
- 人类纠正不只是“多一些示范”。它覆盖了模型自己会进入的错误状态。
- `Advantage: positive` 不是任务指令，而是控制策略生成风格的条件信号。
- throughput 提升不只是成功率提升，还包括更快完成任务。

## 和现有论文笔记怎么配合

本文件负责建立概念地图；[README.md](README.md) 负责记录论文细节。建议你每次遇到不懂的论文段落，就回到本文件查对应概念：

- 看不懂 value bins：回到 “Value Function”。
- 看不懂 positive/negative advantage：回到 “Advantage” 和 “RECAP 一步步看”。
- 看不懂为什么比较 PPO/AWR/CRR：回到 “为什么不是直接用 PPO” 和 “AWR、CRR 和 RECAP 的关系”。
- 看不懂 human corrections：回到 “人类纠正为什么有用”。

## 下一步可以补的材料

为了真正吃透 $\pi^*_{0.6}$，后续可以继续补四类笔记：

- RL 基础短笔记：MDP、Bellman equation、policy gradient、actor-critic。
- 模仿学习短笔记：behavior cloning、DAgger、offline dataset shift。
- Offline RL 短笔记：AWR、CRR、IQL、conservative Q-learning。
- 机器人动作生成短笔记：diffusion policy、flow matching、action chunking。

如果只服务于读这篇论文，优先级最高的是 behavior cloning、advantage、offline RL 和 flow matching。
