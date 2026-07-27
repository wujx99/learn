---
title: 自动驾驶坐标变换与鱼眼相机几何
topic: autonomous-driving-coordinate-transforms
topics:
  - autonomous-driving-coordinate-transforms
  - autonomous-driving-world-models
framework: framework-agnostic
status: learning
updated: 2026-07-27
---

# 自动驾驶坐标变换与鱼眼相机几何

## 学习目标

学完本文后，应当能够：

1. 用无歧义的记号描述 world、ego、LiDAR、camera 与 image 之间的坐标变换；
2. 推导刚体变换的复合、求逆，以及点和方向向量的不同变换方式；
3. 从三维点出发，完整推导针孔相机与鱼眼相机的像素投影；
4. 理解鱼眼相机不是“针孔投影后再做一点畸变”，而是以入射角为核心的投影模型；
5. 从鱼眼像素反求空间射线，并理解为什么通常需要数值求根；
6. 写出优化和标定中常用的投影雅可比，定位坐标方向、单位与模型混用问题。

## 前置知识与全文约定

需要基本的线性代数、三角函数、矩阵求导知识。本文使用列向量和左乘矩阵，并规定：

$$
{}^{A}\tilde{\mathbf p} = {}^{A}\mathbf T_B\,{}^{B}\tilde{\mathbf p}.
$$

上标是“该量用哪个坐标系表达”，而 ${}^{A}\mathbf T_B$ 表示**把 $B$ 系坐标转换成 $A$ 系坐标**。齐次点为

$$
{}^B\tilde{\mathbf p}=\begin{bmatrix}{}^B\mathbf p\\1\end{bmatrix}.
$$

不同数据集的轴定义并不统一。常见车体系可能取 $x$ 向前、$y$ 向左、$z$ 向上；OpenCV 相机系通常取 $x$ 向右、$y$ 向下、$z$ 向前。轴定义必须从数据集或标定文件读取，不能靠名称猜测。

## 核心心智模型：先换观察者，再做成像

一条完整的几何链只有两类操作：

1. **坐标变换**：同一个物理点，改用另一个坐标系表达；
2. **相机投影**：把相机坐标系中的三维射线映射成二维像素。

例如，将 LiDAR 点投影到相机图像：

$$
{}^C\tilde{\mathbf p}
= {}^C\mathbf T_E\,{}^E\mathbf T_L\,{}^L\tilde{\mathbf p},
\qquad
\mathbf u=\pi({}^C\mathbf p).
$$

右边最先作用。前半段与相机是不是鱼眼无关；只有最后的投影函数 $\pi$ 不同。

```text
LiDAR point --SE(3)--> ego --SE(3)--> camera ray --camera model--> pixel
```

这一区分非常重要：外参错误会让整幅图出现方向一致的错位；投影模型错误往往表现为中心附近尚可、越靠边缘误差越大。

## 1. 坐标系与刚体变换

### 1.1 从基向量推导旋转矩阵

设 $B$ 系的三个单位基向量用 $A$ 系表达为 ${}^A\mathbf b_1,{}^A\mathbf b_2,{}^A\mathbf b_3$，则

$$
{}^A\mathbf R_B=
\begin{bmatrix}
{}^A\mathbf b_1 & {}^A\mathbf b_2 & {}^A\mathbf b_3
\end{bmatrix}.
$$

因此 ${}^A\mathbf R_B$ 的**列**回答的是：“$B$ 的各坐标轴在 $A$ 中指向哪里？”方向向量满足

$$
{}^A\mathbf v={}^A\mathbf R_B{}^B\mathbf v.
$$

旋转矩阵属于特殊正交群

$$
SO(3)=\{\mathbf R\in\mathbb R^{3\times3}\mid
\mathbf R^\top\mathbf R=\mathbf I,\det\mathbf R=1\}.
$$

所以旋转的逆等于转置：${}^B\mathbf R_A=({}^A\mathbf R_B)^\top$。

### 1.2 平移究竟表达什么

点坐标的变换为

$$
{}^A\mathbf p={}^A\mathbf R_B{}^B\mathbf p+{}^A\mathbf t_B.
$$

令 ${}^B\mathbf p=\mathbf0$ 可知，${}^A\mathbf t_B$ 是 **$B$ 系原点在 $A$ 系中的坐标**，而不只是一个含糊的“从 $A$ 到 $B$ 的位移”。

方向向量没有位置，不受平移影响。齐次坐标用最后一维区分两者：

$$
\tilde{\mathbf p}=\begin{bmatrix}\mathbf p\\1\end{bmatrix},\qquad
\tilde{\mathbf v}=\begin{bmatrix}\mathbf v\\0\end{bmatrix}.
$$

### 1.3 $SE(3)$、复合与求逆

刚体变换写成

$$
{}^A\mathbf T_B=
\begin{bmatrix}
{}^A\mathbf R_B & {}^A\mathbf t_B\\
\mathbf0^\top & 1
\end{bmatrix}\in SE(3).
$$

中间坐标系必须像分数一样消去：

$$
{}^A\mathbf T_C={}^A\mathbf T_B{}^B\mathbf T_C.
$$

逆变换为

$$
{}^B\mathbf T_A=
\begin{bmatrix}
({}^A\mathbf R_B)^\top &
-({}^A\mathbf R_B)^\top{}^A\mathbf t_B\\
\mathbf0^\top&1
\end{bmatrix}.
$$

注意逆变换的平移不是简单的 $-\mathbf t$；它还必须被逆旋转表达到新坐标系中。

### 1.4 一个最小例子

假设车体系 $E$ 为 $x$ 前、$y$ 左、$z$ 上，相机系 $C$ 为 $x$ 右、$y$ 下、$z$ 前，且先忽略两者原点偏移。由轴对应关系

$$
{}^C\mathbf e_x=\begin{bmatrix}0\\0\\1\end{bmatrix},\quad
{}^C\mathbf e_y=\begin{bmatrix}-1\\0\\0\end{bmatrix},\quad
{}^C\mathbf e_z=\begin{bmatrix}0\\-1\\0\end{bmatrix},
$$

得到

$$
{}^C\mathbf R_E=
\begin{bmatrix}
0&-1&0\\
0&0&-1\\
1&0&0
\end{bmatrix}.
$$

车前方点 $(10,0,0)^\top$ 转到相机系后为 $(0,0,10)^\top$，正好落在光轴上。这是比死记矩阵更可靠的单元测试。

## 2. 时间变化的自动驾驶变换链

传感器相对车体的外参通常固定，而车体在 world/map 系中的位姿随时间变化：

$$
{}^W\mathbf T_{S(t)}
= {}^W\mathbf T_{E(t)}{}^E\mathbf T_S.
$$

把 $t_1$ 时刻 LiDAR 点变换到 $t_2$ 时刻相机：

$$
{}^{C(t_2)}\tilde{\mathbf p}
= {}^C\mathbf T_E
\left({}^W\mathbf T_{E(t_2)}\right)^{-1}
{}^W\mathbf T_{E(t_1)}
{}^E\mathbf T_L
{}^{L(t_1)}\tilde{\mathbf p}.
$$

式中每个相邻上下标都能消去。若错误地假设 $t_1=t_2$，移动物体和远离旋转中心的静态背景都会产生错位；这不是相机畸变能够修复的问题。

## 3. 针孔相机投影

令相机坐标为

$$
{}^C\mathbf p=(X,Y,Z)^\top,\qquad Z>0.
$$

透视除法得到归一化像面坐标

$$
x=\frac XZ,\qquad y=\frac YZ.
$$

若暂不考虑畸变与 skew，内参矩阵为

$$
\mathbf K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},
$$

像素为

$$
u=f_x\frac XZ+c_x,\qquad
v=f_y\frac YZ+c_y.
$$

像素只确定射线而不能单独确定深度：

$$
\lambda
\begin{bmatrix}
(u-c_x)/f_x\\
(v-c_y)/f_y\\
1
\end{bmatrix},\qquad \lambda>0.
$$

这也是单目反投影存在尺度不确定性的根源。

## 4. 鱼眼相机为什么特殊

### 4.1 用入射角统一描述中心相机

定义光线与相机光轴的夹角和方位角：

$$
\rho=\sqrt{X^2+Y^2},\qquad
\theta=\operatorname{atan2}(\rho,Z),\qquad
\phi=\operatorname{atan2}(Y,X).
$$

任何理想的轴对称中心相机，都可以先把角度 $\theta$ 映射为像面半径 $r=g(\theta)$，再写成

$$
x_d=r\cos\phi,\qquad y_d=r\sin\phi.
$$

典型理想投影律为：

| 模型 | 半径函数 $r=g(\theta)$ | 特征 |
| --- | --- | --- |
| 透视/rectilinear | $f\tan\theta$ | 直线保持直线，$\theta\to90^\circ$ 时发散 |
| 等距/equidistant | $f\theta$ | 像面距离与角度成正比 |
| 等立体角/equisolid-angle | $2f\sin(\theta/2)$ | 相等立体角对应相等像面面积 |
| 体视/stereographic | $2f\tan(\theta/2)$ | 保角 |
| 正交/orthographic | $f\sin\theta$ | 边缘压缩明显 |

因此，“鱼眼”不是唯一公式，而是一类广角投影。标定的任务就是确定实际镜头最符合哪个参数化函数。

### 4.2 Kannala–Brandt / OpenCV fisheye 的径向模型

常用的奇次多项式写成

$$
\theta_d
=\theta\left(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8\right).
$$

再令

$$
x_d=\theta_d\frac X\rho,\qquad
y_d=\theta_d\frac Y\rho,
$$

最后应用内参：

$$
u=f_x(x_d+\alpha y_d)+c_x,\qquad
v=f_y y_d+c_y.
$$

其中 $\alpha$ 是 OpenCV 文档采用的 skew 参数。若 $\rho\to0$，利用极限 $\theta_d/\rho\to1/Z$，光轴点应映射到主点，程序中要显式处理这一数值极限。

OpenCV 常把前两步写成 $a=X/Z,b=Y/Z,r=\sqrt{a^2+b^2},\theta=\arctan r$。当 $Z>0$ 时，这与 $\operatorname{atan2}(\rho,Z)$ 等价；后者更直接揭示角度的几何意义，也避免把该公式误用到 $Z\le0$。是否允许光轴后方的射线，还取决于镜头实际视场和具体实现，不能只凭“鱼眼超过 $180^\circ$”就保留所有点。

### 4.3 它与普通径向畸变模型的本质差别

普通 pinhole + RadTan 模型通常先取 $x=X/Z,y=Y/Z$，再对平面半径 $r=\sqrt{x^2+y^2}$ 做多项式修正。鱼眼模型则直接围绕光线角度 $\theta$ 建模。

在小角度下 $\tan\theta\approx\theta$，两者看起来相似；在大视场边缘，$\tan\theta$ 与 $\theta$ 差异迅速增大。把两组看起来都叫 $k_1,k_2,\ldots$ 的参数互换，不具有数学意义。

## 5. 鱼眼反投影：像素到空间射线

先去掉内参。忽略 skew 时：

$$
x_d=\frac{u-c_x}{f_x},\qquad
y_d=\frac{v-c_y}{f_y},\qquad
r_d=\sqrt{x_d^2+y_d^2}.
$$

然后求解标量方程

$$
F(\theta)
=\theta+k_1\theta^3+k_2\theta^5+k_3\theta^7+k_4\theta^9-r_d=0.
$$

它通常没有方便的解析逆，可用 Newton 法：

$$
\theta_{n+1}=\theta_n-
\frac{F(\theta_n)}
{1+3k_1\theta_n^2+5k_2\theta_n^4+7k_3\theta_n^6+9k_4\theta_n^8}.
$$

得到 $\theta$ 后，令 $\phi=\operatorname{atan2}(y_d,x_d)$，单位射线为

$$
\mathbf d_C=
\begin{bmatrix}
\sin\theta\cos\phi\\
\sin\theta\sin\phi\\
\cos\theta
\end{bmatrix}.
$$

中心点 $r_d=0$ 的极限是 $(0,0,1)^\top$。为了反投影唯一且 Newton 法稳定，在使用角度范围内还应检查径向函数单调性：

$$
\frac{d\theta_d}{d\theta}
=1+3k_1\theta^2+5k_2\theta^4+7k_3\theta^6+9k_4\theta^8>0.
$$

若该导数为零或变负，同一像面半径可能对应多个角度，模型会发生折叠。

## 6. 雅可比：优化如何“看见”几何误差

### 6.1 针孔投影对三维点的导数

$$
\frac{\partial(u,v)}{\partial(X,Y,Z)}=
\begin{bmatrix}
f_x/Z&0&-f_xX/Z^2\\
0&f_y/Z&-f_yY/Z^2
\end{bmatrix}.
$$

当 $Z$ 很小时导数爆炸，说明近平面附近投影病态；当 $Z$ 很大时，平移引起的像素变化较小，深度和平移更难估计。

### 6.2 鱼眼径向链式求导

以 $\mathbf q=(X,Y)^\top$、$\rho=\|\mathbf q\|$、$s(\rho,Z)=\theta_d/\rho$ 表示

$$
\begin{bmatrix}x_d\\y_d\end{bmatrix}=s\mathbf q.
$$

则

$$
d(s\mathbf q)=s\,d\mathbf q+\mathbf q\,ds,
$$

并且

$$
d\theta=\frac{Z\,d\rho-\rho\,dZ}{\rho^2+Z^2},
\qquad
d\rho=\frac{X\,dX+Y\,dY}{\rho},
$$

$$
d\theta_d=
\left(1+3k_1\theta^2+5k_2\theta^4+7k_3\theta^6+9k_4\theta^8\right)d\theta,
$$

$$
ds=\frac{\rho\,d\theta_d-\theta_d\,d\rho}{\rho^2}.
$$

这些式子与像素内参的线性导数复合，就得到 $\partial(u,v)/\partial(X,Y,Z)$。实际实现应在 $\rho\approx0$ 使用级数极限，避免除零。

### 6.3 位姿扰动的导数

若采用左扰动

$$
\mathbf T' = \exp(\delta\boldsymbol\xi^\wedge)\mathbf T,
\qquad
\delta\boldsymbol\xi=
\begin{bmatrix}\delta\boldsymbol\omega\\\delta\mathbf t\end{bmatrix},
$$

相机点的一阶变化为

$$
\delta\mathbf p_C
=- [\mathbf p_C]_\times\delta\boldsymbol\omega+\delta\mathbf t,
$$

所以

$$
\frac{\partial\mathbf u}{\partial\delta\boldsymbol\xi}
=\frac{\partial\pi}{\partial\mathbf p_C}
\begin{bmatrix}-[\mathbf p_C]_\times&\mathbf I\end{bmatrix}.
$$

若换成右扰动、不同的 twist 排列或不同变换方向，雅可比会改变。记公式前必须先写清扰动约定。

## 7. 标定的数学本质

给定已知三维标定点 $\mathbf P_i$ 与观测像素 $\hat{\mathbf u}_{ij}$，联合估计内参、鱼眼参数与第 $j$ 帧标定板位姿：

$$
\min_{\mathbf K,\mathbf k,\{\mathbf T_j\}}
\sum_{i,j}
\rho_{\text{robust}}\!\left(
\left\|
\hat{\mathbf u}_{ij}
-\pi_{\mathbf K,\mathbf k}(\mathbf T_j\mathbf P_i)
\right\|_2^2
\right).
$$

这就是重投影误差最小化。参数可估计不等于参数可稳定估计：若标定板只出现在画面中心，高阶 $k_i$ 几乎不受约束；若所有姿态都近似正对相机，焦距、深度与外参之间会强耦合。鱼眼标定尤其需要覆盖边缘和多种倾角。

## 8. 常见误区与可观察症状

| 误区 | 常见症状 | 数学检查 |
| --- | --- | --- |
| 把 ${}^A\mathbf T_B$ 当成相反方向 | 点云整体跑到相机背后或镜像 | 检查原点与三个单位轴变换结果 |
| 复合顺序写反 | 单个外参看似合理，组合后完全错误 | 检查相邻上下标能否消去 |
| 逆变换只写 $-\mathbf t$ | 旋转越大，平移错位越明显 | 使用 $-\mathbf R^\top\mathbf t$ |
| 混淆度和弧度 | 鱼眼投影半径异常大 | $\theta$ 多项式必须用弧度 |
| 混用 RadTan 与 fisheye 参数 | 中心较准、边缘系统性弯曲 | 对照投影变量是 $r$ 还是 $\theta$ |
| 忽略时间戳 | 行驶或转弯时出现方向性拖影 | 显式写出 $t_1,t_2$ 的 ego pose |
| 不处理 $\rho\to0$ | 主点附近出现 NaN | 使用解析极限或小量分支 |
| 只看平均重投影误差 | 平均值不错但边缘明显失真 | 按像面半径画残差与方向 |
| 认为像素能唯一恢复三维点 | 反投影只有方向，没有距离 | 明确射线中的尺度 $\lambda$ |

## 9. 自测与练习

### 概念自测

1. ${}^A\mathbf t_B$ 是哪个原点在哪个坐标系下的坐标？
2. 为什么方向向量的齐次坐标最后一维是 $0$？
3. 为什么针孔模型在 $90^\circ$ 附近发散，而等距鱼眼模型不会？
4. 为什么 OpenCV fisheye 的 $k_1$ 不能直接放入普通 RadTan 模型？
5. 鱼眼反投影为什么需要检查 $d\theta_d/d\theta>0$？

### 推导练习

1. 从 ${}^A\mathbf p={}^A\mathbf R_B{}^B\mathbf p+{}^A\mathbf t_B$ 推导完整逆变换。
2. 推导上文从 $L(t_1)$ 到 $C(t_2)$ 的变换链，并逐个消去坐标系标记。
3. 证明小角度下等距模型与针孔模型一阶等价，并比较它们的三阶项。
4. 从 $x_d=sX,y_d=sY$ 完成鱼眼投影的 $2\times3$ 雅可比矩阵。
5. 用有限差分检查解析雅可比；相对误差应随步长减小而下降，直到浮点消减误差占主导。

### 建议的最小数值实验

- 构造 $SO(3)$ 矩阵，验证 $\mathbf R^\top\mathbf R=\mathbf I$、$\det\mathbf R=1$；
- 随机生成 $SE(3)$，验证 $\mathbf T^{-1}\mathbf T=\mathbf I$；
- 将同一组光线分别通过 $f\tan\theta$ 与 $f\theta$ 投影，画出半径差随视场角的变化；
- 对鱼眼投影后再 Newton 反投影，检查射线夹角误差；
- 人为交换一个外参方向，观察其误差模式与错误相机模型的误差模式有何不同。

## 相关知识与资料

- [自动驾驶世界模型综述笔记](../../papers/autonomous-driving-world-models/2025-survey-world-models-autonomous-driving/README.md)：多传感器时空表示依赖一致的坐标系与运动补偿。
- [Kannala 与 Brandt 原论文](https://doi.org/10.1109/TPAMI.2006.153)：适用于常规、广角和鱼眼镜头的通用中心相机模型。
- [OpenCV fisheye 官方模型说明](https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html)：给出 OpenCV 实现使用的投影公式、参数和接口约定。

## 下一步

建议依次完成：

1. 不看答案重写一次从 LiDAR 到鱼眼像素的完整公式；
2. 实现 $SE(3)$ 的变换、求逆和复合，并用单位轴进行测试；
3. 实现鱼眼投影、Newton 反投影和有限差分雅可比检查；
4. 再进入外参标定、PnP、bundle adjustment、时间同步与运动补偿。
