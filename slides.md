---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
transition: slide-left
css: unocss
title: Deep Learning from Scratch
---

# Deep Learning from Scratch

밑바닥부터 시작하는 딥러닝

<!-- <div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div> -->

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/LeeByeongdo/deeplearningfromscratch" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>


--- 

# 목표

간단한 숫자 이미지 인식 신경망 모델을 예시로, 실제로 학습이 어떻게 구현되는지 알아봅니다.

<div grid="~ cols-2 gap-2" m="-t-2">
<div>

### 목차

- 👨‍💻 **개발 환경**
- 🔣 **퍼셉트론**
- 🧠 **신경망**
- 🏋🏿 **활성화 함수**
- 📖 **신경망 학습**
- ⟷ **손실함수**
- ∂ **경사하강법**
- 🎬 **완성**
</div>

<img style="width: 70%" src="https://camo.githubusercontent.com/cf3b516aff9e645a93048bc1da870e6cfc66294c028a9512d7c8a2b48283e075/68747470733a2f2f692e696d6775722e636f6d2f55346a324a59542e6a7067" />

</div>

<!--
You can have `style` tag in markdown to override the style for the current page.
Learn more: https://sli.dev/guide/syntax#embedded-styles
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
- 언어로는 파이썬을 사용하며, 행렬 계산을 위해 numpy 라이브러리를 사용합니다.
- 다수의 입력을 받아 결과를 출력하는 퍼셉트론에 대해 알아봅니다.
- 퍼셉트론의 개념을 기반으로 신경망에 대해 이해합니다.
- 신경망에서 사용하는 활성화 함수에 대해 알아봅니다.
- 데이터로부터 어떻게 학습을 구현하는지 알아봅니다.
- 학습에서 사용하는 손실함수에 대하여 알아봅니다.
- 계산 그래프 방식으로 오차역전파에 대하여 이해합니다.
- 위의 내용을 종합하여 신경망을 완성합니다.
-->

---

# 👨‍💻 개발환경


|     |     |
| --- | --- |
| python| python은 기계학습과 데이터 과학 분야에서 널리 쓰이는 언어입니다. |
| numpy | 수치 계산과 통계 처리를 해주는 라이브러리입니다. |
| matplotlib.pyplot | 그래프를 그려주는 라이브러리입니다. |
| mnist | 학습에 필요한 데이터를 제공해줍니다. |

---

# 퍼셉트론

퍼셉트론은 다수의 입력을 받아 하나의 활성 또는 비활성 상태를 신호로 나타냅니다. (0 또는 1)

<br>

<div grid="~ cols-2 gap-2" m="-t-2">

<img border="rounded" style="width: 90%" src="https://compmath.korea.ac.kr/appmath2021/_images/fig2-1.png">

$$
y = \begin{cases}
   0 &(w_1x_1 + w_2y_2 \leq \theta) \\
   1 &(w_1x_1 + w_2y_2 > \theta)
\end{cases} \\
\begin{align*} \\
&x_1, x_2: 입력 \\
&w_1, w_2: 가중치 \\
&\theta: 입계값
\end{align*}
$$

</div>

---

# 퍼셉트론 예제 - AND 게이트

퍼셉트론을 이용하여 AND 게이트를 구현할 수 있습니다.

$$
y = \begin{cases}
   0 &(w_1x_1 + w_2y_2 \leq \theta) \\
   1 &(w_1x_1 + w_2y_2 > \theta)
\end{cases}  

\enspace\enspace\enspace\enspace➡\enspace\enspace\enspace\enspace

y = \begin{cases}
   0 &(w_1x_1 + w_2y_2 + b \leq 0) \\
   1 &(w_1x_1 + w_2y_2 + b > 0)
\end{cases} \\
$$

<br />


<div grid="~ cols-2 gap-2" m="-t-2">

| $x_1$ | $x_2$ | y |
| ----- | ----- | - |
| 0     | 0     | 0 |
| 1     | 0     | 0 |
| 0     | 1     | 0 |
| 1     | 1     | 1 |

<div>

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

비슷한 방식으로 NAND, OR 게이트도 만들 수 있습니다.

</div>

</div>

<br />



---

# 퍼셉트론의 한계 

퍼셉트론으로는 선형적인 식만 표현할 수 있다. XOR 게이트를 만들어 볼까요?

$$
y = \begin{cases}
   0 &(w_1x_1 + w_2y_2 + b \leq 0) \\
   1 &(w_1x_1 + w_2y_2 + b > 0)
\end{cases}
$$

<div grid="~ cols-2 gap-2" m="-t-2">

| $x_1$ | $x_2$ | y |
| ----- | ----- | - |
| 0     | 0     | 0 |
| 1     | 0     | 1 |
| 0     | 1     | 1 |
| 1     | 1     | 0 |

<img border="rounded" src="/images/xor.png">

</div>

---

# 다층퍼셉트론

하나로 안되면 여러개 쓰면 됩니다. 사실 퍼셉트론의 참맛은 여러개를 쌓는것입니다.

<br />

<div grid="~ cols-2 gap-2" m="-t-2">

| $x_1$ | $x_2$ | y |
| ----- | ----- | - |
| 0     | 0     | 0 |
| 1     | 0     | 1 |
| 0     | 1     | 1 |
| 1     | 1     | 0 |

<div>
<img border="rounded" style="width:100%" src="https://upload.wikimedia.org/wikipedia/commons/a/a2/254px_3gate_XOR.jpg">

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```
</div>

</div>

---

# 다층퍼셉트론

<br />
<div grid="~ cols-2 gap-2" m="-t-2">

<div>
<img border="rounded" style="width:100%" src="https://upload.wikimedia.org/wikipedia/commons/a/a2/254px_3gate_XOR.jpg">

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```
</div>

<img border="rounded" style="width:100%" src="/images/multilayer-perceptron.png">

</div>

---

# 신경망

<img border="rounded" src="https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg">

---

# 활성화 함수

<br />

$$
y = \begin{cases}
   0 &(w_1x_1 + w_2y_2 + b \leq 0) \\
   1 &(w_1x_1 + w_2y_2 + b > 0)
\end{cases}
$$

<br />
<h2 style="text-align: center;">⬇️</h2>
<br />

$$
y = h(w_1x_1 + w_2y_2 + b) \\
$$

$$
h(x) = \begin{cases}
   0 &(x \leq 0) \\
   1 &(x > 0)
\end{cases}
$$

<br />

<arrow v-click="1" x1="700" y1="480" x2="580" y2="430" color="#63489a" width="3" arrowSize="1" />

---

# 활성화 함수

<br />

<div grid="~ cols-2 gap-2" m="-t-2">

<div>

$$
a = h(w_1x_1 + w_2y_2 + b) \\
$$

$$
y = h(a)
$$

<br />

활성화함수(h)는 학습에 효과가 좋은 다양한 함수가 쓰일 수 있습니다.

- Sigmoid
- Relu

</div>

<img border="rounded" src="/images/activation-function.png">

</div>

---

# 활성화 함수 - Sigmoid

<br />

<div grid="~ cols-2 gap-2" m="-t-2">

<div style="display:flex;flex-direction:column;align-items:center;">

$$
h(x) = {1 \above{1pt} 1 + exp(-x)}
$$

<br />

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
```

</div>

<img border="rounded" src="/images/sigmoid.png">

</div>

---

# 활성화 함수 - Relu

<br />

<div grid="~ cols-2 gap-2" m="-t-2">

<div style="display:flex;flex-direction:column;align-items:center;">

$$
h(x) = \begin{cases}
   0 &(x \leq 0) \\
   x &(x > 0)
\end{cases}
$$

<br />

```python
def relu(x):
    return np.maximum(0, x)
```

</div>

<img border="rounded" src="/images/relu.png">

</div>

---

# 출력층에서의 활성화 함수

신경망은 분류와 회귀 모두에 사용될 수 있습니다. 어떤 문제냐에 따라 적절한 활성화 함수를 사용해야 합니다.

- 분류 -> Softmax
- 회귀 -> 항등함수

<br />


<div grid="~ cols-2 gap-2" m="-t-2" style="width:80%;margin:auto;">

<div style="display:flex;flex-direction:column;align-items:center;">

#### 항등 함수

$$
y_k = x_k
$$

<br />

```python
def identity_function(x):
    return x
```

</div>

<div style="display:flex;flex-direction:column;align-items:center;">

#### 소프트맥스 함수

$$
y_k = {exp(a_k) \above{1pt} \sum_{i=1}^{n} exp(a_i)}
$$

<br />

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

</div>

</div>

---

# MNIST DATASET

학습을 위한 손글씨 숫자 이미지 데이터 모음

<br />


<div grid="~ cols-2 gap-2" m="-t-2">

- 28 x 28 크기의 회색조 이미지 (1채널)
- 각 픽셀은 0에서 255까지의 값
- 실제 이미지가 의미하는 숫자(정답) 레이블 포함
- 6만개의 학습용 데이터와 1만개의 시험용 데이터

<img border="rounded" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png">

</div>

---

# 신경망 학습

학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것입니다.

<img border="rounded" src="/images/neural-network.png" style="width: 80%;" />


---

# 손실함수

현재의 상태를 나타내는 지표로 사용됩니다. (실제 정답과 현재 신경망에서 도출된 결과의 차이)

- 평균 제곱 오차
- 교차 엔트로피 오차


<br />

<div grid="~ cols-2 gap-2" m="-t-2" style="width:80%;margin:auto;">

<div style="display:flex;flex-direction:column;align-items:center;">

#### 평균 제곱 오차

$$
E = {1\above{1pt}\enspace2\enspace }\sum_k(y_k - t_k)^{2}
$$

<br />

```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

</div>

<div style="display:flex;flex-direction:column;align-items:center;">

#### 교차 엔트로피 오차

$$
E = -\sum_{k}t_klogy_k
$$

<br />

```python
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))
```

</div>

</div>

---

# 미분

<br />

<div grid="~ cols-2 gap-2" m="-t-2" style="width:100%;margin:auto;">

<div style="display:flex;flex-direction:column;align-items:center;">

#### 미분

한순간의 변화량

$$
{df(x)\above{1pt}dx}=\lim\limits_{h->0}{f(x+h) - f(x)\above{1pt}h}
$$

<br />

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
```

</div>

<div style="display:flex;flex-direction:column;align-items:center;">

#### 편미분

변수가 여러개인 경우에 각 변수에 대한 변화량을 구하는 방법

$$
f(x_0, x_1) = x_0^2 + x_1^2
$$

예시) $x_0$ = 3, $x_1$=4에서 각각의 편미분

$$
{\partial{f}\above{1pt}\partial{x_0}} = 2x_0 + 4^2 \enspace\enspace\enspace\enspace {\partial{f}\above{1pt}\partial{x_1}} = 3^2 + 2x_1
$$

</div>

</div>

---

# 경사법(경사 하강법)

기울기를 이용해 함수의 최솟값을 찾는 방법입니다.

<br />

<div grid="~ cols-2 gap-2" m="-t-2" style="width:100%;margin:auto;">

<div style="display:flex;flex-direction:column;align-items:center;">

#### 기울기 (모든 매개변수에 대한 편미분 집합)

```python
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

    return grad
```

</div>

<div style="display:flex;flex-direction:column;align-items:center;">

#### 매개변수 조정

$$
x_0 = x_0 - \eta{\partial{f}\above{1pt}\partial{x_0}}
$$

$$
x_1 = x_1 - \eta{\partial{f}\above{1pt}\partial{x_1}}
$$

$\eta$ = 학습률 (0.001, 0.0001 등)

</div>

</div>

---
layout: center
class: text-center
---

# 코드 실행해보기


---
layout: center
class: text-center
---

# 감사합니다
