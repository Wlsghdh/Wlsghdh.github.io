# A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

**저자:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton (Google Brain)  
**발표:** ICML 2020

---

## 1. 연구 배경 및 동기

레이블 없이 시각적 표현을 학습하는 것은 오랜 난제였다. 기존의 자기지도학습(self-supervised learning) 방법들은 크게 두 갈래로 나뉜다.

생성 모델 기반 접근법은 픽셀 수준의 복원을 목표로 하는데, 계산 비용이 크고 표현 학습에 반드시 필요하지 않을 수 있다. 대조 학습(contrastive learning) 기반 판별 접근법은 최근 유망한 결과를 보이고 있으나, 대부분 특수한 아키텍처나 메모리 뱅크를 필요로 했다.

SimCLR은 이러한 복잡성을 제거하고, 단순하고 통합된 프레임워크를 제안한다.

---

## 2. 모델 구조

SimCLR은 네 가지 핵심 구성 요소로 이루어진다.

### 2.1 Data Augmentation Module

하나의 이미지 $x$로부터 두 개의 서로 다른 augmented view $\tilde{x}_i$, $\tilde{x}_j$를 생성한다. 사용되는 augmentation은 random cropping + resize, random color distortion, random Gaussian blur를 순차 적용한 조합이다.

두 view의 쌍을 positive pair로 정의하며, 같은 배치 내 다른 이미지들로부터 생성된 view들은 negative pair가 된다.

### 2.2 Base Encoder $f(\cdot)$

ResNet을 기본 인코더로 사용하며, average pooling layer 이후의 출력 $h_i = f(\tilde{x}_i) \in \mathbb{R}^d$를 representation으로 사용한다. 아키텍처 선택에 제약이 없어 다양한 네트워크를 사용할 수 있다.

### 2.3 Projection Head $g(\cdot)$

representation $h$를 contrastive loss가 적용되는 공간으로 매핑하는 소규모 MLP이다.

$$z_i = g(h_i) = W^{(2)}\sigma(W^{(1)}h_i)$$

하나의 hidden layer와 ReLU activation $\sigma$로 구성되며, 학습 완료 후 downstream task에서는 제거하고 $h$만 사용한다.

![SimCLR Architecture](images/reviews/simclr-architecture.png)
*SimCLR 전체 프레임워크 구조*

### 2.4 Contrastive Loss (NT-Xent)

Normalized Temperature-scaled Cross Entropy Loss를 사용한다. positive pair $(i, j)$에 대한 loss는 다음과 같이 정의된다.

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

여기서 $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$은 cosine similarity이며, $\tau$는 temperature parameter다. 배치 내 $2(N-1)$개의 augmented example이 자동으로 negative sample로 활용된다.

---

## 3. 핵심 발견

### 3.1 Data Augmentation의 조합이 결정적이다

단일 augmentation은 좋은 표현을 학습하지 못한다. Random cropping과 color distortion의 조합이 특히 중요한데, cropping만 사용할 경우 모델이 color histogram이라는 shortcut에 의존하게 되기 때문이다.

또한 contrastive learning은 supervised learning보다 더 강한 color augmentation이 필요하다.

### 3.2 Nonlinear Projection Head가 표현 품질을 향상시킨다

projection head를 사용하지 않는 경우 대비 10% 이상, linear projection 대비 3% 향상이 나타난다.

흥미롭게도 projection head 이전 레이어 $h$가 projection 이후 $z$보다 훨씬 더 좋은 표현을 학습한다. 이는 $g(h)$가 contrastive loss에 의해 transformation에 대한 불변성을 갖도록 학습되면서 downstream task에 유용한 정보(색상, 방향 등)를 손실하기 때문으로 해석된다.

### 3.3 Batch Size와 학습 시간의 영향

큰 batch size는 더 많은 negative sample을 제공하여 학습 초기에 큰 이점을 준다. 학습 epoch이 증가할수록 batch size 간 성능 차이는 줄어든다.

Supervised learning과 달리, contrastive learning에서의 larger batch는 단순한 gradient 안정화 효과를 넘어 negative sample의 다양성 확보라는 구조적 이점을 갖는다.

### 3.4 모델 크기의 효과

모델이 클수록 성능이 향상되며, supervised learning과의 성능 격차가 더 빠르게 줄어든다. 이는 contrastive learning이 supervised learning보다 더 큰 모델에서 이득을 본다는 것을 의미한다.

---

## 4. 실험 결과

Linear Evaluation (ImageNet): ResNet-50 기준 69.3% top-1 accuracy로 이전 최고 성능(CPCv2, 63.8%) 대비 큰 향상을 이루었다. ResNet-50(4$\times$) 기준으로는 76.5%를 달성하여, supervised ResNet-50의 성능과 동등한 수준에 도달했다.

Semi-supervised Learning: 1% label만 사용했을 때 ResNet-50(4$\times$)으로 85.8% top-5 accuracy를 달성하여 AlexNet 대비 100배 적은 label로 더 높은 성능을 보였다.

Transfer Learning: 12개 자연 이미지 분류 데이터셋에서 fine-tuning 시 supervised baseline 대비 5개 데이터셋에서 유의미하게 우수하고, 5개에서 통계적으로 동등한 성능을 보였다.

---

## 5. 한계 및 의의

SimCLR은 여러 우수한 설계 선택들을 하나의 프레임워크로 통합했다는 점에서 의미가 크다. 다만 좋은 성능을 위해 매우 큰 batch size(4096~8192)가 필요하다는 점은 실용적 한계로 작용한다. 이는 대규모 TPU 인프라 없이는 재현하기 어렵다는 것을 의미하며, 이후 MoCo v2, BYOL 등의 후속 연구들이 메모리 효율적인 대안을 제시하는 배경이 된다.

또한 negative sample의 품질이 학습에 큰 영향을 미치는 구조이기 때문에, false negative 문제(같은 클래스의 다른 이미지가 negative로 취급되는 경우)에 대한 명시적 대응이 없다는 점도 향후 연구 과제로 남는다.

전체적으로 SimCLR은 self-supervised visual representation learning 분야의 발전 방향을 명확히 제시했으며, 이후 대조 학습 연구의 중요한 기준점이 되었다.
