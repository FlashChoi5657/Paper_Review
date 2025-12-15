## Segment Any Motion in Videos

### Introduction
- 움직이는 물체의 Segmentation이 어려운 이유는 복잡한 형태, motion blur, 배경에 의한 왜곡이다. 그래서 DINO-based 의미론적 feature와 SAM2의 pixel-level 정밀한 mask를 이용한다.
- **DINO(=self-Distillation with No Labels): Vision Transformer를 라벨 없이 학습시키는데 teacher-student 구조로 같은 이미지의 다른 augmentation을 보고 일관된 representation을 학습한다. DINO로 학습된 ViT의 특징은 명시적인 supervision 없이도 semantic 정보를 담고 있다. 예로 attention map을 시각화 하면 object의 경계나 의미적으로 유사한 부분들이 자연스럽게 클러스트링 된다. 이미지 자체 구조에서 학습신호를 만든다. 예를 들어, 같은 이미지를 두 가지 다른 방식의 증강을 하면 global view와 local view가 생기는데 모델은 이 둘이 같은 이미지에서 왔으니 비슷한 representation을 학습하라고 시킨다. teacher network는 global view를 보고 student network는 local view를 본다. student는 teacher의 output을 따라가도록 학습하고 teacher는 student의 Exponential moving average로 업데이트 된다. 결과적으로 고양이 귀의 일부분만 봐도 고양이 전체와 관련있다는 것을 학습할 수 있다.**
- Point tracking은 deformation과 occulusion에 강인한 장기적 pixel motion 정보를 포착(exploit)한다. 여기에 의미론적 context를 추가하기 위해 DINO feature를 추가한다. 긴 2D track 셋이 주어지면 모델은 움직이는 물체에 해당하는 track을 식별한다. 이러한 sparse point들을 prompt로 주고 dense pixel-level mask로 확장한다.
- **정리하면, 뭐가 움직이는 물체인지 판단하고 SAM2는 그 판단 결과를 정밀한 segmentation mask로 만들어준다.**
- motion 단서와 semantic 정보를 균형있게 조절하기 위해 두 가지 모듈을 제안한다.
- **시공간적(Spatio-Temporal) 추적 어텐션은 주어진 입력 track의 long-term 특성을 고려하여 서로 다른 궤적간의 관계를 파악하는 spatial attention과 하나의 궤적이 시간에 따라 어떻게 변하는지 파악하는 temporal attention을 통합한다.**
- **Motion-Semantic Decoupled Embedding은 모션 패턴의 우선순위를 정하고 보조경로에서 semantic feature를 처리하는 특수 attention 메커니즘**
- 합성데이터와 실제 데이터를 포함한 광범위한 데이터셋으로 모델을 학습했다. DINO feature의 self-supervised 특성은 강력한 일반화 성능을 보여준다.
- 기존 Moving Object Segmentation(MOS)는 optical flow를 활용했는데 optical flow는 단기적 motion에만 국한 되어 long-term에서는 파악이 어렵다. 또한 Spectral clustering은 point trajectory를 사용하는데 복잡한 모션을 처리하는데 어려움을 겪는다. 
- **Optical Flow는 비디오에서 픽셀들이 프레임 간에 어디로 이동했는지 나타내는 2D vector field이다. 예로, 프레임1에서 프레임2로 넘어갈 때 어떤 픽셀이 x+3, y-2로 움직였다를 계산한다. 결과물은 이미지와 같은 크기의 (dx, dy) 벡터맵이다.
- 다른 방법들은 motion cue와 외형 정보를 함께 사용하는데, 두 개를 별개의 단계에서 순차적으로 처리한다. 이 방법은 두 정보를 보완적으로 사용하는 효과를 제한한다. 
- 위 단점들의 보완을 위해 motion(point tracking)과 semantic(DINO feature)를 high level에서 통합한다.
- motion label은 trajectory가 움직이는 물체에 속하는지 아닌지를 나타내는 이진분류 맵이다. 여기서는 기존처럼 affinity matrix와 spectral clustering을 쓰는 대신 motion-semantic decoupled 임베딩 방법을 제안한다. motion 정보와 semantic 정보를 따로 임베딩했다가 결합한다.

### RELATED WORK
- Flow-based Moving Object Segmentation: 이 방식은 통계적 추론이나 반복적 최적화 방식의 모션 추정 모델이었다. 최근에는 CNN encoder나 transformer를 이용해 motion 단서를 얻어 분할하는 딥러닝 방식이 사용됐다. optical-flow는 다른 장면에서 움직임이 있는 개별 object를 구분하고 변화를 감지해야 하는데 그동안은 밝기 변화, 원근감에 취약했고 짧은 장면에서 제한된 성능을 냈다.
- Trajectory-based Moving Object Segmentation: 이 방법은 두 개의 프레임 또는 여러 프레임을 사용하는 방법으로 나뉜다. 두 프레임을 사용할 때는 CNN을 이용해 에너지 최소화로 모션을 추정하고, 여러 프레임을 이용할 때는 affinity metrics에 기반한 spectral clustering을 적용한다. affinity metrices
- **affinity matrices**: 데이터 포인트들 간의 유사도를 행렬로 표현. n개의 점이 있다면 n x n 행렬이 되고 (i,j) 원소는 i와 j가 얼마나 유사한지 나타낸다. 값이 클수록 유사하다. Spectral clustering은 이 유사구조를 자동으로 찾아주는 기법이다. geometric model을 이용한다면 여러 점들의 움직임이 특정 기하학적 변환을 따르는지 최적화 하는 것이다. RANSAC 같은 방법을 이용하면 같은 geometric model을 따르므로 같은 물체라고 그룹화할 수 있다.**
- motion model은 최근 많은 발전이 있는데, trifocal tensor를 분석하는 기법은 세 개의 이미지 매칭을 더 잘해낸다. 그러나 카메라가 일직선일 때 성능이 감소한다. 다양한 motion 단서를 통합하는 기법은 point 궤적과 optical flow를 결합해 두 개의 affinity matrices가 공동으로 규제하는 multi-view spectral clustering을 사용한다. 그러나 이 기법들은 affinity matrices가 가지는 고유한 문제들을 갖는다. 예를 들어, 지역적인 유사성만 포착하여 일관성 없는 분할로 이어지거나, 움직임의 변화가 갑자기 달라지면 파악하는데 어려움을 겪고 있다.
- Unsupervised Video Object Segmentation(VOS): 이 방식은 video 영상에서 자동으로 눈에 띄는 물체를 추적하는 것이다. 반면에 semi-supervised VOS는 첫 프레임의 ground truth 주석에 의존해 연속적인 프레임에서 물체를 분할하는 것이다. 이 연구는 MOS로 다른 VOS와 다른 관점을 갖고 있다.

### METHOD
- 목적은 비디오가 입력되면 움직이는 물체의 pixel 단위 mask를 만드는 것. 전체 파이프라인에서 long-range track은 비디오를 이해하고 motion pattern을 포착하며 long-range prompts는 시각적 분할에서 중요한 역할을 한다.
- **Motion Pattern Encoding**: Point trajectories 모션을 이해하는데 중요한 정보를 제공하며 MOS 기법은 2프레임과 multi-frame의 두가지로 나눌 수 있다. 2-frame은 시간 불일치가 심하고 입력 flow에 노이즈가 많을 때 성능이 저하된다. multi-frame은 노이즈에 매우 민감하고 복잡한 패턴을 처리하기 어렵다. 이러한 한계를 해결하기 위해 특수 궤적 처리 모델을 통해 처리된 long-range point track을 활용하여 궤젹 별 모션 label을 예측하는 방법을 제안한다. 제안된 신경망은 encoder-decoder 구조이며 encoder는 long-range trajectory를 처리하고, 전체 궤적을 시공간 궤적 attention에 적용한다. 각 궤적의 Motion pattern을 임베딩 하기 위해 시간과 공간 단서를 통합하며 local and global 정보를 시공간에서 포착한다. 
- long-range 궤적의 정확도가 모델의 성능에 미치는 영향이 매우 크기 떄문에 매 시간에서 각 track의 confidence score를 만들 수 있는 BootsTAP을 이용해 낮은 confidence score를 커버하도록 하였다. long-range track은 물체가 움직이면서 카메라에서 벗어날 수도 있고 다른 물체와 겹칠 수도 있어 이러한 불규칙한 데이터를 다루기 위해 NLP 접근 방식과 유사한 sequence model인 transformer model을 사용하게 만드는 계기가 되었다. 
- **BootsTAP**: DeepMind에서 만든 long-range tracking모델로 Tracking Any Point 시리즈 중 하나. 비디오에서 임의의 point를 지정하면 전체 프레임에 걸쳐 어디로 이동하는지 추적해준다. 
- long-range 궤적의 입력데이터는 정규화된 pixel 좌표계($u_i$, $v_i$), visibility $\rho_i$, confidence score $c_i$ 이다($i$는 time step). 마스크는 안 보이거나 낮은 confidence일 때도 pixel 좌표를 가리키는데 적용된다. 또한 Depth-Anything으로 추정된 monocular depth map $d_i$ 를 통합한다. $d_i$는 3D 장면 구조를 해석하고 공간 layout과 occulusion에 대한 이해를 향상시킨다. 
- **Depth-Anything**: 단일 이미지에서 depth map을 추정하는 모델로 RGB 이미지 한 장에서 각 픽셀이 카메라에서 얼마나 먼가를 나타내는 depth map을 출력한다. 
- 추가로 입력데이터와 시간단위의 움직임 단서를 강화하기 위해 인접 프레임 간의 trajectory 좌표의 변화와 깊이 변화를 계산했다. 좌표에서 인접한 샘플링 포인트는 공간적으로 가까운 feature의 oversmoothing을 초래할 수 있으므로, NeRF와 비슷하게 위치 인코딩에 주파수 변환을 적용하여 세밀한 공간적 디테일을 효과적으로 포착하게 한다.
- 최종적으로 강화된 trajectories는 두 MLP를 지나며 중간 feature를 생성하여 transformer의 인코더에 입력한다. 입력데이터의 장기적 특성을 고려하여 인코더 $\mathcal{E}(x)$에 **Saptio-Temporal Trajectory Attention** 를 제안하였다. 이 Attention layer는 track과 temporal 차원을 번갈아 가며 작동하는데 한 시점에서 여러 trajectory의 관계를 보고 시간에 따라 어떻게 변화하는지 포착할 수 있다. 한 번에 두 연산을 처리하면 연산량이 커져 번갈아 처리하게 한다. 마지막으로 각각의 포인트보다 전체 trajectory를 표현하는 feature를 얻기 위해 시간차원을 따라 maxpooling을 넣어준다. 매 frame마다 feature가 있는데 이 시점 별 feature를 시간 차원으로 하나의 feature vector로 압축하는 것이다. 
- **2. Per-trajectory Motion Prediction**: motion pattern 인코딩 만으로 물체를 구분하기는 어렵다. 고도로 추상화 된 궤적에서 물체의 움직임과 카메라 모션을 구분하는 학습이 모델에 어렵기 때문이다. 여기에는 texture, 외형, 의미적 정보를 추가하여 움직일 지 움직이지 않을지 정보를 제공하면 motion 분류를 더 쉽게 만들 수 있다. 
- semantic 분할 방법은 미리 정의된 class에 의존한다. "사람은 움직인다"고 가정하는데 학습 안 된 클래스가 움직이면 포착하기 어렵다. 최근 MOS나 VOS방식은 외형과 움직임을 결합하지만 두 가지 문제가 있다. RGB 이미지를 그대로 쓰면 high-level semantic 정보를 포착하기 어렵다. raw pixel 값은 저 수준 정보이다. 두 modality를 별개 단게에서 처리한다. 예로 motion 먼저 mask추출 후 RGB로 mask 다듬는 식이다. 이 방식은 상호보완적 통합을 어렵게 한다.
- sefl-supervised 모델인 DINOv2에서 예측한 DINO feature를 통합하여 외형 정보를 일반화하는데 사용한다. 그러나 DINO feature를 단순히 입력으로 주면 모델이 의미 정보에 의존하게 되어 동적이거나 정적인 물체를 구분하는 능력이 떨어진다.
- 이 문제를 해결하기 위해 transformer 디코더가 semantic 단서를 고려하면서도 모션 정보의 우선순위를 정할 수 있도록 **Motion-Semantic Decoupled Embedding**을 제안한다. 
- Transformer 기반의 디코더는 모션 정보만 embedding 된 인코더에서 나온 trajectory 정보를 입력으로 받도록 디자인했다. Attention 가중치가 적용된 motion feature에 DINO feature를 concat 한 뒤 feed-forward layer로 전달한다. 디코더 layer에서는 self-attention이 motion feature에만 적용되지만, multi-head attention을 사용하여 의미론적 정보가 포함된 메모리에 attention한다. 마지막으로 sigmoid 함수를 적용하고 각 trajectory에 대한 예측 레이블을 도출한다. 
- 예측 레이블과 track 별 GT간 weighted binary cross-entropy loss를 사용한다. 샘플링된 point 좌표가 실제 동적 마스크 내에 있는지 판단하여 각 trajectories에서 dynamic(1), static(0)으로 loss를 계산한다.
- **3. SAM2 Iterative Prompting**: 각 trajectory를 예측하고 필터링한 후, SAM2의 point prompt로 활용하여 iterative two step을 적용한다. 첫 단계는 동일한 물체에 속하는 trajectory을 그룹화하고 각 물체의 trajectory를 메모리에 저장한다. 두번째 단계는 저장된 메모리를 다시 SAM2의 prompt로 활용하여 동적 mask를 생성한다.
- 접근 방식의 motivation은 SAM2가 입력으로 object ID를 요구하기 때문에 필요한 단계이다. 모든 움직이는 물체에 동일한 ID를 부여한다면 분할 성능이 감소한다.
- 첫 단계에서 visible point 수가 가장 많은 frame을 선택하고 이 frame에서 가장 dense한 위치의 point를 찾는다. 이 포인트는 SAM2의 초기 prompt역할을 하고 이후 초기 마스크를 생성한다. sparse 위치의 point는 경계나 노이즈일 가능성이 있다. 마스크 생성 후에는 경계를 확장하기 위해 dilation을 적용하고 확장된 마스크 영역 내의 모든 포인트를 제외하여 동일한 물체에 속한다고 가정한다. 다음으로 포인트의 수가 가장 많은 다음 frame으로 진행하고 다시 모든 frame에서 남아있는 포인트가 작아 처리할 수 없을 때까지 반복한다. 같은 물체에 속하는 것으로 식별된 trajectory는 각각에 고유한 객체 ID가 할당되어 메모리에 저장된다. 각 물체에 대해 확장 되기 전 point들만 저장한다.
- 두번째 단계에서 이 메모리들을 적용하여 prompt를 정제한다. 가장 dense한 곳의 point와 이 point와 가장 먼 두 개의 point를 찾아서 prompt로 넣어준다. SAM2가 중간에 물체 추적을 잃어버리지 않도록 일정 간격으로 prompt를 생성한다. SAM2가 물체의 부분적인(partial) mask를 생성할 수 있으므로, 모든 마스크에 후처리를 수행하여 내부적으로 겹치거나 동일 마스크 경계 내에 있는 마스크를 병합한다. 

### Limitation
- tracking 모델에 대한 의존도가 높다. SAM2 prompt의 interval 사이에 물체가 나타났다 사라질 경우 실패가능성이 높다. 더 뚜렷한 움직임을 보이는 물체가 있는 경우 다른 물체는 간과될 수 있다. SAM2의 prompt에 대한 의존도가 높다. 대부분의 물체가 유사한 동작 상태를 공유하는 경우, 개별 물체를 효과적으로 구분하지 못한다.


