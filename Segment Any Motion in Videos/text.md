Segment Any Motion in Videos

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
-**Motion Pattern Encoding**: Point trajectories 모션을 이해하는데 중요한 정보를 제공하며 MOS 기법은 2프레임과 multi-frame의 두가지로 나눌 수 있다. 2-frame은 시간 불일치가 심하고 입력 flow에 노이즈가 많을 때 성능이 저하된다. multi-frame은 노이즈에 매우 민감하고 복잡한 패턴을 처리하기 어렵다. 이러한 한계를 해결하기 위해 특수 궤적 처리 모델을 통해 처리된 long-range point track을 활용하여 궤젹 별 모션 label을 예측하는 방법을 제안한다. 제안된 신경망은 encoder-decoder 구조이며 encoder는 long-range trajectory를 처리하고, 전체 궤적을 시공간 궤적 attention에 적용한다. 각 궤적의 Motion pattern을 임베딩 하기 위해 시간과 공간 단서를 통합하며 local and global 정보를 시공간에서 포착한다. 
