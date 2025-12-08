Segment Any Motion in Videos

- 움직이는 물체의 Segmentation이 어려운 이유는 복잡한 형태, motion blur, 배경에 의한 왜곡이다. 그래서 DINO-based 의미론적 feature와 SAM2의 pixel-level 정밀한 mask를 이용한다.
- **DINO(=self-Distillation with No Labels): Vision Transformer를 라벨 없이 학습시키는데 teacher-student 구조로 같은 이미지의 다른 augmentation을 보고 일관된 representation을 학습한다. DINO로 학습된 ViT의 특징은 명시적인 supervision 없이도 semantic 정보를 담고 있다. 예로 attention map을 시각화 하면 object의 경계나 의미적으로 유사한 부분들이 자연스럽게 클러스트링 된다. 이미지 자체 구조에서 학습신호를 만든다. 예를 들어, 같은 이미지를 두 가지 다른 방식의 증강을 하면 global view와 local view가 생기는데 모델은 이 둘이 같은 이미지에서 왔으니 비슷한 representation을 학습하라고 시킨다. teacher network는 global view를 보고 student network는 local view를 본다. student는 teacher의 output을 따라가도록 학습하고 teacher는 student의 Exponential moving average로 업데이트 된다. 결과적으로 고양이 귀의 일부분만 봐도 고양이 전체와 관련있다는 것을 학습할 수 있다.**
- Point tracking은 deformation과 occulusion에 강인한 장기적 pixel motion 정보를 포착(exploit)한다. 여기에 의미론적 context를 추가하기 위해 DINO feature를 추가한다. 긴 2D track 셋이 주어지면 모델은 움직이는 물체에 해당하는 track을 식별한다. 이러한 sparse point들을 prompt로 주고 dense pixel-level mask로 확장한다.
- **정리하면, 뭐가 움직이는 물체인지 판단하고 SAM2는 그 판단 결과를 정밀한 segmentation mask로 만들어준다.**
- motion 단서와 semantiv 정보를 균형있게 조절하기 위해 두 가지 모듈을 제안한다.
- **시공간적(Spatio-Temporal) 추적 어텐션은 주어진 입력 track의 long-term 특성을 고려하여 서로 다른 궤적간의 관계를 파악하는 spatial attention과 하나의 궤적이 시간에 따라 어떻게 변하는지 파악하는 temporal attention을 통합한다.**
- **Motion-Semantic Decoupled Embedding은 모션 패턴의 우선순위를 정하고 보조경로에서 semantic feature를 처리하는 특수 attention 메커니즘**
- 합성데이터와 실제 데이터를 포함한 광범위한 데이터셋으로 모델을 학습했다. DINO feature의 self-supervised 특성은 강력한 일반화 성능을 보여준다.
- 기존 Moving Object Segmentation(MOS)는 optical flow를 활용했는데 optical flow는 단기적 motion에만 국한 되어 long-term에서는 파악이 어렵다. 또한 Spectral clustering은 point trajectory를 사용하는데 복잡한 모션을 처리하는데 어려움을 겪는다. 
- **Optical Flow는 비디오에서 픽셀들이 프레임 간에 어디로 이동했는지 나타내는 2D vector field이다. 예로, 프레임1에서 프레임2로 넘어갈 때 어떤 픽셀이 x+3, y-2로 움직였다를 계산한다. 결과물은 이미지와 같은 크기의 (dx, dy) 벡터맵이다.
- 다른 방법들은 motion cue와 외형 정보를 함께 사용하는데, 두 개를 별개의 단계에서 순차적으로 처리한다. 이 방법은 두 정보를 보완적으로 사용하는 효과를 제한한다. 
- 위 단점들의 보완을 위해 motion(point tracking)과 semantic(DINO feature)를 high level에서 통합한다.
- motion label은 trajectory가 움직이는 물체에 속하는지 아닌지를 나타내는 이진분류 맵이다. 여기서는 기존처럼 affinity matrix와 spectral clustering을 쓰는 대신 motion-semantic decoupled 임베딩 방법을 제안한다. motion 정보와 semantic 정보를 따로 임베딩했다가 결합한다.
