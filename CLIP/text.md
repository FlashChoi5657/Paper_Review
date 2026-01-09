## Contrastive Languange-Image Pre-training, CLIP

### 기본 개념
- Contrastive Learning: 같은 의미의 text-image pair 가깝게, 다른 의미의 pair는 멀게
- Zero-shiot 학습: CLIP는 특정 task에 대해 별도 학습 없이 사전 학습된 데이터만으로 강력한 성능
### Architecture
- Vision encoder: CNN 또는 ViT 기반, 입력 이미지의 feature 추출
- Text encoder: Transformer 기반, 임베딩 벡터로 변환
- Image와 text를 같은 차원의 벡터 공간에 매핑한 후, 의미가 비슷한 경우 벡터 거리가 가까워지도록 학습
### 학습 방식
- Contrastive loss, 한 batch에 여러 개의 (T-I) pair 포함
- 같은 pair 유사도 점수 최대화, 다른 pair 유사도 점수 최소화

### 배경 & 기여
- 기존의 CV 모델들은 supervision 방식으로 학습했다. 이는 generalit와 usability를 한정짓는다. 대안으로 Raw text로부터 바로 이미지를 러닝할 수 있다.
- Text-image pair를 이용해 image representation 학습을 한다.
- Multi-modal 분야는 NLP와 CV의 모델 발전과 함께 발전함. 그러나 자연어를 이미지 representation learning에 사용하는 것은 드물다. 성능이 낮기 때문
- 이전 연구들은 제한된 라벨링 데이터셋으로 학습하는 것과 un-labeling 데이터로 학습하는 것 사이의 위치를 보여준다. 두 방법 모두 static softmax classifiers를 예측에 사용해서 동적인 output 부족. 이로 인해 zero-shot 능력도 제한됨.
- CLIP 모델은 weakly supervised approach와 zero-shot learning using raw text 간의 차이를 감소시킬 수 있다. 

