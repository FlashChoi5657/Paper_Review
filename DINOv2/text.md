## DINOv2 : Learning Robust Visual Features without Supervision

### Abstract
**Background and objectivie**: 대규모 비지도학습(SSL)로 얻은 visual feature가 general downstream(cls, det, retrieval)에 매우 중요하다. 목표는 Annotation 없이 robust and versatile feature 학습. 다양한 downstream task에서 높은 성능. Low-shot, linear probe, K-NN 설정에 robustness.
**Contribution**: ViT와 결합한 clustering/teacher-studnet 학습 구조 설계. 이미지를 다양한 scale로 자르고 pixel and patch level feature 일관성을 유지하도록 학습. EMA기반 teacher+centering으로 안정적인 학습. feature가 semantic + geometric 정보를 모두 포착하는 목표
**Methodology**: ViT backbone, teacher(EMA로 업데이트)-student 구조. Global representation을 학습하는 목표(이미지의 global feature 일관성 유지, teacher-student 간 유사성 최대화), Local/Dense Matching(다양한 view, crop 간 patch2patch 일치, crop은 크기에 따라 Large는 전체 semantic 정보 학습, small은 local detail 정보 보존)
1. Self-distillation 구조에서 teacher모델이 학습 대상의 soft target을 일정하게 유지하여 contrast negative없이 representation collapsing을 방지한다. 
2. 목적함수에서 Global feature는 semantic 정보만 파악하고 patch, geometry 정보가 손실 될 수 있어 dense matching으로 구조정보를 보존한다. 
3. ViT는 attention 구조가 이미지 내 다양한 관계를 잘 포착하고 dense matching과도 결합이 잘된다.

### Introduction
- 특정 task에 종속되지 않는 사전학습 표현을 학습하는 것은 NLP의 표준이 되었다. 이 학습 방식은 fine-tuning 없이 feature들을 '있는 그대로' 사용할 수 있고 downstream task에서 task-specific 모델보다 더 우수한 성능을 달성할 수 있다. 이러한 성공은 언어 모델링 또는 워드 벡터와 같은 감독이 필요 없는 pretext objetivis를 사용한 대량의 raw text로 사전학습함으로써 이루어졌다.
- NLP의 이러한 패러다임 전환처럼 CV에서도 유사한 foundation 모델의 출현을 기대한다. foundation 모델이라 함은 이미지의 분류 또는 분할 task에서 별도 설정없이 바로 동작하여 시각적 feature를 생성할 수 있어야 한다. 이 분야에서 성공이 가장 기대되는 방식은 텍스트가 보조하는 사전학습(text-guided pretraining)이다. 그러나 형태의 텍스트 보조 사전학습 형식은 caption이 이미지의 풍부한 정보를 대략적으로 표현하고 복잡한 pixel-level 정보는 이러한 supervision에서 드러나지 않게 만들어 이미지에서 보존될 수 있는 정보를 제한한다(CLIP 같은 방식이 caption이 고양이가 소파에 앉아있다 정도의 high level 의미만 담고 고양이의 texture, edge 등의 fine-grained detail은 담지 못하는 한계를 지적한다). 이러한 이미지 인코더들은 텍스트와 이미지 뭉치의 정렬도 필요하고 raw data로만 학습 가능한 텍스트 쪽 언어 모델이 갖는 유연성을 제공하지 못한다.
- 텍스트 보조 사전학습을 대체하는 것은 feature를 이미지로만 학습하는 자기지도 학습이다. 이러한 접근은 언어모델의 pretext task와 개념적으로 유사하고 이미지와 pixel level에서 정보를 기반으로 학습한다. 게다가 자기지도 모델로 생성한 feature 출력은 다양하고 유용한 특성을 보여주고 다양한 분야에 적용도 가능하다. 이러한 잠재력에도 불구하고 그 동안은 ImageNet-1K와 같은 small curated 데이터에 대한 사전학습 맥락에서만 이루어졌다. ImageNet-1K 규모의 한계를 넘어서기 위한 시도가 있었으나 정제되지 않은 데이터로 접근했고 feature quality가 하락했다. 이는 우수한 feature를 만들기 위해 데이터의 quality와 diversity를 제어하는 것이 중요하다는 것을 보여준다.
- 논문에서는 자기지도학습이 광범위한 정제 데이터에서 사전학습 된 경우 다목적 visual feature를 만들 잠재력이 있는지 탐색한다. 현존하는 판별형 자기지도 학습 접근 방법인 iBOT과 같은 방식의 이미지와 patch단위에서 feature를 학습하는 방식을 재검토하고 그들의 설계 선택을 더 큰 규모의 데이터셋을 기준으로 재고해 본다. 본 논문의 대부분 기술적 기여는 판별형 자기지도 학습에서 모델과 데이터 크기를 키울 때 학습을 안정화, 가속화 하는 것이다. 이러한 개선은 유사한 방법론보다 2배 더 빠르고 3배 더 적은 메모리를 사용하여 더 큰 batchsize를 적용해 더 긴 학습을 진행할 수 있게 한다.
- 사전학습 데이터를 만들기 위해 광범위한 미정제 데이터에서 필터링과 재조정을 할 수 있는 자동화 라인을 구축했다. 이것은 NLP 파이프라인과 유사한데, 외부의 메타데이터 대신 데이터 유사도를 사용하며 추가적인 수작업이 필요없다. 광범위한 데이터를 수집하면서 특정 카테고리의 이미지가 압도적으로 많기 때문에 이것을 재조정하고 몇몇의 분포에 과적합 되는 것을 피하는 작업이 어려웠다. 논문에서는 naive clustering 접근법이 이 이슈를 합리적으로 해결하는데 합리적으로 잘 동작시킨다. 우리의 접근을 검증하기 위해 작지만 다양한 종류의 1억4천2백만개의 이미지를 수집했다.
- 마지막으로 DINOv2라 명칭한 자체 데이터로 학습한 다양한 ViT 구조 기반의 사전학습된 visual model들을 제공한다. 모든 모델과 코드를 어떤 데이터에도 적용할 수 있도록 배포한다. 그림2에서 요약한 것처럼 DINOv2는 다양한 CV 벤치마크의 이미지와 픽셀 단위에서 모델을 스케일링 할 수록 훌륭한 성능을 보이는 것을 검증했다. 자기지도 학습에서는 크게 앞서고 약지도 학습과는 근사하게 경쟁할 수 있는 transfer 가능한 고정 feature를 학습하기에 좋은 후보라 결론 짓는다.

### Related Work
- Intra-image self-supervised training(단일 이미지에서 생성한 신호를 이용한 자기지도학습): 자기지도학습 방법의 첫 번째 시도는 이미지로 만들어진 pretext task에 초점을 맞춘다. 즉, 이미지로 학습 신호를 추출하고 이를 이미지의 나머지 부분으로부터 예측하도록 학습한다(이미지의 일부를 맞춰보는 task). Doersch등은 특정 패치의 문맥을 예측하는 방식으로 모델을 학습시켰다. 이후 이미지 re-colorizing, 변환 예측, inpainting 또는 패치 재정렬 등 다양한 pretext task가 제안되었다. 최근에는 ViT와 같은 patch 기반 아키텍처의 등장으로 사전학습 단계에서 inpainting을 다시 활용하는 연구가 활발해졌고 경우에 따라 pixel space가 아닌 feature space에서 수행되기도 한다. 특히, he 등은 MAE가 downstream task에 대해 fine-tuning할 경우 상당한 성능 향상을 제공하는 특징 표현을 학습한다는 것을 보였다. 이러한 MAE의 특성은 비디오, 오디오, 또는 다양한 modality에서 검증되었다. 하지만 이러한 방법들이 학습한 특징들은 supervised learning 기반 fine-tuning이 필요하며 우리 방식은 별도의 fine-tuning 없이 우수한 성능을 보인다.
- Discriminative self-supervised learning: 우리 연구와 더 가까운 두 번째 항목은 이미지와 이미지 그룹사이의 판별형 신호를 사용하여 feature를 학습하는 것이다. 초기 딥러닝 방식의 한 형태였으나 즉각 분류 방식이 대두되며 유명해졌다. 여러 개선은 인스턴스 수준 목표 또는 군집화를 기반으로 이뤄졌다. 이 방식들은 이미지넷에서 획득한 성능 좋은 feature들을 제공하지만 더 큰 모델 사이즈로 만들기는 어렵다. 이번 연구에서 이 방식들을 거대 사전 학습용 데이터셋과 모델 관점에서 재검토 해본다. 
- scaling self-supervised pretraining: 데이터와 모델 크기 측면에서 SSL의 크기 증가에 초점을 맞추는 연구들이 점점 많아지고 있다. 그러나 대부분 연구들이 정렬되지 않은 양만 많은 데이터를 사용하고 있다. 판별형 방식의 데이터 스케일링 증거를 보여주고 있지만 사전학습용 데이터의 품질이 낮기 때문에 대부분의 결과는 feature를 fine-tuning해서 얻은 것이다.
- Automatic data curation: 연구에 사용한 데이터셋 구축은 이미지 검색 분야의 방법론(쿼리 이미지와 유사한 이미지를 데이터 베이스에서 검색)을 차용했다. 검색을 데이터 증강에 사용하는 것은 자기지도학습에서 데이터 증강하는 방법이다. 해쉬태그 또는 다른 metadata 또는 사전학습된 vision encoder를 정렬되지 않은 데이터를 필터링하는데 사용했다. 이런 방식들과 다르게 연구에서는 사전학습된 encoder, metadata, 감독을 사용하지 않고 이미지 간의 visual 유사도를 활용하여 이미지를 필터링 했다. 이러한 접근은 위키피디아 텍스트를 스코어로 정렬하여 학습한 언어 모델의 텍스트 정렬 파이프라인에서 온 것이다.

### Data Processing
- 정렬된 여러 데이터와 유사한 가공되지 않은 이미지를 대규모 pool에서 검색하여 LVD-142M 데이터셋을 모았다. 연구의 파이파라인은 어떤 metadata나 텍스트가 필요하지 않고 이미지와 직접 작동한다. 
- Data sources: 연구 데이터셋은 ImageNet-22K, ImageNet-1K의 학습split, 구글Landmark와 몇몇 상세 정제된 데이터셋을 포함한다. 가공되지 않은 데이터셋 생성은 웹에서 크롤링한 데이터를 모아둔 공개 저장소에서 raw 이미지데이터를 수집했다. 안전하지 않거나 제한된 도메인을 제외하고 다운로드한 이미지를 후처리했다(PCA 해쉬 중복제거, NSFW 필터링, 식별가능한 얼굴 블러링). 12억개의 고유 이미지를 생성했다.
- Deduplication: 가공되지 않은 데이터에 Pizzi 방식을 적용하여 유사-중복 이미지를 제거한다. 이미지들 간의 유사성을 제거하고 다양성을 증가시킬 수 있다. 이 작업은 검증과 테스트셋에서도 동일하게 수행한다.
- Self-supervised image retrieval: 연구의 사전학습용 데이터셋은 curated source(상세 정제된 데이터셋)의 이미지들과 가까운 이미지들을 가공되지 않은 데이터 소스에서 검색하여 구축했다. 이를 위해, ImageNet-22K로 사전학습된 ViT-H/16 신경망으로 이미지 임베딩을 연산하고 이미지 간의 cosine-유사도로 거리를 구한다. 그 다음 선별되지 않은 데이터는 k-means 군집화를 시도한다. 검색을 위한 query 데이터셋이 충분히 커지면 각 query 이미지에 대해 N개(보통 4개)의 nearest neighbor를 검색한다. 작은 경우에는 각 query 이미지와 관련된 집합에서 M개의 이미지를 샘플링한다. 육안으로 검사했을 때 4보다 더 큰 N도 좋아보였지만, 중복과 같은 더 많은 충돌을 만들었다. N=4는 그런 맥락에서 정했다.
- Implementation Details: 연구 파이프라인의 중복제거와 검색 단계는 Faiss 라이브러리에 의존하며 검색과 최근접 임베딩 batch  연산을 효율적으로 수행한다. 특히, product quantization(벡터를 압축해서 메모리/속도 효율을 높이는 기법) 코드를 사용한 inverted file indices(검색 속도를 높이는 인덱싱 방식)를 활용하여 GPU 가속화를 적극적으로 활용했고 전체 처리과정은 8개의 V100이 장착된 20개 노드의 컴퓨팅 클러스터에 분산되어 수행되며, 이틀이 걸렸다.

### Discriminative Self-supervised Pre-training
- SwAV를 중심으로 DINO와 iBOT 손실함수들의 조합을 적용해 판별형 자기지도학습으로 feature를 학습시킨다. Regularizer를 추가하여 feature와 짧은 고해상도의 학습 phase를 확장한다. 
- **SwAV**(Swapping Assignments between Views): Negative pair없이 View 사이의 클러스터 할당 일치가 목표이다. 고정된 라벨 없이, 학습 중에 feature를 prototype에 할당하고 View A의 feature는 View B의 클러스터 할당을 예측한다. clustering 붕괴 방지를 위해 Sinkhorn-Knopp(Batch 단위 제약 알고리즘)를 사용한다.
- **iBOT**(Image BERT Pre-Training with Online Tokenizer): ViT 기반 masked image modeling에 self-distillation을 결합하여 패치 단위의 표현을 학습한다. 이미지를 patch 토큰화 하고 일부를 임의로 마스킹한다. studnet가 마스킹 된 patch의 토큰 분포를 예측하고 teacher(EMA)가 제공한 soft label에 맞춰 학습한다. 
- Image-level objective: student-teacher 네트워크에서 추출된 feature들 간의 cross-entropy loss를 고려한다. 두 feature는 모두 ViT의 클래스 토큰에서 추출되는데, 같은 이미지의 다른 결과를 포함한다. student class 토큰은 "prototype scores"라 부르는 벡터 스코어를 출력하는 MLP모델로 구성된 DINO head에 통과시킨다. 그리고 출력은 softmax($p_s$)까지 통과시킨다. 유사한 관점에서 teacher 클래스 토큰은 teacher DINO head에 통과시켜 prototype score를 얻고 softmax를 적용한 뒤 teacher 모델의 출력 분포 편향을 막기 위해 moving average로 보정(centering)을 추가한다. 정리하면 student의 출력분포$p_s$가 teacher가 의도적으로 설계한 target 분포 $p_t$를 잘 따르고 있는지 측정하는 손실함수로 cross entropy를 선택했다.
- patch-level objective: Student에 입력하는 patch는 임의로 마스킹하지만 teacher 경우에는 하지 않는다. 각 마스킹된 patch에서 두 네으워크의 patch feature 사이의 cross-entropy loss를 추가한다. 
- Untying head weights between both objectives: ViT backbone은 공유하고 backbone에서 얻은 token 출력에 대해 DINO objective와 iBOT objective 각각에 대응하는 서로 다른 MLP projection head를 적용한 뒤, 각각의 출력으로 DINO loss와 iBOT loss를 계산했다. 
- Sinkhorn-Knopp centering: teacher softmax-centering 스텝을 S-K batch 정규화로 대체하였다. 3회 반복하였고 student model에서는 softmax 정규화를 하였다.
- KoLeo regularizer: KoLeo regularizer는 Kozachenko-Leonenko 미분 엔트로피 추정량에서 유도된 것으로, batch 내 feature들이 특정 영역에 몰리지 않고 representation space 전반에 고르게 퍼지도록 유도한다. 다시 정리하면 batch 내 feature들의 nearest neighbor distance기반으로 표현 분포의 엔트로피를 높이는 방향으로 작동한다. regularizer는 가장 가까운 이웃으로부터 멀어지도록 유도한다.

