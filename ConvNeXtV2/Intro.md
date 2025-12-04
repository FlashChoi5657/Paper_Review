ConvNeXtV2


**Representation Learning은 AI 이전에도 존재했던 개념으로 PCA, 푸리에 transform, k-means clustering, SVM 등이 해당된다. 딥러닝 관점에서 feature의 quality를 결정하는 요소는 AI 아키텍쳐, 학습방법, 데이터이다.
- convolution은 수동 feature 엔지니어링에 의존하기보다는 다양한 시각적 인식 task에 대한 일반적인 feature 학습 방법을 사용할 수 있도록 함으로써 컴퓨터 비전 연구에 상당한 영향을 미쳤다. 최근에는 자연어 처리를 위해 개발된 transformer 학습 방법을 사용할 수 있도록 모델과 데이터셋 크기에 대한 강력한 확장 가능성으로 인해 주목받았다. 최근에는 ConvNeXt 아키텍처가 기존 ConvNet을 현대화 했고 순수 convolutional model도 확장 가능한 아키텍처가 될 수 있음을 증명했다. 
- Visual representation learning은 pretext 목적함수를 사용한 self-supervised 학습으로 이동하고 있다.
**downstream(진짜 과제)를 바로 풀 수 없다면 데이터에서 만들 수 있는 surrogate task를 설계한다. 예를 들어 original image에서 랜덤하게 0, 90, 180, 270도 회전시킨 이미지를 생성하고 4 classes 모델을 학습한다. colorization이라면 컬러이미지를 grayscale로 바꾸고 흑백이미지를 모델을 이용해 컬러 이미지로 SSL 방식으로 예측하도록 학습한다.  
- mask autoencoder(MAE)는 vision domain에서 masked language modeling의 성공을 가져왔고 빠르게 시각적 표현 학습을 위한 대중적 접근방식이 되었다. self-supervised learning(SSL)의 일반적 방식은 학습을 위해 미리 정의된 아키텍처를 사용하고 디자인은 고정되었다고 가정한다.
**MAE: 이미지의 75% patch로 가림, MAE를 이용해 이미지 복원. 이는 가려진 부분 복원을 위해 이미지의 의미를 이해해야 하고 좋은 representation 학습으로 이어진다. 인코더는 unmasked token만(visible patch) 추출해 ViT encoder에 넣고 latent vector를 생성한다. 디코더는 encoder output과 positional embedding된 mask token으로 학습하고 전체 patch를 복원한다. Loss는 MSE(복원된 masked patches와 원본). 학습 후에 Encoder만 사용한다.
- 아키텍처와 self-supervised learning 프레임워크를 결합하는 것은 가능하지만 MAE와 ConvNeXt에서는 문제가 생길 수 있다. 그 중 하나로 MAE는 transformer의 시퀀스 처리 능력에 최적화된 특정 인코더-디코더 디자인을 가지고 있다는 것이다. 계산량이 많은 인코더는 visible patch에 집중하고 사전학습비용을 줄일 수 있다. 이 디자인은 ConvNet이 이용하는 dense sliding window 방식과 양립하기 어렵다. 또한 아키텍처와 목적 함수 간의 관계를 고려하지 않으면 최적화도 어렵다. 이전의 연구들에서 ConvNet은 mask 기반의 self-supervised 학습이 어렵고 transformer와 다른 feature 학습을 한다는 것이 알려졌다. 
- 이러한 문제를 해결하기 위해 마스크 기반 SSL을 효과적으로 만들고 transformer를 사용하여 얻은 결과와 경쟁가능한 ConvNeXt 모델을 적용해 MAE와 공동 설계할 것을 제안한다.
MAE를 설계할 때 마스킹된 입력을 sparse patch set으로 처리하고 sparse convolution을 사용하여 보이는 부분만 처리한다. 이는 대규모 3D 포인트 클라우드를 처리할 때 sparse convolution을 사용하는 것과 유사하다. 
**LiDAR나 depth sensor로 얻는 3D point cloud는 극도로 sparse 하다(전체 voxel 중 실제 데이터는 1% 미만). sparse convolution은 모든 위치에서 연산하는 일반 conv와 다르게 occupied voxel 위치에서만 연산하고 저장도 sparse하게 저장한다. Masked Auto Encoder 도 sparse 고 볼 수 있고 visible patch만 연산하도록 하는 것이다. 
- 실제로 sparse convolution을 이용하여 ConvNeXt를 구현할 수 있고 fine-tuning에서 가중치는 standard로 변환하면 추가적인 조작 없이 dense layer가 된다. 사전 학습 효율성 향상을 위해 transformer 디코더를 단일 ConvNeXt 블록으로 대체하여 전체를 fully convolutional로 만든다. 학습된 feature는 기본 결과를 개선하지만 fine-tuning성능은 transformer기반 모델보다는 안좋다.
- ConvNeXt의 다양한 학습 구성의 feature space 분석을 하였다. 마스킹된 입력에서 ConvNeXt를 직접 학습할 때 MLP layer에서 feature collapse의 잠재적 문제를 발견했다. 
**Feature Collapse분석: feature matrix의 SVD를 수행해서 signular value의 분포확인. collapse가 심할수록 effective rank가 낮아지고 variance와 std가 대부분 0에 수렴하여 죽은 dimension이 된다. cosine similarity 분석 시 서로 다른 feature간 값이 1에 수렴한다.(=대부분 비슷). Representation leraning 관점에서 SSL 도중 feature collapse는 label 없이 학습하기 때문에 모델이 trivial solution을 찾기 쉽다.
- 문제 개선을 위해 채널 간 feature 경쟁을 강화하는 수단으로 Global Response Normalization layer를 추가로 제안. 모델이 MAE로 사전 학습될 때 효과적이다.
- Masking: 60% random 마스킹 활용. 여러 단계의 downsampling되는 계층적 디자인을 가지므로 마스크는 인코더의 마지막 단계에서 생성되고 첫 단계까지 재귀적으로 upsampling한다. 
- Encoder design: 인코더로 ConvNeXt 적용. ViT와 다르게 2차원 구조를 유지해야하는 CNN 계열 모델에서는 masking 영역에서 정보가 leak될 수 있다. 저자의 관찰은 마스킹된 이미지가 2D sparse 픽셀 배열로 표현될 수 있다는 것. 채워야 할 공간으로 취급하는 것이 아니라 애초에 데이터가 없는 sparse data로 처리하기 위해서 sparse conv 적용.
- Decoder design: plain(단일) ConvNeXt block 이용.
- Global Response Normalization: feature collapse 이슈 해결
