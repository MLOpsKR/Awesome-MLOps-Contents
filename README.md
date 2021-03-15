# Awesome-MLOps-Contents
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMLOpsKR%2FAwesome-MLOps-Contents)](https://hits.seeyoufarm.com)
- DataOps(Data Operation), MLOps(Machine Learning Operation) Contents
- [MLOps KR](https://www.facebook.com/groups/MLOpsKR/)에서 공유된 자료를 모아둡니다


- 20.01.19에 최초 작성되었으며, 아직 초안입니다
	- 자료를 꾸준히 아카이빙할 예정이니 관심있으시면 Watch를 눌러두시면 좋을 것 같습니다 :)
	- 직접 Pull Request로 좋은 자료를 추가해주셔도 좋고, 원하시는 내용이 있으시면 Issue에 등록해주세요 :)


---

## Contents
- [Basic](#basic)
- [Serving](#serving)
- [Feature Store](#feature-store)
- [Experiment](#experiment)
- [AutoML](#automl)
- [Data Validation](#data-validation)
- [Hyper Paramter Tuning](#hyper-parameter-tuning)
- [Kubeflow](#kubeflow)
- [Conference Review](#conference-review)
- [Company Use Case & Presentation](#company-use-case-&-presentation)


### Basic
- Mercari의 [머신러닝 시스템 디자인 패턴(번역본)](https://mercari.github.io/ml-system-design-pattern/README_ko.html)
- [Why is DevOps for Machine Learning so Different?](https://hackernoon.com/why-is-devops-for-machine-learning-so-different-384z32f1)(Eng)
	- MLOps와 DevOps의 차이에 대해 잘 작성된 글
- [Move Fast and Break Things? The AI Governance Dilemma](https://hackernoon.com/move-fast-and-break-things-the-ai-governance-dilemma-dsq32ix)(Eng)
- [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)(Eng)
	- 구글의 머신러닝 엔지니어링 가이드 글로, 모델을 적용하는 과정에 대해 잘 나와있음
	- 번역 글 : [Rules of Machine Learning: Best Practices for ML Engineering 정리](https://zzsza.github.io/data/2019/12/15/rules-of-ml/)(Kor)
- [Awesome production machine learning](https://github.com/EthicalML/awesome-production-machine-learning)(Eng)
	- 매우 다양한 오픈소스 라이브러리를 테마별로 모아둔 Github Repository. [EthicalML](https://github.com/EthicalML)의 다른 Repository에도 좋은 자료가 많음
- [머신러닝 오퍼레이션 자동화, MLOps](https://zzsza.github.io/mlops/2018/12/28/mlops/)(Kor)
- [Deep-Learning-in-Production](https://github.com/ahkarami/Deep-Learning-in-Production)(Eng)
	- 딥러닝 Production과 관련한 Repository. PyTorch, TensorFlow, MXNet, Mobile Development, Back-End 등에 대한 자료를 모아둠
- [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/march2019)(Eng)
	- 딥러닝 전반 ~ 프러덕션 전반까지 잘 알려주는 강의
	- 강의 정리 글 : [Full Stack Deep Learning Bootcamp 정리](https://zzsza.github.io/mlops/2019/10/06/fullstack-deeplearning-bootcamp/)(Kor)
	- 후기 글 : [The 7 questions you need to ask to operate deep learning infrastructure at scale](https://jameskle.com/writes/deep-learning-infrastructure-tooling)(Eng)
- [Production-Level-Deep-Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)(Eng)
	- Full Stack Deep Learning Bootcamp을 기반으로 추가적인 자료를 모아둔 Repo
- [Machine Learning Systems](https://ucbrise.github.io/cs294-ai-sys-fa19/)(Eng)
	- UC Berkeley 수업 자료로 동영상은 없지만 자료만 봐도 유익
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf)(Eng)
- [MLOps Done Right](https://towardsdatascience.com/mlops-done-right-47cec1dbfc8d)(Eng)
- [CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html) : 스탠포드에서 만든 자료로 동영상은 거의 없지만 발표 자료만 봐도 유익함
	
### Serving
- TBU(너무 많아서 추후에 더 올릴 예정입니다)
- [TF Serving -Auto Wrap your TF or Keras model & Deploy it with a production-grade GRPC Interface](https://towardsdatascience.com/using-tensorflow-serving-grpc-38a722451064)(Eng)


### Feature Store
- [Feature Stores for ML](http://featurestore.org/)(Eng)
	- 각종 Feature Store를 비교하고, 관련 영상이 존재
- Feature Store에 대한 설명 글
	- [Feature Stores: Components of a Data Science Factory](https://towardsdatascience.com/feature-stores-components-of-a-data-science-factory-f0f1f73d39b8)(Eng)
	- [Rethinking Feature Stores](https://medium.com/@changshe/rethinking-feature-stores-74963c2596f0)(Eng)
- Feature Store 라이브러리
	- Logicalclocks의 [Hopsworks](https://github.com/logicalclocks/hopsworks)
)
	- Gojek의 [feast](https://github.com/gojek/feast)
		- [introducing feast an open source feature store for machine learning](https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning)(Eng)

### Experiment
- [머신러닝 실험을 도와줄 Python Sacred 소개 ](https://zzsza.github.io/mlops/2019/07/21/python-sacred/)(Kor)
- [Sacred와 Omniboard를 활용한 실험 및 로그 모니터링](https://zzsza.github.io/mlops/2019/07/22/sacred-with-omniboard/)(Kor)



### AutoML
- [Microsoft NNI](https://github.com/microsoft/nni)(Eng)


### Data Validation
- [Tensorflow Data Validation 사용하기 ](https://zzsza.github.io/mlops/2019/05/12/tensorflow-data-validation-basic/)(Kor)
- [Amazon의 Deequ](https://github.com/awslabs/deequ)(Eng)



### Hyper Parameter Tuning 
- TBU

### Kubeflow
- [End to End Pipeline : KubeFlow +Keras/TensorFlow2 +TF Extended (TFX) +Kubernetes +PyTorch +XGBoost +Airflow +MLflow +Spark](https://www.youtube.com/watch?v=OhIa2cnGD8Y)(Eng) : 6시간 실습 영상
	- [슬라이드 자료](https://www.slideshare.net/cfregly/handson-learning-with-kubeflow-kerastensorflow-20-tf-extended-tfx-kubernetes-pytorch-xgboost-airflow-mlflow-spark-jupyter-tpu)
- [Kubeflow 소개와 활용법](https://youtu.be/szygR7G3ZY8)(Kor) : 두다지에서 어떻게 활용하고 있는지 발표해주신 자료
- [Kubeflow Handson Kubeflow](https://www.youtube.com/watch?v=cFXplM3IdyI) : 핸즈온 실습 영상 
- [딥러닝 추천 시스템 in production](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-in-production-fa623877e56a)(Kor) : 당근마켓에서 Kubeflow pipeline을 활용한 사례 글




### Conference Review
- [MLOps NYC19 Conference 정리](https://zzsza.github.io/mlops/2019/10/27/mlops-nyc19-review/)(Kor)



### Company Use Case
- 당근마켓
	- [딥러닝 추천 시스템 in production](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-in-production-fa623877e56a)(Kor)
- Spotify
	- [The Winding Road to Better Machine Learning Infrastructure Through Tensorflow Extended and Kubeflow](https://labs.spotify.com/2019/12/13/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow)(Eng)
