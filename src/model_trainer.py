# [변경점 1] imbalanced-learn의 Pipeline과 SMOTE를 import 합니다.
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, preprocessor):
    """
    전처리, SMOTE 오버샘플링, 모델을 파이프라인으로 결합하여 훈련시킵니다.
    
    Args:
        X_train: 훈련용 피처 데이터
        y_train: 훈련용 타겟 데이터
        preprocessor: 데이터 전처리기
        
    Returns:
        imblearn.pipeline.Pipeline: 훈련된 모델 파이프라인
    """
    print("⏳ [model_trainer] SMOTE 포함 모델 훈련 시작...")
    
    # RandomForest 모델 정의
    # [변경점 2] SMOTE로 데이터 불균형을 직접 해소하므로, class_weight='balanced' 옵션은 제거합니다.
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # [변경점 3] 전처리기, SMOTE, 모델을 파이프라인으로 연결합니다.
    # 파이프라인은 각 단계를 순서대로 실행합니다.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)), # 전처리 후 SMOTE 적용
        ('classifier', rf_classifier)      # SMOTE 적용된 데이터로 모델 학습
    ])

    # 모델 훈련 (파이프라인이 알아서 SMOTE를 훈련 데이터에만 적용합니다)
    model_pipeline.fit(X_train, y_train)
    
    print("✅ [model_trainer] 모델 훈련 완료")
    return model_pipeline