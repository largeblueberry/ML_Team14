from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, preprocessor):
    """
    전처리 파이프라인과 모델을 결합하여 훈련시키고, 훈련된 모델을 반환합니다.
    
    Args:
        X_train: 훈련용 피처 데이터
        y_train: 훈련용 타겟 데이터
        preprocessor: 데이터 전처리기
        
    Returns:
        sklearn.pipeline.Pipeline: 훈련된 모델 파이프라인
    """
    print("⏳ [model_trainer] 모델 훈련 시작...")
    
    # RandomForest 모델 정의
    rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
    )

    # 전처리기와 모델을 파이프라인으로 연결
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', rf_classifier)])

    # 모델 훈련
    model_pipeline.fit(X_train, y_train)
    
    print("✅ [model_trainer] 모델 훈련 완료")
    return model_pipeline
